from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(0)
import sys
from torch import optim
from tensorboardX import SummaryWriter
from models.data import Dataset
# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import warnings
from IPython.core.display import display, HTML

import os
from models.utils import EMA, set_requires_grad, init_weights, update_moving_average, loss_fn, repeat_1d_tensor, currentTime
import copy
import pandas as pd

from models.embedder import embedder
from models.utils import config2string
from models.embedder import Encoder
import faiss


from sklearn.preprocessing import StandardScaler, MinMaxScaler
import models.utils
from models.momentum import knowledge_distillation_loss

import matplotlib.pyplot as plt

    
class MoDET_ModelTrainer(embedder):
    
    def __init__(self, args):
        embedder.__init__(self, args)
        self._args = args
        self._init()
        self.config_str = config2string(args)
        print("\n[Config] {}\n".format(self.config_str))
        self.writer = SummaryWriter(log_dir="runs/{}".format(self.config_str))
        #######################
        self.train_losses = []
        ##################################
    def _init(self):
        args = self._args
        self._task = args.task
        print("Downstream Task : {}".format(self._task))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self._device = "cpu"
        self._dataset = Dataset(root=args.root, dataset=args.dataset)
        self._loader = DataLoader(dataset=self._dataset)
        layers = [self._dataset.data.x.shape[1]] + self.hidden_layers
        self.temperature = 2.5
        self.momentum_buffer = None
        self.alpha =0.5# 初始化 alpha
        self.beta = 0.9
        self._model = MoDET(layers, args, temperature=self.temperature, momentum_buffer=self.momentum_buffer, beta=self.beta, alpha=self.alpha).to(self._device)
        self._optimizer = optim.AdamW(params=self._model.parameters(), lr=args.lr, weight_decay=1e-5)
       
   

   
  

        
    def train(self):
        self.best_test_acc, self.best_dev_acc, self.best_test_std, self.best_dev_std, self.best_epoch, self.best_ari = 0, 0, 0, 0, 0, 0
        self.best_dev_accs = []
        self.best_embeddings = None
        sillog = []
        momentum_buffer = None
    # get Random Initial accuracy
        self.infer_embeddings(0)
        print("initial accuracy ")
        self.evaluate(self._task, 0, sillog)

        f_final = open("results/{}.txt".format(self._args.embedder), "a")

    # Start Model Training
        print("Training Start!")
        self._model.train()
        for epoch in range(self._args.epochs):
            for bc, batch_data in enumerate(self._loader):
                epoch_loss = 0
                batch_data.to(self._device)
                last_batch_data=batch_data
                emb, loss= self._model(x=batch_data.x, y=batch_data.y, edge_index=batch_data.edge_index,
                                                     neighbor=[batch_data.neighbor_index, batch_data.neighbor_attr],
                                                     edge_weight=batch_data.edge_attr, epoch=epoch,
                                                     momentum_buffer=momentum_buffer)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                self._model.update_moving_average()
                epoch_loss += loss.item()
                st = '[{}][Epoch {}/{}] Loss: {:.10f}'.format(currentTime(), epoch, self._args.epochs, loss.item())
                print(st)

            if (epoch) % 5 == 0:
                self.infer_embeddings(epoch)
                self.evaluate(self._task, epoch, sillog)
                self.train_losses.append(epoch_loss / len(self._loader))
            print(f"Epoch {epoch} Average Loss: {self.train_losses[-1]:.10f}")

        
        print("\nTraining Done!")
        self.st_best = '** [last epoch: {}] last NMI: {:.4f}, ARI: {:.4f}, ACC: {:.4f} **\n'.format(self._args.epochs, self.best_dev_acc, self.best_ari, self.best_acc)
        print("[Final] {}".format(self.st_best))
        print('Saving checkpoint...')
        #a = pd.DataFrame(self.best_embeddings).T
        #a.to_csv("./results/klein-imputed.csv")
        f_final.write("{} -> {}\n".format(self.config_str, self.st_best))
        

    

class MoDET(nn.Module):
    def __init__(self, layer_config, args, temperature,momentum_buffer, beta, alpha, **kwargs):
        super().__init__()
        dec_dim = [512, 256]
        self.student_encoder = Encoder(layer_config=layer_config, dropout=args.dropout, **kwargs)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(args.mad, args.epochs)
        
        rep_dim = layer_config[-1]
        rep_dim_o = layer_config[0]
        self.student_predictor = nn.Sequential(nn.Linear(rep_dim,args.pred_hid*2), nn.BatchNorm1d(2*args.pred_hid), nn.ReLU(), nn.Linear(2*args.pred_hid, rep_dim), nn.ReLU())
        self.ZINB_Encoder = nn.Sequential(nn.Linear(rep_dim, dec_dim[0]), nn.ReLU(),
                                          nn.Linear(dec_dim[0], dec_dim[1]), nn.ReLU())
        self.pi_Encoder =  nn.Sequential(nn.Linear(dec_dim[1], rep_dim_o),nn.Sigmoid())
        self.disp_Encoder = nn.Sequential(nn.Linear(dec_dim[1], rep_dim_o), nn.Softplus())
        self.mean_Encoder = nn.Linear(dec_dim[1], rep_dim_o)
        self.student_predictor.apply(init_weights)
        self.relu = nn.ReLU()
        self.topk = args.topk
        self._device = "cpu"
        self.temperature = temperature
        self.beta = beta
        self.momentum_buffer = momentum_buffer
        self.alpha = alpha
    def clip_by_tensor(self,t, t_min, t_max):
        """
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
        t = torch.tensor(t,dtype = torch.float32)
        t_min = torch.tensor(t_min,dtype = torch.float32)
        t_max = torch.tensor(t_max,dtype = torch.float32)

        result = torch.tensor((t >= t_min),dtype = torch.float32) * t + torch.tensor((t < t_min),dtype = torch.float32) * t_min
        result = torch.tensor((result <= t_max),dtype = torch.float32) * result + torch.tensor((result > t_max),dtype = torch.float32) * t_max
        return result

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def forward(self, x, y, edge_index, neighbor, edge_weight, epoch=None, momentum_buffer=None):
        student = self.student_encoder(x=x, edge_index=edge_index,edge_weight=edge_weight)
        pred = self.student_predictor(student)
        modify = 0
        with torch.no_grad():
            teacher = self.teacher_encoder(x=x, edge_index=edge_index,edge_weight=edge_weight)
        if  self.momentum_buffer is None:
            self.momentum_buffer = teacher
        else:
             self.momentum_buffer = self.alpha * self.momentum_buffer + (1 - self.alpha) * teacher
            
        loss3 = (1 - self.beta) * knowledge_distillation_loss(pred, teacher, self.temperature) + \
                self.beta * knowledge_distillation_loss(pred,  self.momentum_buffer, self.temperature)
        if modify == 0:
            loss = loss3
        return student, loss.mean()



 

    def create_sparse(self, I):
        
        similar = I.reshape(-1).tolist()
        index = np.repeat(range(I.shape[0]), I.shape[1])
        
        assert len(similar) == len(index)
        indices = torch.tensor([index, similar],dtype=torch.int32).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)), [I.shape[0], I.shape[0]])

        return result

    def create_sparse_revised(self, I, all_close_nei_in_back):
        n_data, k = I.shape[0], I.shape[1]

        index = []
        similar = []
        for j in range(I.shape[0]):
            for i in range(k):
                index.append(int(j))
                similar.append(I[j][i].item())

        index = torch.masked_select(torch.LongTensor(index).to(self.device), all_close_nei_in_back.reshape(-1))
        similar = torch.masked_select(torch.LongTensor(similar).to(self.device), all_close_nei_in_back.reshape(-1))

        assert len(similar) == len(index)
        indices = torch.tensor([index.cpu().numpy().tolist(), similar.cpu().numpy().tolist()]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones(len(index)).to(self.device), [n_data, n_data])

        return result