import numpy as np
np.random.seed(0)
import torch
import torch.nn as nn
from models import LogisticRegression
from models.utils import printConfig
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import models.utils
from scipy.special import rel_entr  
# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

import random
import math
from scipy.optimize import linear_sum_assignment
random.seed(0)
import models.data_Preprocess
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score, pairwise
from sklearn.metrics import adjusted_rand_score
def cluster_acc(y_true, y_pred):
     D = max(y_pred.max(), y_true.max()) + 1
     w = np.zeros((D, D), dtype=int)
     for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
     ind = linear_sum_assignment(w.max() - w)
     ind = list(zip(ind[0], ind[1]))  # Ensure ind is a list of (i, j) tuples
     return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
class embedder:
    def __init__(self, args):
        self.args = args
        self.hidden_layers = eval(args.layers)
        printConfig(args)

    def infer_embeddings(self, epoch):
        self._model.train(False)
        self._embeddings = self._labels = None
        self._train_mask = self._dev_mask = self._test_mask = None

        for bc, batch_data in enumerate(self._loader):
         
            batch_data.to(self._device)
        

            emb, loss = self._model(x = batch_data.x, y=batch_data.y, edge_index=batch_data.edge_index,
                                                                           neighbor=[batch_data.neighbor_index, batch_data.neighbor_attr],
                                                                           edge_weight=batch_data.edge_attr, epoch=epoch)
        
            emb = emb.detach()
            y = batch_data.y.detach()
            if self._embeddings is None:
                self._embeddings, self._labels = emb, y
            else:
                self._embeddings = torch.cat([self._embeddings, emb])
                self._labels = torch.cat([self._labels, y])

    def evaluate(self, task, epoch, sillog):
        if task == "node":
            self.evaluate_node(epoch)
        elif task == "clustering":
            self.evaluate_clustering(epoch, sillog)
        elif task == "similarity":
            self.run_similarity_search(epoch)

    def evaluate_node(self, epoch):
        emb_dim, num_class = self._embeddings.shape[1], self._labels.unique().shape[0]

        dev_accs, test_accs = [], []

        for i in range(20):
            self._train_mask = self._dataset[0].train_mask[i]
            self._dev_mask = self._dataset[0].val_mask[i]
            if self._args.dataset == "wikics":
                self._test_mask = self._dataset[0].test_mask
            else:
                self._test_mask = self._dataset[0].test_mask[i]

            classifier = LogisticRegression(emb_dim, num_class).to(self._device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=1e-5)

            for _ in range(100):
                classifier.train()
                logits, loss = classifier(self._embeddings[self._train_mask], self._labels[self._train_mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            dev_logits, _ = classifier(self._embeddings[self._dev_mask], self._labels[self._dev_mask])
            test_logits, _ = classifier(self._embeddings[self._test_mask], self._labels[self._test_mask])
            dev_preds = torch.argmax(dev_logits, dim=1)
            test_preds = torch.argmax(test_logits, dim=1)

            dev_acc = (torch.sum(dev_preds == self._labels[self._dev_mask]).float() /
                       self._labels[self._dev_mask].shape[0]).detach().cpu().numpy()
            test_acc = (torch.sum(test_preds == self._labels[self._test_mask]).float() /
                        self._labels[self._test_mask].shape[0]).detach().cpu().numpy()

            dev_accs.append(dev_acc * 100)
            test_accs.append(test_acc * 100)

        dev_accs = np.stack(dev_accs)
        test_accs = np.stack(test_accs)

        dev_acc, dev_std = dev_accs.mean(), dev_accs.std()
        test_acc, test_std = test_accs.mean(), test_accs.std()

        print('** [{}] [Epoch: {}] Val: {:.4f} ({:.4f}) | Test: {:.4f} ({:.4f}) **'.format(self.args.embedder, epoch, dev_acc, dev_std, test_acc, test_std))

        if dev_acc > self.best_dev_acc:
            self.best_dev_acc = dev_acc
            self.best_test_acc = test_acc
            self.best_dev_std = dev_std
            self.best_test_std = test_std
            self.best_epoch = epoch

        self.best_dev_accs.append(self.best_dev_acc)
        self.st_best = '** [Best epoch: {}] Best val | Best test: {:.4f} ({:.4f}) / {:.4f} ({:.4f})**\n'.format(
            self.best_epoch, self.best_dev_acc, self.best_dev_std, self.best_test_acc, self.best_test_std)
        print(self.st_best)


    def evaluate_clustering(self, epoch, sillog):
        embeddings = F.normalize(self._embeddings, dim=-1, p=2).detach().cpu().numpy()
        nb_class = len(self._dataset[0].y.unique())
        true_y = self._dataset[0].y.detach().cpu().numpy()

        estimator = KMeans(n_clusters=nb_class, n_init=10)
        NMI_list = []
        ARI_list = []
        ACC_list = []

        for i in range(10):
            estimator.fit(embeddings)
            y_pred = estimator.predict(embeddings)
            s1 = normalized_mutual_info_score(true_y, y_pred, average_method='arithmetic')
            s2 = adjusted_rand_score(true_y, y_pred)
            s3 = cluster_acc(true_y, y_pred)
            NMI_list.append(s1)
            ARI_list.append(s2)
            ACC_list.append(s3)

        s1 = sum(NMI_list) / len(NMI_list)
        s2 = sum(ARI_list) / len(ARI_list)
        s3 = sum(ACC_list) / len(ACC_list)
        sillog.append(s2)
        arr_sil = np.array(sillog)

        silhid = metrics.silhouette_score(self._embeddings.detach().cpu().numpy(), y_pred, metric='euclidean')

       
        print('** [{}] [Current Epoch {}] this epoch NMI values: {:.4f}, ARI: {:.4f}, ACC: {:.4f}, and this epoch sil values: {:.4f} **'.format(self._args.embedder, epoch, s1, s2, s3, silhid))
        if s1 > self.best_dev_acc:
            self.best_epoch = epoch
            self.best_dev_acc = s1
            self.best_ari = s2
            self.best_acc= s3
            print("~~~~~~~~~~~~~~~~~~")
            print(self.best_dev_acc)
            if self._args.checkpoint_dir:
                print('Saving checkpoint...')
                #torch.save(embeddings, os.path.join(self._args.checkpoint_dir, 'spleen-MoDET_0_{}_{}.pt'.format(self._args.dataset, self._args.task)))
                #a = pd.DataFrame(self._embeddings.detach().cpu().numpy()).T
                #a.to_csv("./results/scGCL-Tosches_turtle-euclidean.csv")
            print("save")
            print("~~~~~~~~~~~~~~~~~~")
           # if math.floor(silhid*100) >= math.floor(self.best_dev_acc*100):

           #     self.best_dev_acc = round(silhid, 2)
           #     self.best_embeddings = embeddings
            #    self.best_test_acc = s1
          #      print("~~~~~~~~~~~~~~~~~~")
          #      print(silhid)
           #     if self._args.checkpoint_dir is not '':
           #         print('Saving checkpoint...')
           #         torch.save(embeddings, os.path.join(self._args.checkpoint_dir,
           #                                             'embeddings_{}_{}.pt'.format(self._args.dataset, self._args.task)))
           #         # zzz = np.concatenate((true_y.reshape(3660, 1), y_pred.reshape(3660, 1)), axis=1)
           #         a = pd.DataFrame(self.best_embeddings).T
           ##         a.to_csv("./results/scGCL-Tosches_turtle-euclidean.csv")
          #      print("save")
          #      print("~~~~~~~~~~~~~~~~~~")
            # if abs(self.current_loss - self.last_loss) < 1e3:
            #     if self._args.checkpoint_dir is not '':
            #         print('Saving checkpoint...')
            #         torch.save(embeddings, os.path.join(self._args.checkpoint_dir, 'embeddings_{}_{}.pt'.format(self._args.dataset, self._args.task)))
            #         # zzz = np.concatenate((true_y.reshape(3660, 1), y_pred.reshape(3660, 1)), axis=1)
            #         a = pd.DataFrame(self._embeddings.detach().cpu().numpy()).T
            #         a.to_csv("./results/scGCL-Tosches_turtle-euclidean.csv")
            #         self.st_best = '** Finally NMI: {:.4f} **\n'.format(s1)
            #         print(self.st_best)
            #         return True
            #
            # self.last_loss = self.current_loss

    def run_similarity_search(self, epoch):
        test_embs = self._embeddings.detach().cpu().numpy()
        test_lbls = self._dataset[0].y.detach().cpu().numpy()
        numRows = test_embs.shape[0]

        cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
        st = []
        for N in [5, 10]:
            indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
            tmp = np.tile(test_lbls, (numRows, 1))
            selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
            original_label = np.repeat(test_lbls, N).reshape(numRows, N)
            st.append(np.round(np.mean(np.sum((selected_label == original_label), 1) / N), 4))

        print("** [{}] [Current Epoch {}] sim@5 : {} | sim@10 : {} **".format(self.args.embedder, epoch, st[0], st[1]))

        if st[0] > self.best_dev_acc:
            self.best_dev_acc = st[0]
            self.best_test_acc = st[1]
            self.best_epoch = epoch

        self.best_dev_accs.append(self.best_dev_acc)
        self.st_best = '** [Best epoch: {}] Best @5 : {} | Best @10: {} **\n'.format(self.best_epoch, self.best_dev_acc, self.best_test_acc)
        print(self.st_best)

        return st


import torch
from torch import nn
from torch_geometric.nn import GATConv
from torch.nn import BatchNorm1d

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, layer_config, dropout=None, project=False, **kwargs):
        super().__init__()
        self.stacked_gnn = nn.ModuleList([
            GCNConv(layer_config[i - 1], layer_config[i])
            for i in range(1, len(layer_config))
        ])
        self.stacked_bns = nn.ModuleList([
            nn.BatchNorm1d(layer_config[i], momentum=0.01)
            for i in range(1, len(layer_config))
        ])
        self.stacked_prelus = nn.ModuleList([
            nn.ReLU() for _ in range(1, len(layer_config))
        ])

    def forward(self, x, edge_index, edge_weight):
    
        for i, gnn in enumerate(self.stacked_gnn):
            x = gnn(x, edge_index, edge_weight=edge_weight)
            x = self.stacked_bns[i](x)
            x = self.stacked_prelus[i](x)
        return x
