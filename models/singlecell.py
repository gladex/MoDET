import os.path as osp
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.io import read_npz
from torch_geometric.data import Data
import models.data_Preprocess
import scipy.sparse as sp
import pandas as pd
from torch_geometric.utils import remove_self_loops, to_undirected
from models.graph_funtion import *

class Singlecell(InMemoryDataset):

    def __init__(self, root, name, filepath, transform=None, pre_transform=None): 
        self.name = name.lower()  
        self.filepath = filepath
        #self.labelpath = "./test_csv/klein/label.csv"
        #self.labelpath = "./test_csv/Kolodziejczyk/label.csv"
        super(Singlecell, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name.capitalize(), 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name.capitalize(), 'processed')

    @property
    def raw_file_names(self):
        return 'amazon_electronics_{}.npz'.format(self.name.lower())

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        raw = False
        
        #data, data_label = models.data_Preprocess.nomalize_for_AF(self.filepath, 1999, raw)
        data, data_label = models.data_Preprocess.normalize_for_AL(self.filepath, 1999, raw)
        #data, data_label, size_factor,gene = models.data_Preprocess.nomalize_for_Klein(self.filepath, self.labelpath, 1999)
        #data, data_label = models.data_Preprocess.normalize_for_AL_new(self.filepath, 1999, raw); 
        x = torch.tensor(np.array(data), dtype=torch.float32)
        y = torch.tensor(data_label, dtype=torch.long)
        adj, adj_weights = get_adj_with_fixed_knn(data)
        adj = sp.coo_matrix(adj)
        edge_index = torch.tensor([adj.row, adj.col], dtype=torch.long)
        edge_weight = torch.tensor(adj_weights[adj.row, adj.col], dtype=torch.float32)  
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight) 
        edge_index, edge_weight = to_undirected(edge_index, edge_weight, num_nodes=x.size(0))
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        data, slices = self.collate([data])
        print(self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])
    def __repr__(self):
        return '{}{}()'.format(self.__class__.__name__, self.name.capitalize())