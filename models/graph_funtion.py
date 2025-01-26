import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import scanpy as sc
import torch
from torch import nn
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity

import torch
from torch import nn
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity

import torch
from torch import nn
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph

def get_adj_with_fixed_knn(count, k=15, pca=50, mode="connectivity"):
   
    if pca:
        countp = dopca(count, dim=pca)
    else:
        countp = count
    sim_matrix = cosine_similarity(countp)

    A = kneighbors_graph(countp, k, mode=mode, metric="cosine", include_self=True).toarray()
    adj_weights = sim_matrix * A  
  
    adj_weights = torch.tensor(adj_weights, dtype=torch.float32)
   
    for i in range(adj_weights.size(0)):
        row = adj_weights[i]
        non_zero_indices = row.nonzero().squeeze()

        if len(non_zero_indices) > 0:
            row[non_zero_indices] = torch.nn.functional.softmax(row[non_zero_indices], dim=0)
            row[non_zero_indices] *= 15
    
    return A, adj_weights



def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output

def dopca(X, dim=10):
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10