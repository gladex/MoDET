import csv
import numpy as np
import pandas as pd
import cmath as cm
import h5py
from scipy import sparse
import scipy
from sklearn.utils.class_weight import compute_class_weight
import sklearn

import scanpy as sc
from matplotlib import rcParams
from sklearn.metrics import normalized_mutual_info_score, pairwise, adjusted_rand_score,silhouette_score
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


def nomalize_for_AF(filename,gene_num,raw, sparsify = False, skip_exprs = False): #处理数据 当前处理的Alzheimer filename读取test里的计数矩阵
    with h5py.File(filename, "r") as f:
        obs = np.array(f["obs_names"][...])
        var = np.array(f["var_names"][...])
        cell_name = np.array(f["obs"]["cell_type1"])
        cell_type, cell_label = np.unique(cell_name, return_inverse=True)
        class_num = np.array(cell_label).max() + 1
        data_label = []
        data_array = []
        for i in range(cell_label.shape[0]):
            x = np.zeros(class_num)
            x[cell_label[i]] = 1
            data_label.append(x)
        data_label = np.array(data_label)
        cell_type = np.array(cell_type)
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                         exprs_handle["indptr"][...]), shape=exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sparse.csr_matrix(mat)
        else:
            mat = sparse.csr_matrix((obs.shape[0], var.shape[0]))
        X = np.array(mat.toarray())
        X = np.ceil(X).astype(int)
        adata = sc.AnnData(X)
        adata.obs['Group'] = cell_label
        adata = normalize(adata, copy=True, highly_genes=gene_num, size_factors=True, normalize_input=raw, logtrans_input=True)
        count = adata.X
        if raw == False:
             a = pd.DataFrame(count).T
             a.to_csv("./results/adam-raw.csv")
        return count,adata.obs['Group']
def normalize_for_AL(filename, gene_num, raw):
    with h5py.File(filename, "r") as f:
        gene_names = np.array(f["gene_names"][...])
        cell_names = np.array(f["cell_names"][...])
        data_label = f["Y"][...].astype(int)
        # 读取密集矩阵X
        X = f["X"][...].astype(np.float32)
        adata = sc.AnnData(X)
        adata.obs_names = cell_names
        adata.var_names = gene_names
        adata.obs['cell_type'] = data_label
        adata = normalize(adata, copy=True, highly_genes=gene_num, size_factors=True, normalize_input=raw, logtrans_input=True)
        count = adata.X
        group_labels = adata.obs['cell_type']   
    return count, group_labels

def normalize_for_AL_new(filename, gene_num, raw):
    with h5py.File(filename, "r") as f:
        X = np.array(f["X"][...]).astype(np.float32)
        data_label = np.array(f["Y"][...]).astype(int)

        if X.shape[0] != len(data_label):
            raise ValueError(f"基因表达矩阵的细胞数 ({X.shape[0]}) 和细胞标签数 ({len(data_label)}) 不匹配")

        adata = sc.AnnData(X)
        adata.obs['Y'] = data_label 
        adata = normalize(adata, copy=True, highly_genes=gene_num, size_factors=True, normalize_input=raw, logtrans_input=True)
        count = adata.X
        group_labels = adata.obs['Y']
        
    return count, group_labels

def nomalize_for_Klein(file_name, label_path, gene_num): 
    data = pd.read_csv(file_name, header=0, sep="\t") 
    print(data.head())
    label = pd.read_csv(label_path, header=0, sep=",")  
    # data_label = []
    label = np.array(label)[:, 1]
    cell_type, cell_label = np.unique(label, return_inverse=True)
    data_label = []
    for i in range(len(cell_label)): 
        data_label.append(cell_type[cell_label[i]])
    data_label = np.array(data_label)
    print(data_label) 
    arr1 = np.array(data) 
    gene_name = np.array(arr1[1:, 0]) 
    cell_name = np.array(arr1[0, 1:])
    X = arr1[1:, 1:].T 
    adata = sc.AnnData(X)
    print(cell_type.shape)
    adata.obs['Group'] = data_label 
    adata.obs['cell_name'] = cell_name 
    adata.var['Gene'] = gene_name
    adata = normalize(adata, copy=True, highly_genes=gene_num, size_factors=True, normalize_input=False,
                      logtrans_input=True)
    count = adata.X
    a = pd.DataFrame(count).T
    a.to_csv("./compare_funtion/MDTS/MoDET-master/results/raw-klein.csv")
    return count, cell_label,adata.obs['size_factors'],adata.var['Gene']



def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
def dopca(X, dim=10):
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10
def normalize(adata, copy=True, highly_genes = None, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if isinstance(adata, sc.AnnData): # # 检查 adata 是否为 sc.AnnData 类型的对象，sc.AnnData 是 scanpy 中用于存储数据的结构
        if copy: #为true
            adata = adata.copy() #赋值adata
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.' # # 确保 adata 对象中没有 'n_count' 这个字段，这表示数据应该未被标准化
    assert 'n_count' not in adata.obs, norm_error
    #if adata.X.size < 50e6: # check if adata.X is integer only if array is small # 如果 adata.X 的大小小于 50e6（即500万），则检查 adata.X 是否只包含整数计数数据
       # if sparse.issparse(adata.X):
       #     assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
       # else:
       #     assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts: # # 如果设置了 filter_min_counts
        sc.pp.filter_genes(adata, min_counts=1) # 则移除在任何细胞中的表达计数小于1的基因 
        sc.pp.filter_cells(adata, min_counts=1) # 移除在所有基因中的表达计数小于1的细胞
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)

    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)

    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata


