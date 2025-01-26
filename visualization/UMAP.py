import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
from matplotlib.colors import ListedColormap
import torch
import pandas as pd

# 设置全局字体为 Times New Roman，字号14
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14

# 定义五种颜色对应细胞类型
colors = ['#eba339', '#ff0066',  '#ba55d3','#4aacc5', '#2ca02c']

def generate_umap(embedding_path, dataset, method_name, ax, annotation, annotation_pos, xlim, ylim, xticks, yticks):
    # 加载嵌入
    embedding = torch.load(embedding_path)
    
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()
    
    # 对 embeddings 进行归一化
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / norm.clip(min=1e-10)
    
    data = dataset[0]
    true_y = data.y.detach().cpu().numpy()

    # 确保 true_y 是一个分类数据类型
    adata = sc.AnnData(X=embedding)
    adata.obs['true_labels'] = pd.Categorical(true_y.astype(int))

    cmap = ListedColormap(colors)

    # 计算邻域图
    sc.pp.neighbors(adata)
    
    # 计算 UMAP
    sc.tl.umap(adata)
    
    # 设置背景颜色为白色
    ax.set_facecolor('white')

    # 绘制 UMAP 图，显示坐标轴
    sc.pl.umap(adata, color='true_labels', palette=colors, show=False, ax=ax, frameon=True, legend_loc=None)
    ax.set_title(f'{method_name}', fontsize=14)

    # 设置坐标轴的尺度
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # 设置坐标轴标签，并通过 labelpad 调整与图的距离
    ax.set_xlabel("UMAP 1", fontsize=12, labelpad=2)  # 调整 labelpad 使标签更靠近图
    ax.set_ylabel("UMAP 2", fontsize=12, labelpad=-10)

    # 设置刻度
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # 显示刻度
    ax.tick_params(axis='both', which='both', length=5, width=1, direction='in')

    # 添加注释，使用 annotation_pos 参数来手动调节位置
    ax.text(annotation_pos[0], annotation_pos[1], annotation, transform=ax.transAxes, 
            fontsize=14, va='top', ha='left', family='Times New Roman')

# 手动输入四个不同的嵌入文件路径
embedding_paths = [
    "./model_checkpoints/Autoclass.pt",
    "./model_checkpoints/scTAG.pt",
    "./model_checkpoints/scGCL.pt",
    "./model_checkpoints/spleen-MoDET_0.6_quake_10x_spleen_clustering.pt"
]

# 定义对应的方法名称
method_names = ['Autoclass', 'scTAG', 'scGCL', 'MoDET']

# 加载 dataset 数据
dataset = torch.load("./data/pyg/quake_10x_spleen/processed/data.pt")

# 创建子图布局，调整每个子图的大小为1:1比例
fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # 调整图像大小，每个子图比例为1:1

# 通过 wspace 和 hspace 控制图之间的距离
plt.subplots_adjust(wspace=0.17, hspace=0.21)

# 添加注释列表及其位置
annotations = ['(a)', '(b)', '(c)', '(d)']
annotation_positions = [(-0.14, 1.06), (-0.14, 1.06), (-0.14, 1.06), (-0.14, 1.06)]  # 调整 x 坐标位置靠左

xlim = (-12.5, 22.5)
ylim = (-12.5, 23.5)

# 设置刻度范围，包含起点和终点 -12.5 到 22.5，每 5 个单位一个刻度
xticks = [-10 , 0, 10,20]
yticks =  [-10 , 0, 10,20]




# 绘制四个 UMAP 图像
for i, (embedding_path, method_name) in enumerate(zip(embedding_paths, method_names)):
    generate_umap(embedding_path, dataset, method_name, axs[i//2, i%2], annotations[i], annotation_positions[i], xlim, ylim, xticks, yticks)

# 添加图例，保留 "Cell Type" 标签并不留空白
legend_elements = [
    plt.Line2D([0], [0], marker='', color='none', label='Cell Type')
] + [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=12, label=label) 
    for color, label in zip(colors, ['B cell', 'T cell', 'Dendritic cell', 'Macrophage', 'Natural killer cell'])
]

# 设置图例的位置，使其放在图像的下方，并靠近图像
fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.6, 0.02), ncol=6, fontsize=14,
           handletextpad=0.05, columnspacing=0.1)

# 保存图片，DPI设为600
fig.savefig('UMAP.png', dpi=600, bbox_inches='tight')
plt.show()
