# A Dual-level Momentum Distillation Method with Extreme Threshold for Imputing scRNA-seq Data

Here, we propose a single-cell data imputation method for scRNA-seq data, named MoDET, which integrates dual-level momentum distillation and self-supervised learning to estimate dropout values. 

Through extensive experiments on the Quake_10x_Spleen dataset, we verify that MoDET outperforms existing state-of-the-art imputation methods in both clustering performance and gene imputation accuracy. The other datasets used in these experiments are mentioned in the data availability section of the manuscript.

 Specifically, if the dataset is in a sparse matrix H5 format, the preprocessing uses `data_Preprocess.normalize_for_AF`, whereas if the dataset is in a dense matrix H5 format, it uses `data_Preprocess.normalize_for_AL`. Additionally, if the dataset is in CSV format, you can refer to `data_Preprocess.normalize_for_Klein` for preprocessing.

### Requirement

- Python version: 3.8.8
- Pytorch version: 1.10.0
- torch-cluster version: 1.6.0
- torch-geometric version: 2.2.0
- torch-scatter version: 2.0.9
- torch-sparse version: 0.6.13
- faiss-cpu: 1.7.3 

## Usage
You can run MoDET from the command line:
```
$ python main.py --dataset quake_10x_spleen --epochs 300
```
## Arguments
|    Parameter    | Introduction                                                 |
| :-------------: | ------------------------------------------------------------ |
 Additionally, if the dataset is in a sparse matrix format, appropriate preprocessing steps are applied to efficiently handle sparse data structures.
|     epochs     | Number of training epochs. Adjust this parameter to control the duration of training. |

## Visualization
For visualizing the results, we provide visualization scripts in the `visualization` folder. These scripts can be directly run to generate plots that help analyze the performance of the imputation method across different datasets and metrics.

## Who do I talk to?
* Have any questions or issues related to the repository, please contact: Dr. Binhua Tang (bh.tang@outlook.com) or Xinyu Gao (2415622926@qq.com).
