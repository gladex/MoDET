import torch

# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import random
random.seed(0)
import models.data_Preprocess
import numpy as np
np.random.seed(0)
import models.utils
import pandas as pd

def main():
    args, unknown = models.utils.parse_args()  
    if args.embedder == 'MoDET': 
        from models import MoDET_ModelTrainer
        embedder = MoDET_ModelTrainer(args)  
    embedder.train()
    embedder.writer.close()

def imputation(file_path):
    args, unknown = models.utils.parse_args()
    imputation_m = torch.load(file_path)

    a = pd.DataFrame(imputation_m).T
    a.to_csv("./results/MoDET-imputed-" + args.dataset + ".csv")

if __name__ == "__main__":
    main() # TODO 
    #imputation("./model_checkpoints/embeddings_klein_clustering.pt")  
  