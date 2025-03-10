# Python imports
from types import SimpleNamespace

# Installed modules import
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

# Torch imports
import torch
from torch import nn

# Periocular imports
from periocular.datasets import PeriocularDataset

@torch.no_grad()
def extract_embeddings( 
                        dataset : PeriocularDataset, 
                        model : nn.Module,
                        gpu_device
                       ):
    embeddings = {}
    model.eval()
    N_imgs = len(dataset.image_names)
    for idx in tqdm(range(N_imgs), desc="Calc. Emb.", unit="Emb."):
        
        img_name = dataset.image_names[idx]
        img = dataset.load_image(img_name).unsqueeze(0).to(gpu_device)

        emb = model(img).cpu().squeeze()
        embeddings[img_name] = emb / torch.sqrt((emb**2).sum())
        del img

    return embeddings