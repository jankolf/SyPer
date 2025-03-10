# Python imports
from types import SimpleNamespace

# Installed modules import
from sklearn.metrics import roc_curve, auc
import numpy as np
from tqdm import tqdm

# Periocular imports
from periocular.datasets import PeriocularDataset


def evaluate_verification( 
                            dataset : PeriocularDataset, 
                            embeddings : dict
                         ):
    # Calculate scores for pairs
    N_comps = len(dataset)
    scores = np.zeros((N_comps,), dtype=np.float32)
    labels = np.zeros((N_comps,), dtype=np.int32)

    for idx in tqdm(range(N_comps), unit="Comp", desc="Calc. Pairs"):

        P, R, L = dataset[idx]
        P = embeddings[P]
        R = embeddings[R]

        scores[idx] = P.dot(R).item()
        labels[idx] = L

    # Calculate metrics
    metrics = SimpleNamespace()

    far, tar, thr = roc_curve(labels, scores)
    
    eer_index = np.argmin(np.abs(1 - tar - far))
    metrics.eer = far[eer_index]
    metrics.eer_thr = thr[eer_index]

    metrics.fnmr_1e_2 = 1 - tar[np.argmin(np.abs(far-1e-2))]
    metrics.fnmr_1e_3 = 1 - tar[np.argmin(np.abs(far-1e-3))]
    metrics.fnmr_1e_4 = 1 - tar[np.argmin(np.abs(far-1e-4))]
    metrics.fnmr_1e_5 = 1 - tar[np.argmin(np.abs(far-1e-5))]
    metrics.fnmr_1e_6 = 1 - tar[np.argmin(np.abs(far-1e-6))]
    
    metrics.auc = auc(far, tar)

    return metrics
