# Python imports
from types import SimpleNamespace

# Installed modules import
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm


# Periocular imports
from periocular.datasets import PeriocularDataset

def evaluate_rank_accuracy( 
                            dataset : PeriocularDataset, 
                            embeddings : dict
                         ):
    emb_size = len(embeddings[next(iter(embeddings))])
    N = len(dataset.image_names)

    emb = np.zeros((N, emb_size), dtype=np.float64)
    labels = dataset._image_labels.numpy()

    for idx in range(N):
        key = dataset.image_names[idx]
        emb[idx,:] = embeddings[key].numpy()

    similarity = cosine_similarity(emb)

    rank_1_sim = 0
    rank_5_sim = 0

    for idx in tqdm(range(N), desc="Ranking", unit="rank"):
        sorting = np.argsort(similarity[idx])[::-1]
        ranking = labels[sorting]

        current_label = labels[idx]

        if current_label == ranking[1]:
            rank_1_sim += 1

        if (current_label == ranking[1:6]).any():
            rank_5_sim += 1

    metrics = SimpleNamespace()

    metrics.rank_1_acc = rank_1_sim / N
    metrics.rank_5_acc = rank_5_sim / N

    return metrics