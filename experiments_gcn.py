from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from models import TADW, TriDnr, DeepWalk, Node2Vec, Hope
from text_transformers import SBert, LDA, W2V, Sent2Vec, Doc2Vec, BOW, TFIDF
from datasets import Cora, CiteseerM10, Dblp

import numpy as np

from task import Task

from train_gcn import train_gcn

datasets = [
   ('Cora', Cora),
   ('CiteseerM10', CiteseerM10),
   ('DBLP', Dblp)
]

seeds = [1, 10, 100, 1000, 10000]
test_ratios = [0.5, 0.7, 0.9, 0.95]

res = {}

for ds_name, ds_constr in tqdm(datasets, desc='datasets'):
    ds = ds_constr()
    ds.transform_features(TFIDF())
    for test_ratio in tqdm(test_ratios, desc='test ratio'):
        scores = []
        for seed in seeds:
            score = train_gcn(ds, test_ratio, seed=seed, verbose=False)
            scores.append(score)

        res[f'{1 - test_ratio} - {ds_name} - GCN'] = scores


for name, scores in res.items():
    print(name, scores, np.mean(scores), np.std(scores))