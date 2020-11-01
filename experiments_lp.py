from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from models import TADW, TriDnr, DeepWalk, Node2Vec, Hope, GCN_Model, GCN_Model_LP
from text_transformers import SBert, LDA, W2V, Sent2Vec, Doc2Vec, BOW, TFIDF, Index
from datasets import Cora, CiteseerM10, Dblp, ScopusEcon

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from collections import defaultdict

from task import LpTask


candidates = [
#    (TriDnr, None, 'TriDnr'),
#     (TADW, SBert, 'TADW + SBert'),
#     (TADW, LDA, 'TADW + LDA'),
#      (TADW, W2V, 'TADW + W2V'),
#     (TADW, Sent2Vec, 'TADW + Sent2Vec'),
#    (TADW, Doc2Vec, 'TADW + Doc2Vec'),
#     (TADW, CountVectorizer, 'TADW + BOW'),
#    (TADW, TfidfVectorizer, 'TADW + TFIDF')
]

datasets = [
   ('Cora', Cora),
   # ('CiteseerM10', CiteseerM10),
    # ('DBLP', Dblp)
#    ('Scopus_Econ', ScopusEcon)
]

test_ratios = [0.5, 0.7, 0.9, 0.95]

tasks = [
    # ('BOW', lambda ds: LpTask(ds, test_ratios, lambda: BOW(), None, d=None)),
    # ('TFIDF', lambda ds: LpTask(ds, test_ratios, lambda: TFIDF(), None, d=None)),
    # ('LDA', lambda ds: LpTask(ds, test_ratios, lambda: LDA(), None, d=None)),
    # ('SBERT pretrained', lambda ds: LpTask(ds, test_ratios, lambda: SBert(train=False, d=300), None, d=None)),
    # ('W2V pretrained (d=300)', lambda ds: LpTask(ds, test_ratios, lambda: W2V(train=False, d=300), None, d=None)),
    # ('W2V (d=300)', lambda ds: LpTask(ds, test_ratios, lambda: W2V(train=True, d=300), None, d=None)),
    # ('W2V (d=64)', lambda ds: LpTask(ds, test_ratios, lambda: W2V(train=True, d=64), None, d=None)),
    # ('Doc2Vec pretrained (d=300)', lambda ds: LpTask(ds, test_ratios, lambda: Doc2Vec(train=False, d=300), None, d=None)),
    # ('Doc2Vec (d=300)', lambda ds: LpTask(ds, test_ratios, lambda: Doc2Vec(train=True, d=300), None, d=None)),
    # ('Doc2Vec (d=64)', lambda ds: LpTask(ds, test_ratios, lambda: Doc2Vec(train=True, d=64), None, d=None)),
    # ('Sent2Vec pretrained (d=600)', lambda ds: LpTask(ds, test_ratios, lambda: Sent2Vec(train=False, d=600), None, d=None)),
    # ('Sent2Vec (d=600)', lambda ds: LpTask(ds, test_ratios, lambda: Sent2Vec(train=True, d=600), None, d=None)),
    # ('Sent2Vec (d=64)', lambda ds: LpTask(ds, test_ratios, lambda: Sent2Vec(train=True, d=64), None, d=None)),
    # ('DeepWalk (d=100)', lambda ds: LpTask(ds, test_ratios, None, DeepWalk, d=100)),
    # ('Node2Vec (d=100)', lambda ds: LpTask(ds, test_ratios, None, Node2Vec, d=100)),
    # ('Hope (d=100)', lambda ds: LpTask(ds, test_ratios, None, Hope, d=100)),
    # ('TADW - BOW', lambda ds: LpTask(ds, test_ratios, BOW, TADW, d=160)),
    # ('TADW - TFIDF', lambda ds: LpTask(ds, test_ratios, TFIDF, TADW, d=160)),
    # ('TADW - Sent2Vec', lambda ds: LpTask(ds, test_ratios, lambda: Sent2Vec(train=True, d=64), TADW, d=160)),
    # ('TADW - Word2Vec', lambda ds: LpTask(ds, test_ratios, lambda: W2V(train=True, d=64), TADW, d=160)),
    # ('TriDNR', lambda ds: LpTask(ds, test_ratios, None, TriDnr, d=160)),
    # ('BOW:DeepWalk', lambda ds: LpTask(ds, test_ratios, BOW, DeepWalk, d=100,
     #                                 concat=True)),
    # ('Word2Vec:DeepWalk', lambda ds: LpTask(ds, test_ratios, lambda: W2V(train=True, d=64), DeepWalk, d=100,
      #                                     concat=True)),
    # ('Sent2Vec:DeepWalk', lambda ds: LpTask(ds, test_ratios, lambda: Sent2Vec(train=True, d=64), DeepWalk, d=100, concat=True)),
    ('GCN (Sent2Vec)', lambda ds: LpTask(ds, test_ratios, SBert, GCN_Model_LP, d=100, labels=True)),
 #   ('GCN (Word2Vec)', lambda ds: LpTask(ds, test_ratios, lambda: W2V(train=True, d=64), GCN_Model, d=100, labels=True)),
]


res = {}

for ds_name, ds_constr in tqdm(datasets, desc='datasets'):
    ds = ds_constr()
    for task_name, task_constr in tqdm(tasks, desc='Tasks'):
        task = task_constr(ds)
        task_res = task.evaluate()
        for test_ratio in task_res:
            scores = task_res[test_ratio]
            res[f'{1 - test_ratio} - {ds_name} - {task_name}'] = scores

        print(res)

for name, scores in res.items():
    print(name, scores, np.mean(scores), np.std(scores))

