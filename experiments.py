from tqdm import tqdm_notebook as tqdm

from models import TADW, TriDnr
from text_transformers import SBert, LDA, W2V, Sent2Vec, Doc2Vec
from datasets import Cora, CiteseerM10

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


candidates = [
    (TriDnr, None, 'TriDnr'),
#     (TADW, SBert, 'TADW + SBert'),
#     (TADW, LDA, 'TADW + LDA'),
#     (TADW, W2V, 'TADW + W2V'),
#     (TADW, Sent2Vec, 'TADW + Sent2Vec'),
#     (TADW, Doc2Vec, 'TADW + Doc2Vec'),
#     (TADW, CountVectorizer, 'TADW + BOW'),
#    (TADW, TfidfVectorizer, 'TADW + TFIDF')
]

ds = Cora()

d = 160
seeds = [1]  # [1, 10, 100]


res = defaultdict(list)
for constr, transf, name in tqdm(candidates, desc='candidates'):
    if transf is not None:
        transformer = transf()
        ds.transform_features(transformer)

    data = ds.get_data()

    if name != 'TriDnr':
        model = constr(data['graph'], data['features'], dim=d)
        model.learn_embeddings()

    for seed in tqdm(seeds, desc='seeds'):
        train_indx, test_indx = train_test_split(data['main_ids'], stratify=data['main_labels'], test_size=0.5,
                                                 random_state=seed)

        if name == 'TriDnr':
            labels = []
            for i, label in enumerate(data['labels']):
                if i in train_indx:
                    labels.append(label)
                else:
                    labels.append(-1)

            model = constr(data['graph'], data['features'], labels, dim=d)
            model.learn_embeddings()

        y = data['main_labels'].reshape(-1, 1)
        ids = data['main_ids'].reshape(-1, 1)
        dev_df = pd.DataFrame(np.hstack((ids, model.get_embeddings_for_ids(ids), y)),
                              columns=['index'] + [f'{i}' for i in range(d)] + ['label'])
        dev_df = dev_df.set_index('index')

        train_X, train_y = dev_df.loc[train_indx].values[:, :-1], dev_df.loc[train_indx].values[:, -1]
        test_X, test_y = dev_df.loc[test_indx].values[:, :-1], dev_df.loc[test_indx].values[:, -1]

        clf = OneVsRestClassifier(GradientBoostingClassifier())
        clf.fit(train_X, train_y)
        pred_y = clf.predict(test_X)
        f1 = f1_score(test_y, pred_y, average='micro')

        print(name, f1)

        res[name].append(f1)