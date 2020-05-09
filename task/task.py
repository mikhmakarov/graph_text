from time import time

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


class Task:
    def __init__(self, dataset, test_ratios, text_transformer_constr, network_model_constr, d=100, labels=True,
                 concat=False):
        self.dataset = dataset
        self.test_ratios = test_ratios
        self.text_transformer_constr = text_transformer_constr
        self.network_model_constr = network_model_constr
        self.d = d
        self.labels = labels
        self.concat = concat
        self.seeds = [1, 10, 100, 1000, 10000]

    def evaluate(self):
        if self.text_transformer_constr is not None:
            transformer = self.text_transformer_constr()
            self.dataset.transform_features(transformer)

        data = self.dataset.get_data()

        if self.network_model_constr is not None:
            if not self.labels:
                model = self.network_model_constr(data['graph'], data['features'], dim=self.d)
                model.learn_embeddings()

        res = {}
        for test_ratio in tqdm(self.test_ratios, desc='test_ratios'):
            scores = []
            for seed in tqdm(self.seeds, desc='seeds'):
                train_indx, test_indx = train_test_split(data['main_ids'], stratify=data['main_labels'],
                                                         test_size=test_ratio,
                                                         random_state=seed)

                if self.labels:
                    labels = []
                    for i, label in zip(data['ids'], data['labels']):
                        if i in train_indx:
                            labels.append(label)
                        else:
                            labels.append(-1)
                else:
                    labels = None

                y = data['main_labels'].reshape(-1, 1)
                ids = data['main_ids'].reshape(-1, 1)

                if self.network_model_constr is not None:
                    if self.labels:
                        model = self.network_model_constr(data['graph'], data['features'], labels=labels, dim=self.d)
                        model.learn_embeddings()
                    embeddings = model.embeddings
                else:
                    # use only document embeddings
                    embeddings = data['features']

                embeddings = embeddings[data['main_ids']]

                if self.concat:
                    embeddings = np.hstack([embeddings, data['features'][data['main_ids']]])

                d = embeddings.shape[1]
                dev_df = pd.DataFrame(np.hstack((ids, embeddings, y)),
                                      columns=['index'] + [f'{i}' for i in range(d)] + ['label'])
                dev_df = dev_df.set_index('index')

                train_X, train_y = dev_df.loc[train_indx].values[:, :-1], dev_df.loc[train_indx].values[:, -1]
                test_X, test_y = dev_df.loc[test_indx].values[:, :-1], dev_df.loc[test_indx].values[:, -1]

                clf = OneVsRestClassifier(LogisticRegression())
                clf.fit(train_X, train_y)
                pred_y = clf.predict(test_X)
                f1 = f1_score(test_y, pred_y, average='micro')

                scores.append(f1)

            res[test_ratio] = scores

        return res
