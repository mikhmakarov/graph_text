from time import time

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


class LpTask:
    def __init__(self, dataset, test_ratios, text_transformer_constr, network_model_constr, modelType, modelArgs, d=100, concat=False, labels=True, modelName="GAT"):
        self.dataset = dataset
        self.test_ratios = test_ratios
        self.text_transformer_constr = text_transformer_constr
        self.network_model_constr = network_model_constr
        self.d = d
        self.labels = labels
        self.seeds = [1, 10]
        self.concat = concat
        self.modelType = modelType
        self.modelArgs = modelArgs
        self.modelName = modelName

    @staticmethod
    def __get_edge_embeddings(edges, mapping):
        edges_embs = []
        for edge in edges:
            u = mapping[edge[0]]
            v = mapping[edge[1]]
            edge_emb = np.multiply(u, v)
            edges_embs.append(edge_emb)

        return edges_embs

    def evaluate(self):
        if self.text_transformer_constr is not None:
            transformer = self.text_transformer_constr()
            self.dataset.transform_features(transformer)

        res = {}
        for test_ratio in tqdm(self.test_ratios, desc='test_ratios'):
            scores = []
            for seed in tqdm(self.seeds, desc='seeds'):
                lp_data = self.dataset.get_lp_data(test_ratio, seed)

                if self.network_model_constr is not None:
                    model = self.network_model_constr(lp_data['graph'], lp_data['features'],
                                                      self.modelType,
                                                      self.modelArgs,
                                                      labels=lp_data['labels'],
                                                      dim=self.d,
                                                      modelName=self.modelName)
                    model.learn_embeddings()
                    embeddings = model.embeddings
                else:
                    # use only document embeddings
                    embeddings = lp_data['features']

                if self.concat:
                    embeddings = np.hstack([embeddings, lp_data['features']])

                embeddings_map = {i: v for i, v in enumerate(embeddings)}

                train_X = self.__get_edge_embeddings(lp_data['train_edges'], embeddings_map)
                train_y = lp_data['train_target']
                test_X = self.__get_edge_embeddings(lp_data['test_edges'], embeddings_map)
                test_y = lp_data['test_target']

                clf = OneVsRestClassifier(LogisticRegression())
                clf.fit(train_X, train_y)
                pred_y = clf.predict(test_X)
                f1 = f1_score(test_y, pred_y, average='micro')

                scores.append(f1)

            res[test_ratio] = scores

        return res
