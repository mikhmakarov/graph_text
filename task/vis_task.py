from time import time

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


class VisTask:
    def __init__(self, dataset, text_transformer_constr, network_model_constr, d=100, labels=True, concat=False):
        self.dataset = dataset
        self.text_transformer_constr = text_transformer_constr
        self.network_model_constr = network_model_constr
        self.d = d
        self.labels = labels
        self.concat = concat
        self.seeds = [1, 10, 100, 1000, 10000]

    def get_embeddings(self):
        if self.text_transformer_constr is not None:
            transformer = self.text_transformer_constr()
            self.dataset.transform_features(transformer)

        data = self.dataset.get_data()

        if self.network_model_constr is not None:
            model = self.network_model_constr(data['graph'], data['features'], labels=data['labels'], dim=self.d)
            model.learn_embeddings()
            embeddings = model.embeddings
        else:
            embeddings = data['features']

        embeddings = embeddings[data['main_ids']]

        if self.concat:
            embeddings = np.hstack([embeddings, data['features'][data['main_ids']]])

        return embeddings, data['main_labels']
