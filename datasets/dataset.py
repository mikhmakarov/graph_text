import networkx as nx
import pandas as pd
import numpy as np
import scipy

from sklearn.preprocessing import LabelEncoder


class Dataset(object):
    def __init__(self, graph_path, texts_path, labels_path, sep='\t'):
        graph = nx.from_edgelist(pd.read_csv(graph_path, sep=sep).values.tolist())
        node_map = {n: i for i, n in enumerate(graph.nodes)}
        self.graph = nx.relabel_nodes(graph, node_map)

        texts_df = pd.read_csv(texts_path, header=None, sep=sep, names=['id', 'text'], index_col='id')
        labels_df = pd.read_csv(labels_path, header=None, sep=sep, names=['id', 'label'], index_col='id')
        df = pd.merge(texts_df, labels_df, left_index=True, right_index=True)

        ids = texts_df.index.values
        ids = np.vectorize(node_map.get)(ids)

        texts = df['text'].values
        labels = df['label'].values

        le = LabelEncoder()
        labels = le.fit_transform(labels)

        sort_indx = ids.argsort()

        self.texts = texts[sort_indx]
        self.ids = ids[sort_indx].astype(int)
        self.labels = labels[sort_indx].astype(int)

        self.features = None

    def transform_features(self, transformer):
        text_embs = transformer.fit_transform(self.texts)

        if scipy.sparse.issparse(text_embs):
            text_embs = text_embs.todense()

        self.features = text_embs.astype(float)

    def get_data(self):
        return {
            'graph': self.graph,
            'features': self.features,
            'ids': self.ids,
            'labels': self.labels
        }

