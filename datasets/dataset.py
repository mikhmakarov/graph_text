import networkx as nx
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


class Dataset(object):
    def __init__(self, graph_path, texts_path, text_bow=True, sep='\t'):
        graph = nx.from_edgelist(pd.read_csv(graph_path, sep=sep).values.tolist())
        node_map = {n: i for i, n in enumerate(graph.nodes)}
        self.graph = nx.relabel_nodes(graph, node_map)

        if text_bow:
            features = pd.read_csv(texts_path, header=None, sep=sep)
            ids = np.array(features)[:, 0]
            ids = np.vectorize(node_map.get)(ids)

            le = LabelEncoder()
            labels = le.fit_transform(np.array(features)[:, -1])
            features = np.array(features)[:, 1:-1]
        else:
            raise ValueError('Not implemented')

        sort_indx = ids.argsort()
        self.features = features[sort_indx]
        self.ids = ids[sort_indx]
        self.labels = labels[sort_indx]

    def get_data(self):
        return {
            'graph': self.graph,
            'features': self.features,
            'ids': self.ids,
            'labels': self.labels
        }

