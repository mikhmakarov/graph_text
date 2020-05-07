import networkx as nx
import pandas as pd
import numpy as np
import scipy

from copy import deepcopy

from sklearn.preprocessing import LabelEncoder


class Dataset(object):
    def __init__(self, graph_path, texts_path, labels_path, partial=False, sep='\t'):
        texts_df = pd.read_csv(texts_path, header=None, sep=sep, names=['id', 'text'], index_col='id')
        labels_df = pd.read_csv(labels_path, header=None, sep=sep, names=['id', 'label'], index_col='id')
        df = pd.merge(texts_df, labels_df, left_index=True, right_index=True)

        graph = nx.from_edgelist(pd.read_csv(graph_path, sep=sep).values.tolist())

        # isolated nodes
        for _id in texts_df.index.values:
            if _id not in graph.nodes():
                graph.add_node(_id)

        if partial:
            # nodes without attributes
            for _id in graph.nodes():
                if _id not in df.index:
                    df.loc[_id] = ('', -1)

        node_map = {n: i for i, n in enumerate(graph.nodes)}
        self.graph = nx.relabel_nodes(graph, node_map)

        ids = df.index.values
        ids = np.vectorize(node_map.get)(ids)

        texts = df['text'].values
        labels = df['label'].values

        le = LabelEncoder()
        labels = le.fit_transform(labels)
        if partial:
            # find where -1 was mapped
            na_mapping = le.transform([-1])[0]
            labels = np.array([-1 if label == na_mapping else label for label in labels])

        sort_indx = ids.argsort()

        self.texts = texts[sort_indx]
        self.ids = ids[sort_indx].astype(int)
        self.labels = labels[sort_indx].astype(int)
        self.features = self.texts

        # nodes with non empty text and label
        self.main_ids = self.ids[self.labels != -1]
        self.main_labels = self.labels[self.labels != -1]

        self.n_classes = len(np.unique(self.main_labels))

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
            'labels': self.labels,
            'main_ids': self.main_ids,
            'main_labels': self.main_labels,
            'n_classes': self.n_classes
        }

    def get_lp_data(self, test_ratio, seed):
        edges = deepcopy(list(self.graph.edges()))

        n_test = int(test_ratio * len(edges))
        n_train = len(edges) - n_test

        np.random.seed(seed)
        np.random.shuffle(edges)

        test_edges = edges[:n_test]
        test_target = [1 for _ in test_edges]
        train_edges = edges[n_test:]
        train_target = [1 for _ in train_edges]

        train_graph = deepcopy(self.graph)

        for edge in test_edges:
            train_graph.remove_edge(*edge)

        i = 0
        while i < n_test:
            u = np.random.randint(0, len(self.graph.nodes()) - 1)
            v = np.random.randint(u + 1, len(self.graph.nodes()))

            if (u, v) not in self.graph.edges() and (u, v) not in test_edges:
                test_edges.append((u, v))
                test_target.append(0)
                i += 1

        j = 0
        while j < n_train:
            u = np.random.randint(0, len(self.graph.nodes()) - 1)
            v = np.random.randint(u + 1, len(self.graph.nodes()))

            if (u, v) not in self.graph.edges() and (u, v) not in test_edges and (u, v) not in train_edges:
                train_edges.append((u, v))
                train_target.append(0)
                j += 1

        return {
            'graph': train_graph,
            'features': self.features,
            'ids': self.ids,
            'labels': self.labels,
            'main_ids': self.main_ids,
            'main_labels': self.main_labels,
            'train_edges': train_edges,
            'test_edges': test_edges,
            'train_target': train_target,
            'test_target': test_target,
            'n_classes': self.n_classes
        }




