import networkx as nx
import numpy as np

from typing import Optional


class BaseModel(object):
    def __init__(self, graph: nx.Graph, features: np.array, dim: int, labels: Optional[np.array] = None):
        self.graph = graph
        self.features = features
        self.dim = dim
        self.labels = labels
        self.embeddings = None

    def learn_embeddings(self):
        pass

    def get_embeddings_for_ids(self, ids):
        result = []
        for i, embedding in enumerate(self.embeddings):
            if i in ids:
                result.append(embedding)

        return result
