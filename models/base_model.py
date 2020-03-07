import networkx as nx
import numpy as np


class BaseModel(object):
    def __init__(self, graph: nx.Graph, features: np.array):
        self.graph = graph
        self.features = features
        self.embeddings = None

    def learn_embeddings(self):
        pass
