import numpy as np
from scipy import sparse


def normalize_adjacency(graph):
    """
    Method to calculate a sparse degree normalized adjacency matrix.
    :param graph: Sparse graph adjacency matrix.
    :return A: Normalized adjacency matrix.
    """
    ind = range(len(graph.nodes()))
    degs = [1.0/graph.degree(node) for node in graph.nodes()]
    edges = [edge for edge in graph.edges()]
    index_1 = [edge[0] for edge in edges] + [edge[1] for edge in edges]
    index_2 = [edge[1] for edge in edges] + [edge[0] for edge in edges]
    values = [1.0 for _ in edges] + [1.0 for _ in edges]
    shape = (len(ind), len(ind))
    A = sparse.coo_matrix((values, (index_1, index_2)), shape=shape, dtype=np.float32)
    degs = sparse.coo_matrix((degs, (ind, ind)), shape=A.shape, dtype=np.float32)
    A = A.dot(degs)
    return A
