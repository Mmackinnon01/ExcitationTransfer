
import networkx as nx
import numpy as np


def random_adjacency(n, p, seed=None):
    return nx.to_numpy_array(nx.erdos_renyi_graph(n , p , seed = seed)).astype(np.complex128)
