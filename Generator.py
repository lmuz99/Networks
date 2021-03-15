import numpy as np
import scipy as sp
from numba import njit
import networkx as nx
import timeit

def GenerateInitial(nodes, m):
    #Method to generate initial network before BA attachment with nodes and m edges per node
    edge_list = np.array(np.empty(0))
    sampling_list = np.empty(0)
    n = 0       #number of nodes in network
    nubs = 0    #keep track of number of entries of vert_list without calling len function

    
    return edge_list, vert_list, n, nubs

def SampleNode(sample_list, max_val, number):
    pass

def BA(n_total, m, n_initial):
    
    #Generate initial network as random with n_inital nodes of m edges
    edge_list, sampling_list, n, nubs = GenerateInitial(nodes, m)

    pass


