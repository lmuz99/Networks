import numpy as np
import scipy as sp
import random
from numba import jit
from numba import cuda
from numba import prange
import networkx as nx
import timeit

def GenerateInitial(nodes, m):
    #Method to generate initial network before BA attachment with nodes and m edges per node
    edge_list = np.array(np.empty(0))
    sampling_list = np.empty(0)
    random_vals = np.arange(nodes)
    n = 0       #number of nodes in network
    nubs = 0    #keep track of number of entries of vert_list without calling len function

    for i in nodes:
        for j in m:
            #choose n*m random pairs from the N(N-1)/2 pairs
            #check we haven't chose pairs twice
            #work out which pairs these correspond to
            #fill in edge list
            #leave opportunity to break here for analysis
            pass

    return edge_list, sampling_list, n, nubs

@jit(nopython = True)
def randomSample(max_index):
    x1 = random.randint(0, max_index)
    x2 = random.randint(0, max_index)
    if x1 == x2:
        x2 = random.randint(0, max_index)
        
    return (x1, x2)

@jit(nopython = True, parallel = True)
def randomEdges(n_edges, max_index):
    output = np.zeros(shape=(n_edges,2), dtype= np.int64)
    for i in prange(n_edges):
        output[i] = randomSample(max_index)

    return output

        

def BA(n_total, m, n_initial):
    
    #Generate initial network as random with n_inital nodes of m edges
    edge_list, sampling_list, n, nubs = GenerateInitial(n_initial, m)

    pass


