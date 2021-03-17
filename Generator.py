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

    for i in nodes:
        for j in m:
            #choose n*m random pairs from the N(N-1)/2 pairs
            #check we haven't chose pairs twice
            #work out which pairs these correspond to
            #fill in edge list
            #leave opportunity to break here for analysis

    return edge_list, sampling_list, n, nubs

def SampleNode(sample_list, max_val, number):
    pass

def BA(n_total, m, n_initial):
    
    #Generate initial network as random with n_inital nodes of m edges
    edge_list, sampling_list, n, nubs = GenerateInitial(n_initial, m)

    pass


