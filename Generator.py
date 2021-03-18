import numpy as np
import scipy as sp
import random
from numba import jit
from numba import cuda
from numba import prange
from numba.typed import List
import networkx as nx
import timeit
import cProfile

@jit(nopython = True)
def GenerateInitial(nodes, m):
    #Method to generate initial network before BA attachment with nodes and m edges per node
    n = 0       #number of nodes in network
    nubs = 0    #keep track of number of entries of vert_list without calling len function
    edges = nodes * m

    edge_list = randomEdges(edges, nodes-1)
    adj_list = edgeToAdjacency(edge_list, nodes)
    sampling_list = [len(x) for x in adj_list]
    #convert to adj list
    #get n and nubs
    #get sample list as len of each element of edge list


    return adj_list, sampling_list, n, nubs


@jit(nopython = True)
def randomSample(max_index):
    #randomly sample a new edge between a pair of nodes, repeating if we get a self loop

    x1 = random.randint(0, max_index)
    x2 = random.randint(0, max_index)
    while x1 == x2:
        x2 = random.randint(0, max_index)
        
    return (x1, x2)


@jit(nopython = True, parallel = True)
def randomEdges(n_edges, max_index):
    #Creates an edge list as this parallelises random number generation

    edge_list = np.zeros(shape=(n_edges,2), dtype= np.int32)
    for i in prange(n_edges):
        edge_list[i] = randomSample(max_index)

    return edge_list

@jit(nopython = True)
def edgeToAdjacency(edge_list, nodes):
    
    #this will do for now, but can potentially speed up by producing a density matrix
    #by creating a numpy 2d array and filling values, more memory intensive but
    #no costly creation of empty lists

    #alternatively find a good sorting algo and then we can create arrays as we go rather than needing to initalise empty ones from the start

    adj_list = [[np.int32(x) for x in range(0)] for i in range(nodes)]

    #could split this up/ sort to allow for multithreading down the line
    for a in edge_list:
        #only need check for one of the nodes as always write both lists together
        if isEdgePresent(a[1], a[0]):
            #get new pair and check this is unique
            is_pair_unique = False
            while not is_pair_unique:
                #generate new pair
                #if not filled then 
                pass
        else:
            adj_list[a[0]].append(a[1])
            adj_list[a[1]].append(a[0])


    return adj_list

@jit(nopython = True, parallel = True)
def isEdgePresent(val, list):
    for i in prange(len(list)):
        if list[i] == val:
            return True
        else:
            return False
        


def BA(n_total, m, n_initial):
    
    #Generate initial network as random with n_inital nodes of m edges
    edge_list, sampling_list, n, nubs = GenerateInitial(n_initial, m)

    pass


