import numpy as np
import scipy as sp
import random
from numba import jit
from numba import prange
from numba.typed import List
import networkx as nx
import timeit


@jit(nopython = True)
def GenerateInitial(nodes, m):
    #Method to generate initial network before BA attachment with nodes and m edges per node
    edges = nodes * m

    edge_list = randomEdges(edges, nodes-1)
    adj_list = [[np.int32(x) for x in range(0)] for i in range(nodes)]
    
    adj_list, count = edgeToAdjacency(edge_list, adj_list, nodes)
    
    while count != 0:
        edge_list = randomEdges(count, nodes-1)
        adj_list, count = edgeToAdjacency(edge_list, adj_list, nodes)
   
    sampling_list = List()
    freq_list = [len(x) for x in adj_list]
    
    for i in range(len(freq_list)):
        for j in range(freq_list[i]):
            sampling_list.append(i)
    
    #convert to adj list
    #get n and nubs
    #get sample list as len of each element of edge list


    return adj_list, sampling_list, edges


@jit(nopython = True)
def randomSample(max_index):
    #randomly sample a new edge between a pair of nodes, repeating if we get a self loop

    x1 = random.randint(0, max_index)
    x2 = random.randint(0, max_index)
    while x1 == x2:
        x2 = random.randint(0, max_index)
        
    return (x1, x2)

@jit(nopython = True, parallel=True)
def BASample(node_id, sampling_list):
    index = random.randint(0, len(sampling_list)-1)
    edge = np.array(sampling_list[index], node_id)
    
    sampling_list.append(edge[0])
    sampling_list.append(edge[1])
    return edge, sampling_list


@jit(nopython = True, parallel = True)
def randomEdges(n_edges, max_index):
    #Creates an edge list as this parallelises random number generation

    edge_list = np.zeros(shape=(n_edges,2), dtype= np.int32)
    for i in prange(n_edges):
        edge_list[i] = randomSample(max_index)

    return edge_list

@jit(nopython = True)
def edgeToAdjacency(edge_list, adj_list, nodes):
    
    #this will do for now, but can potentially speed up by producing a density matrix
    #by creating a numpy 2d array and filling values, more memory intensive but
    #no costly creation of empty lists
    #alternatively find a good sorting algo and then we can create arrays as we go rather than needing to initalise empty ones from the start
    
    count = 0
    
    for a in edge_list:
        #only need check for one of the nodes as always write both lists together
        valid = True
        if len(adj_list[a[0]]) != 0:
            if isEdgePresent(a[1], adj_list[a[0]]) == True:
                valid = False
                count += 1
                #get new pair and check this is unique
                #is_pair_unique = False
                #while not is_pair_unique:
                    #generate new pair
                    #if not filled then 
                #   pass
        if valid:
            adj_list[a[0]].append(a[1])
            adj_list[a[1]].append(a[0])

    return adj_list, count

@jit(nopython = True)
def isEdgePresent(val, list):
    for i in range(len(list)):
        if list[i] == val:
            return True
    return False
        


def BA(n_total, m, n_initial):
    
    #Generate initial network as random with n_inital nodes of m edges
    adj_list, sampling_list, n_edges = GenerateInitial(n_initial, m)
    n_current = n_initial
    
    for i in range(n_total - n_initial):   
        n_current += 1
        edge, sampling_list = BASample(n_current, sampling_list)
        adj_list, count = edgeToAdjacency(edge_list, adj_list, nodes)
    
    #now drive network using samoling
    pass

a, b, c = GenerateInitial(5, 1)

#NOTES: Currently need to split BA sample, into sampling and updating sampling list
#So sample, then we need to check this is not a duplicate, before finally updating sampling list

