import numpy as np
import scipy as sp
import random
from numba import jit
from numba import prange
from numba.typed import List
import timeit
import pandas as pd
import matplotlib.pyplot as plt

@jit(nopython = True)
def GenerateRandom(nodes, m):
    """
    Method to generate initial random network, method generates prospective edges in 
    parallel before checking for duplicates and removing these, generating more edges
    until m*nodes unique edges have been created. These are then transferred from an
    edge list to an adjacency list format, with a sampling list and count of edges
    returned.

    Parameters
    ----------
    nodes : Number of nodes to generate for this initial network
    m : Number of edges per node to attach on average, s.t n*m edges are always created

    Returns
    -------
    adj_list : Adjacency list representation of the edges of the network
    sampling_list : Array listing the node location of each nub explicitely for PA sampling
    freq_list : Array of number of nubs per node
    """
    
    edges = nodes * m

    edge_list = randomEdges(edges, nodes-1)     #max node index will be nodes-1
    adj_list = [[np.int64(x) for x in range(0)] for i in range(nodes)]      #explicitely typing blank list of lists for numba acceleration
    
    adj_list, count = edgeToAdjacency(edge_list, adj_list)   #get adjacency list from edge list
    
    while count != 0:   #whilst we duplicate edges that need replacing, replace these
        edge_list = randomEdges(count, nodes-1)
        adj_list, count = edgeToAdjacency(edge_list, adj_list)   #update count with updated number of duplicate entries following replacing of initial duplicates
   
    #sampling_list = List()
    freq_list = [np.uint32(len(x)) for x in adj_list]
    
    #for i in range(len(freq_list)):
     #   for j in range(freq_list[i]):
      #      sampling_list.append(i)
    
    #convert to adj list
    #get n and nubs
    #get sample list as len of each element of edge list


    return adj_list, freq_list

@jit(nopython = True, parallel = True)
def GenerateInitial(m, nodes_final):
    """
    Generates a single complete graph with m edges per node, and so this method always
    returns a network of m+1 nodes, with 1 unique network per value of m

    Parameters
    ----------
    m : Number of edges per node, all m+1 nodes will have m edges
    nodes_final: Used for initialising length of adj_list, final number of nodes in network, must be > m

    Returns
    -------
    adj_list : Adjacency list representation of the edges of the network
    sampling_list : Array listing the node location of each nub explicitely for PA sampling
    freq_list : Array of number of nubs per node
    """
    if(nodes_final <= m):
        raise ValueError("ERROR: Nodes must be greater than m")
    
    adj_list = [[np.int64(x) for x in range(0)] for i in range(nodes_final)]      #explicitely typing blank list of lists for numba acceleration

    for i in prange(m+1):
        edge_list = [x for x in range(m+1) if x != i]
        adj_list[i] = edge_list

    sampling_list = List()
    freq_list = [np.uint32(len(x)) for x in adj_list]
    
    for i in range(len(freq_list)):
        for j in range(freq_list[i]):
            sampling_list.append(i)

    return adj_list, sampling_list, freq_list


@jit(nopython = True)
def randomSample(max_index):
    """
    Randomly sample a new edge between a pair of nodes, repeating if we get a self loop.
    Runs in parallel and is used when generating initial and ER graphs. Returns an edge
    of node1 and node2.
    """
    x1 = random.randint(0, max_index)
    x2 = random.randint(0, max_index)
    while x1 == x2:
        x2 = random.randint(0, max_index)
        
    return (x1, x2)

@jit(nopython = True)
def BASample(n_edges, node_id, sampling_list):
    
    max_index = len(sampling_list) - 1   
    edge_list = np.zeros(shape=(n_edges,2), dtype= np.int_) #dtype as int32 for performance so long as n_edges < 2 billion
    
    for i in range(n_edges):      #create edges for this new node in parallel
        edge_list[i] = (sampling_list[random.randint(0, max_index)], node_id)
    
    return edge_list

@jit(nopython = True)
def MixedSample(n_edges, node_id):

    edge_list = np.zeros(shape=(n_edges,2), dtype= np.int_) #dtype as int32 for performance so long as n_edges < 2 billion
    
    for i in range(n_edges):      #create edges for this new node in parallel
        edge_list[i] = (random.randint(0, node_id-1), node_id)
    
    return edge_list

@jit(nopython = True, parallel = True)
def randomEdges(n_edges, max_index):
    """
    Creates an list of length n_edges of unique edges in parallel with random attatchment 
    """
    edge_list = np.zeros(shape=(n_edges,2), dtype= np.int32) #dtype as int32 for performance so long as n_edges < 2 billion
    for i in prange(n_edges):       #create all edges in parallel
        edge_list[i] = randomSample(max_index)

    return edge_list


@jit(nopython = True)
def edgeToAdjacency(edge_list, adj_list):
    """
    This will do for now, but can potentially speed up by producing a density matrix
    by creating a numpy 2d array and filling values, more memory intensive but
    no costly creation of empty lists
    alternatively find a good sorting algo and then we can create arrays as we go rather than needing to initalise empty ones from the start
    """
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
def edgeToAdjacencyBA(edge_list, adj_list, sampling_list, freq_list, first_attempt):
    """
    This will do for now, but can potentially speed up by producing a density matrix
    by creating a numpy 2d array and filling values, more memory intensive but
    no costly creation of empty lists
    alternatively find a good sorting algo and then we can create arrays as we go rather than needing to initalise empty ones from the start
    """
    count = 0
    
    for a in edge_list:
        #only need check for one of the nodes as always write both lists together
        valid = True
        if len(adj_list[a[0]]) != 0:
            if isEdgePresent(a[1], adj_list[a[0]]) == True:
                valid = False
                count += 1
                #print(adj_list)
                #print(a[])
                #get new pair and check this is unique
                #is_pair_unique = False
                #while not is_pair_unique:
                    #generate new pair
                    #if not filled then 
                #   pass
        if valid:
            adj_list[a[0]].append(a[1])
            adj_list[a[1]].append(a[0])
            sampling_list.append(a[0])
            sampling_list.append(a[1])
            freq_list[a[0]] += 1
            freq_list[a[1]] += 1

    return adj_list, sampling_list, freq_list, count


@jit(nopython = True)
def isEdgePresent(val, list):
    for i in range(len(list)):
        if list[i] == val:
            return True
    return False
        
@jit(nopython = True)
def BA(n_total, m):
    
    #Generate initial network as random with n_inital nodes of m edges
    adj_list, sampling_list, freq_list = GenerateInitial(m, n_total)
    n_current = m
    
    for i in range(n_total - m - 1):   
        n_current += 1
        edge_list = BASample(m, n_current, sampling_list)
        
        adj_list, sampling_list, freq_list, count = edgeToAdjacencyBA(edge_list, adj_list, sampling_list, freq_list, True)   #get adjacency list from edge list
        
        while count != 0:
            edge_list = BASample(count, n_current, sampling_list)
            adj_list, sampling_list, freq_list, count = edgeToAdjacencyBA(edge_list, adj_list, sampling_list, freq_list, False)
        
    return adj_list, sampling_list, freq_list

@jit(nopython = True)
def Mixed(n_total, m, p_BA):
    
    #Generate initial network as random with n_inital nodes of m edges
    adj_list, sampling_list, freq_list = GenerateInitial(m, n_total)
    n_current = m
    
    for i in range(n_total - m - 1):   
        n_current += 1
        isBANode = False
        #Depending on random value then choose sampling method
        if random.random() > p_BA:
            edge_list = MixedSample(m, n_current)

        else: 
            edge_list = BASample(m, n_current, sampling_list)
            isBANode = True
        
        adj_list, sampling_list, freq_list, count = edgeToAdjacencyBA(edge_list, adj_list, sampling_list, freq_list, True)   #get adjacency list from edge list
        
        while count != 0:
            if isBANode:
                edge_list = BASample(count, n_current, sampling_list)
            else:
                edge_list = MixedSample(count, n_current)

            adj_list, sampling_list, freq_list, count = edgeToAdjacencyBA(edge_list, adj_list, sampling_list, freq_list, False)
        
    return adj_list, sampling_list, freq_list

#adj, samp, freq = BA(200000, 9)
adj, samp, freq = Mixed(100, 3, (2/3))
#adj, freq = GenerateRandom(10000000, 2)
#need to change appending to adjacency list to add a new node


