import numpy as np
import scipy as sp
from numba import njit
import networkx as nx
import timeit
import Generator as gen

#-----------GENERATE OR LOAD NETWORK--------------#
edge_list = np.array(np.empty(0))
sampling_list = np.empty(0)
n = 0       #number of nodes in network
nubs = 0    #keep track of number of entries of vert_list without calling len function