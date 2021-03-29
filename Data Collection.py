import Generator as Gn
import numpy as np
import scipy as sp
import random
from numba import jit
from numba import prange 
from numba.typed import List
import timeit
import pandas as pd



def RunBA(n, m_array, runs):

    for i in m_array:
        print("Start: " + str(i))
        base_name = "BA_N" + str(n) + "m" + str(i) + "R" + str(runs)

        frequencies, maxes = BABatch(n, i, runs)

        #pk_df = pd.DataFrame(frequencies).value_counts(normalize=True).sort_index()
        k_df =  pd.DataFrame(frequencies)  
        kMax_df = pd.DataFrame(maxes)

        #pk_df.to_csv(path_or_buf=base_name+"PROB.csv")
        k_df.to_csv(path_or_buf=base_name+"FREQ.csv")
        kMax_df.to_csv(path_or_buf=base_name+"MAX.csv", index=False)

        print("Finish: " + str(i))
        
def RunER(n, m_array, runs):

    for i in m_array:
        print("Start: " + str(i))
        base_name = "ER_N" + str(n) + "m" + str(i) + "R" + str(runs)

        frequencies, maxes = ERBatch(n, i, runs)

        #pk_df = pd.DataFrame(frequencies).value_counts(normalize=True).sort_index()
        k_df =  pd.DataFrame(frequencies) 
        kMax_df = pd.DataFrame(maxes)

        #pk_df.to_csv(path_or_buf=base_name+"PROB.csv")
        k_df.to_csv(path_or_buf=base_name+"FREQ.csv")
        kMax_df.to_csv(path_or_buf=base_name+"MAX.csv", index=False)

        print("Finish: " + str(i))

def RunMixed(n, m_array, runs, p):

    for i in m_array:
        print("Start: " + str(i))
        base_name = "MIX_N" + str(n) + "m" + str(i) + "R" + str(runs) + "P23"

        frequencies, maxes = MixedBatch(n, i, runs, p)

        
        k_df =  pd.DataFrame(frequencies)  
        pk_df = k_df.value_counts(normalize=True).sort_index()
        kMax_df = pd.DataFrame(maxes)

        pk_df.to_csv(path_or_buf=base_name+"PROB.csv")
        k_df.to_csv(path_or_buf=base_name+"FREQ.csv")
        kMax_df.to_csv(path_or_buf=base_name+"MAX.csv", index=False)

        print("Finish: " + str(i))

@jit(nopython=True, parallel=True)
def BABatch(nodes, m, batch_runs):

    frequencies = [[np.uint32(x) for x in range(0)] for i in range(batch_runs)]
    maxes = np.zeros(batch_runs, dtype=np.uint32)
    for i in prange(batch_runs):
        adj, samp, freq = Gn.BA(nodes, m)
        frequencies[i] = freq
        maxes[i] = max(freq)
    
    frequencies = np.array(frequencies).flatten()
    return frequencies, maxes

@jit(nopython=True, parallel=True)
def ERBatch(nodes, m, batch_runs):

    frequencies = [[np.uint32(x) for x in range(0)] for i in range(batch_runs)]
    maxes = np.zeros(batch_runs, dtype=np.uint32)
    for i in prange(batch_runs):
        adj, freq = Gn.GenerateRandom(nodes, m)
        frequencies[i] = freq
        maxes[i] = max(freq)
    
    frequencies = np.array(frequencies).flatten()
    return frequencies, maxes

@jit(nopython=True, parallel=True)
def MixedBatch(nodes, m, batch_runs, p):

    frequencies = [[np.uint32(x) for x in range(0)] for i in range(batch_runs)]
    maxes = np.zeros(batch_runs, dtype=np.uint32)
    for i in prange(batch_runs):
        adj, samp, freq = Gn.Mixed(nodes, m, p)
        frequencies[i] = freq
        maxes[i] = max(freq)
    
    frequencies = np.array(frequencies).flatten()
    return frequencies, maxes

#freq, maxes = BABatch(20, 2, 1)
#freq, maxes = ERBatch(20, 2, 1)
#freq, maxes = MixedBatch(20, 2, 1, 0.5)
#RunFreqMax(10, [4], 1)
#freq, maxes = BABatch(10000, 20, 20, 100)
#RunBA(10000000, [4], 3)
#RunER(10000000, [4], 3)
RunMixed(100000, [4], 1000, 2/3)
RunMixed(1000000, [2,4,8,16,32], 100, 2/3)