import Generator as Gn
import numpy as np
import scipy as sp
import random
from numba import jit
from numba import prange 
from numba.typed import List
import timeit
import pandas as pd



def RunFreqMax(n, m_min, m_max, runs):
    frequencies, maxes = Gn.BABatch(n, m_min, m_max, runs)
    a = pd.DataFrame(frequencies).value_counts(normalize=True)
    return a, maxes

@jit(nopython=True, parallel=True)
def BABatch(nodes, m_start, m_finish, batch_runs):

    frequencies = [[np.uint16(x) for x in range(0)] for i in range(batch_runs)]
    maxes = np.zeros(batch_runs, dtype=np.uint16)
    for i in prange(batch_runs):
        adj, samp, freq = Gn.BA(nodes, m_start)
        frequencies[i] = freq
        maxes[i] = max(freq)
    
    frequencies = np.array(frequencies).flatten()
    return frequencies, maxes

a, b, c = Gn.BA(100, 2)
freq, maxes = BABatch(100000, 2, 2, 1000)