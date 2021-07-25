import matplotlib.pyplot as plt
import numpy as np
import pickle
import h5py
from scipy.io import loadmat
import pandas as pd
from sklearn import decomposition
from sklearn.metrics import average_precision_score
import pickle
from sklearn.metrics import precision_recall_curve


EPS = 1e-8
USE_KDIST = True
class LOF():
    def __init__(self):
        self.pair_distances = None
        self.sorted_indeces = None

    def fit(self, x):
        self.pair_distances = np.power(((x[:, None] - x) ** 2).sum(axis=2), 0.5) #is needed? AS ** 2
        self.l = self.pair_distances.shape[0]
        self.kdist = np.sort(self.pair_distances, axis=1)
        self.reduce_diag = ((np.diag(np.ones(self.pair_distances.shape[0])) + 1) % 2).astype(np.int32)
        #print(self.rd[3])

    def score(self, k):
        k_distances = self.kdist[:, k + 1].reshape(-1, 1)
        N_k_matrix = (self.pair_distances <= k_distances) * self.reduce_diag
        nof_N_k = N_k_matrix.sum(axis=1)
        if USE_KDIST:
            reach_dist = self.pair_distances #.copy()
            reach_dist[N_k_matrix == 1] = np.repeat(k_distances.reshape(-1), nof_N_k) # копия не нужна т к kdist увеличивается с ростом к
        else:
            reach_dist = self.pair_distances
        lrd = nof_N_k / ((reach_dist * N_k_matrix).sum(axis=1) + EPS)
        LOF = (np.repeat(lrd[None, :], N_k_matrix.shape[0], axis=0) * N_k_matrix).sum(axis=1) / nof_N_k / lrd
        return np.abs(LOF - 1) # LOF # может ли быть < 1?


if __name__ == "__main__":
    pass