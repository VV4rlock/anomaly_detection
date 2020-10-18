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


TEST = 1
class kNN():
    def __init__(self):
        self.pair_distances = None
        self.sorted_indeces = None

    def fit(self, x):
        self.pair_distances = ((x[:, None] - x) ** 2).sum(axis=2) **0.5 #is needed? AS ** 2
        self.sorted_indeces = np.argsort(self.pair_distances, axis=1)
        print(self.sorted_indeces.shape)

    def score(self, k):
        # main diag is 0, lookslile k = k + 1
        '''ll return AS for every vector in X'''
        return self.pair_distances[np.arange(self.sorted_indeces.shape[0]), self.sorted_indeces[:, k + 1]]


def generate_claster2d_data(center, count, dispersion=1):
    return np.stack((np.random.uniform(center[0] - dispersion,center[0] + dispersion,(count,1)),
                     np.random.uniform(center[1]- dispersion,center[1] + dispersion,(count,1))),
                    axis=1).reshape(count,-1)


def average_precisiion(score, y, draw=False):
    sorted_indices = np.argsort(score)[::-1]
    sorted_labels = y[sorted_indices]
    #print(sorted_labels)
    tp_fp_thresholds = np.where((sorted_labels == 1).reshape(-1))[0] + 1
    tp_fn = len(tp_fp_thresholds)

    tp = np.arange(tp_fn) + 1
    #precision = (tp / tp_fp_thresholds)[::-1]
    recall = np.insert(tp / tp_fn, 0, 0)
    precision = np.insert((tp / tp_fp_thresholds)[::-1], -1, 1)

    #p = np.where(np.diff(precision) > 0)[0]
    #while(len(p)> 0):
    #    precision[p] = precision[p+1]
    #    p = np.where(np.diff(precision) > 0)[0]
    precision = np.maximum.accumulate(precision)
    precision = precision[::-1]

    #AP = (np.diff(recall) * precision).sum() + recall[0]*precision[0]
    AP = np.trapz(precision, recall)

    pr_rec1 = np.stack((precision, recall), axis=1)  # np.array(pr_rec)
    if draw:
        d = pd.DataFrame(pr_rec1, columns=['precision', 'recall'])
        plt.plot(pr_rec1[:,1], pr_rec1[:,0])

        pr,rec,th = precision_recall_curve(y,score)
        plt.plot(rec, pr)
        plt.show()
    #print(score[sorted_indices][thresholds])
    return AP, pr_rec1, score[sorted_indices][tp_fp_thresholds-1]

MATFILE = False
if MATFILE:
    MAMMOGRAPHY = '/home/warlock/projects/third_sem/anomaly_detection/mammography.mat'
    SAT = '/home/warlock/projects/third_sem/anomaly_detection/satimage-2.mat'
else:
    MAMMOGRAPHY = '/home/warlock/projects/third_sem/anomaly_detection/mammography'
    SAT = '/home/warlock/projects/third_sem/anomaly_detection/sat'


DATASET = SAT#MAMMOGRAPHY
if __name__ == "__main__":
    global Y
    #file = h5py.File(DATASET, 'r')


    if MATFILE:
        data = loadmat(DATASET)
        X = data['X']
        Y = data['y']
        Y = Y.reshape(-1)
    else:
        with open(DATASET, "rb") as f:
            data = pickle.load(f)
        X = data['vectors']
        Y = data['labels']
        Y = Y.reshape(-1)

    #X = X[-1000:]
    #Y = Y[-1000:]
    print(f"statistic:\n\tnof_vectors: {len(X)}\n\tx_dimensions={len(X[0])}\n\ttnrof1: {len(Y[Y==1])}\n\tnrof0: {len(Y[Y==0])}")



    from anomaly_detection.LOF import LOF
    knn = LOF()#kNN()
    if TEST:
        np.random.seed(0)
        cl1 = generate_claster2d_data((1, 1), 20, dispersion=0.15)
        cl2 = generate_claster2d_data((8, 7), 30, dispersion=0.7)
        anomaly = np.array([[1, 8], [3, 3], [4, 1], [2,1]])
        d = pd.DataFrame({
            'x': np.concatenate((cl1[:, 0], cl2[:, 0], anomaly[:, 0])),
            'y': np.concatenate((cl1[:, 1], cl2[:, 1], anomaly[:, 1])),
        })

        orig_labels = np.concatenate((np.zeros(len(cl1)), np.zeros(len(cl2)) + 2, np.zeros(len(anomaly)) + 1))
        Y = orig_labels % 2
        knn.fit(d.to_numpy())

        labels = knn.score(2)
        #print(labels)
        d['cl'] = labels
        d.plot.scatter('x', 'y', c='cl', colormap='rainbow')
        plt.show()
        exit(0)
    else:
        knn.fit(X)

        pca = decomposition.PCA(n_components=2)
        pca_X = pca.fit_transform(X)
        d = pd.DataFrame(pca_X, columns=['x','y'])

    #validating...
    print("validating...")
    max_ap, max_prc, max_k, edges_score = 0, None, 0, None
    for k in range(1, len(X) - 1):
        score = knn.score(k)
        skap = average_precision_score(Y,score)
        ap, prc, edges = average_precisiion(score, Y, draw=False)
        if ap > max_ap:
            max_ap, max_prc, max_k, edges_score, max_score = ap, prc, k, edges, score
            print(f"\tnew Max AP: {max_ap} k={max_k} sklearn_AP:{skap}")

    f1_score = [pr*rec/(pr+rec) for pr, rec in max_prc] #
    maxf1 = max(f1_score)
    index_of_maxf1 = f1_score.index(maxf1)
    edge = edges_score[index_of_maxf1]
    labels = np.zeros(len(X))
    labels[max_score >= edge] = 1
    labels[Y == 1] += 2 # 1 - FP, 2 - FN, 3 - TP
    tp, fp, fn, tn = (labels == 3).sum(), (labels == 1).sum(), (labels == 2).sum(), (labels == 0).sum()
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    confusion_matrix = np.array([[tp, fp],[fn, tn]])
    print(f"confusion matrix:\n{confusion_matrix}")

    print(f"precision={tp/(tp+fp)} recall={tp/(tp+fn)}")

    print(f"result: k={max_k}, f1={2 * maxf1}, anomaly_edge={edge}")

    plt.figure()
    last = max_prc[0]
    for i in range(1, len(max_prc)):
        cur = max_prc[i]
        plt.plot([last[1], last[1], cur[1]],[last[0], cur[0], cur[0]], 'r-')
        last = cur
    plt.plot([0, max_prc[0, 1],max_prc[0, 1]], [1, 1, max_prc[0, 0]],'r-')
    plt.plot([1, 1], [max_prc[-1, 0], 0], 'r-')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

    d['cl'] = labels
    d.plot.scatter('x','y', c='cl', colormap='rainbow')


    plt.show()
    #print(len(Y[Y==0]))
    #with open(DATASET,'rb') as file:
    #    data = pickle.load(file)