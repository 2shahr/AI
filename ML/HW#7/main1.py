import numpy as np
from scipy.sparse import data
from sklearn.cluster import KMeans
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from copy import copy
import pandas as pd
import sys
import os


sys.setrecursionlimit(1500)


def my_mode(y):
    unq_labels = np.unique(y)
    count = [np.sum(y == label) for label in unq_labels]
    i = np.argmax(count)
    return unq_labels[i]


class tree_node:
    def __init__(self, w=None, p=None, is_leaf=False, label=None, childs=None):
        self.w = w
        self.p = p
        self.is_leaf = is_leaf
        self.label = label
        if childs is None:
            self.childs = []


class BDTKS:
    def __init__(self, lmbd, ver):
        self.root = None
        self.lmbd = lmbd
        self.ver = ver

    def fit(self, x, y):
        n_labels = int(np.max(y) + 1)
        p_r = [np.sum(y == i) / y.shape[0] for i in range(n_labels)]
        self.p_r = p_r
        self.N = y.shape[0]
        self.n_labels = n_labels

        if self.ver == "v1":
            self.root = self.make_tree_v1(x, y)
        else:
            self.root = self.make_tree_v2(x, y)

    def make_tree_v1(self, x, y):
        is_leaf = self.is_leaf_node(x, y)
        if is_leaf == True:
            label = my_mode(y)
            node = tree_node(is_leaf=True, label=label)
            return node
        else:
            kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(x)
            c = kmeans.cluster_centers_
            cluster_labels = kmeans.labels_

            w = (c[1] - c[0]) / np.linalg.norm(c[1] - c[0])
            p = np.dot(w, (c[0] + c[1]) / 2)

            idx = cluster_labels == 0
            x_1 = x[idx]
            y_1 = y[idx]

            x_2 = x[np.logical_not(idx)]
            y_2 = y[np.logical_not(idx)]

            node = tree_node()
            node.w = w
            node.p = p
            node.childs.append(self.make_tree_v1(x_1, y_1))
            node.childs.append(self.make_tree_v1(x_2, y_2))

            return node

    def make_tree_v2(self, x, y):
        is_leaf = self.is_leaf_node(x, y)
        if is_leaf == True:
            label = my_mode(y)
            node = tree_node(is_leaf=True, label=label)
            return node
        else:
            pca = PCA(n_components=1)
            pca.fit(x)
            w_ = pca.components_

            new_pos = np.dot(x, w_.T)
            p_ = np.median(new_pos)

            idx = (new_pos <= p_).squeeze()

            c = np.zeros((2, x.shape[1]))
            c[0] = np.mean(x[idx, :], axis=0)
            c[1] = np.mean(x[np.logical_not(idx), :], axis=0)

            kmeans = KMeans(n_clusters=2, init=c, n_init=1).fit(x)
            c = kmeans.cluster_centers_
            cluster_labels = kmeans.labels_

            w = (c[1] - c[0]) / np.linalg.norm(c[1] - c[0])
            p = np.dot(w, (c[0] + c[1]) / 2)

            idx = cluster_labels == 0
            x_1 = x[idx]
            y_1 = y[idx]

            x_2 = x[np.logical_not(idx)]
            y_2 = y[np.logical_not(idx)]

            node = tree_node()
            node.w = w
            node.p = p
            node.childs.append(self.make_tree_v2(x_1, y_1))
            node.childs.append(self.make_tree_v2(x_2, y_2))

            return node

    def is_leaf_node(self, x, y):
        N_t = y.shape[0]
        p_t = [np.sum(y == label) / N_t for label in range(self.n_labels)]
        l_major = np.argmax(p_t)
        is_leaf = p_t[l_major] > self.p_r[l_major]

        p_r = copy(self.p_r)

        del p_t[l_major]
        del p_r[l_major]

        idx = [
            not p_t_i < p_r_i * self.lmbd * self.N / N_t ** 2
            for p_t_i, p_r_i in zip(p_t, p_r)
        ]

        is_leaf = is_leaf and np.sum(idx) == 0

        return is_leaf

    def predict(self, x):
        prd = np.zeros((x.shape[0],))
        for i, s in enumerate(x):
            node = self.root
            while node.is_leaf == False:
                if np.dot(np.expand_dims(node.w, axis=0), s.T) - node.p < 0:
                    node = node.childs[0]
                else:
                    node = node.childs[1]
            prd[i] = node.label
        return prd


def load_data(file_name):
    adr = "datasets/" + file_name
    data = load_svmlight_file(adr)

    trn_x = data[0].todense()
    trn_y = data[1]

    trn_x, unq_ind = np.unique(trn_x, axis=0, return_index=True)
    trn_y = trn_y[unq_ind]

    trn_y, unq_labels = pd.factorize(trn_y, sort=True)
    if os.path.isfile(adr + ".t"):
        data = load_svmlight_file(adr + ".t")
        tst_x = data[0].todense()
        tmp = data[1]
        tst_y = np.zeros(tmp.shape)

        for i, label in enumerate(unq_labels):
            tst_y[tmp == label] = i
    else:
        trn_x, tst_x, trn_y, tst_y = train_test_split(
            trn_x, trn_y, test_size=0.2, stratify=trn_y
        )

    trn_x, val_x, trn_y, val_y = train_test_split(
        trn_x, trn_y, test_size=0.2, stratify=trn_y
    )

    return trn_x, trn_y, tst_x, tst_y, val_x, val_y


data_sets = [
    "acoustic",
    "aloi",
    "combined",
    "covtype",
    "gisette",
    "ijcnn1",
    "letter",
    "madelon",
    "mushrooms",
    "pendigits",
    "phishing",
    "satimage",
    "segment",
    "seismic",
    "shuttle",
    "skin_nonskin",
    "usps",
    "w8a",
    
]

lmbds = [10 ** i for i in range(-6, 4)]
n_itr = 5
for ds in data_sets:
    trn_x, trn_y, tst_x, tst_y, val_x, val_y = load_data(ds)
    for ver in ["v2", "v1"]:
        acc = np.zeros((len(lmbds),))
        for i, lmbd in enumerate(lmbds):
            tree = BDTKS(lmbd, ver)
            tree.fit(trn_x, trn_y)
            prd = tree.predict(val_x)
            acc[i] = np.mean(prd == val_y)
        i = np.argmax(acc)
        best_lmbd = lmbds[i]
        #best_lmbd = 1

        trn_x = np.concatenate((trn_x, val_x), axis=0)
        trn_y = np.concatenate((trn_y, val_y), axis=0)
        if ver == "v2":
            tree = BDTKS(best_lmbd, ver)
            tree.fit(trn_x, trn_y)
            prd = tree.predict(tst_x)
            acc = np.mean(prd == tst_y)
            prc, recall, f1, _ = precision_recall_fscore_support(
                tst_y, prd, average="macro"
            )
            print(
                f"dataset: {ds}, tree version: {ver}, accuracy: {acc}, precision: {prc}, recall: {recall}, f1: {f1}"
            )
        else:
            prc = np.zeros((n_itr,))
            recall = np.zeros((n_itr,))
            f1 = np.zeros((n_itr,))
            acc = np.zeros((n_itr,))
            for i in range(n_itr):
                tree = BDTKS(best_lmbd, ver)
                tree.fit(trn_x, trn_y)
                prd = tree.predict(tst_x)
                acc[i] = np.mean(prd == tst_y)
                prc[i], recall[i], f1[i], _ = precision_recall_fscore_support(
                    tst_y, prd, average="macro"
                )
            print(
                f"dataset: {ds}, tree version: {ver}, accuracy: {np.mean(acc)}{chr(177)}{np.std(acc)}, precision: {np.mean(prc)}{chr(177)}{np.std(prc)}, recall: {np.mean(recall)}{chr(177)}{np.std(recall)}, f1: {np.mean(f1)}{chr(177)}{np.std(f1)}"
            )