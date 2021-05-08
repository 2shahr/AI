import pandas as pd
import numpy as np
from info_gain import info_gain
from sklearn.svm import LinearSVC
import copy
from itertools import combinations
from sklearn import tree
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from imblearn.metrics import geometric_mean_score
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.filterwarnings("ignore")

def load_dna(file_adr):
    df = pd.read_csv(file_adr, header=None, sep=',\s*', engine='python')
    y = df.iloc[:, 0].factorize()[0]
    x = df.iloc[:, 2].values
    x = [[ord(j) - ord('A') for j in i] for i in x]
    x = np.array(x)
    shape = x.shape
    x = pd.factorize(x.reshape(-1,))[0]
    x = x.reshape(shape)
    
    ig = np.zeros((x.shape[1],))
    for i in range(x.shape[1]):
        ig[i] = info_gain.info_gain(x[:, i], y)
        
    return x, y, ig

def load_coding(file_adr):
    df = pd.read_csv(file_adr, header=None, sep=',\s*', engine='python')
    y = df.iloc[:, 0].factorize()[0]
    x = df.iloc[:, 1:].values

    ig = np.zeros((x.shape[1],))
    for i in range(x.shape[1]):
        ig[i] = info_gain.info_gain(x[:, i], y)
        
    return x, y, ig

def load_satimage(file_adr):
    df = pd.read_csv(file_adr, header=None, sep='\s*', engine='python')
    y = df.iloc[:, -1].factorize()[0]
    x = df.iloc[:, :-1].values

    ig = np.zeros((x.shape[1],))
    for i in range(x.shape[1]):
        ig[i] = info_gain.info_gain(x[:, i], y)
        
    return x, y, ig


class tree_node:
    def __init__(self, feat_ind=None, unq_values=None, is_leaf=False, label=None, clf=None):
        self.feat_ind = feat_ind
        self.is_leaf = is_leaf
        self.branch = []
        self.unq_values = unq_values
        self.clf = clf
        self.label = label



def build_level_tree(x, y, l, j, h, base_clf, n_min=4):
    if x.shape[0] <= n_min or np.unique(y).shape[0] == 1:
        return tree_node(is_leaf=True, label=stats.mode(y)[0][0])
    
    if j == h:
        clf = copy.deepcopy(base_clf)
        clf.fit(x, y)
        return tree_node(is_leaf=True, clf=clf)
    
    feat_ind = l[j]
    unq_values = np.unique(x[:, feat_ind])
    
    node = tree_node(feat_ind, unq_values)
    
    for val in unq_values:
        ind = x[:, feat_ind]==val
        x_f = x[ind]
        y_f = y[ind]
        
        node.branch.append(build_level_tree(x_f, y_f, l, j + 1, h, base_clf))
        
    return node


def classify_with_level_tree(tree, x):
    labels = np.zeros((x.shape[0],))
    for i in range(x.shape[0]):
        node = copy.deepcopy(tree)
        while node.is_leaf == False:
            feat_ind = node.feat_ind
            try:
                b_ind = np.where(node.unq_values == x[i, feat_ind])[0][0]
            except:
                b_ind = 0
            node = node.branch[b_ind]
        if node.label == None:
            labels[i] = node.clf.predict(x[i, :].reshape(1, -1))
        else:
            labels[i] = node.label
            
    return labels


def feating(x, y, ig, h, base_clf):
    feat_list = [i for i in range(x.shape[1])]
    
    trees = []
    for i, cmb in enumerate(combinations(feat_list, h)):
        l = np.array(cmb)
        order = ig[l].argsort()[::-1]
        l = list(l[order])
        trees.append(build_level_tree(x=x, y=y, l=l, j=0, h=h, base_clf=base_clf, n_min=4))
        
    return trees


def classify_with_feating(trees, x):
    prd = np.zeros((x.shape[0], len(trees)))
    labels = np.zeros((x.shape[0],))
    p = np.zeros((x.shape[0],))
    for i, tree in enumerate(trees):
        prd[:, i] = classify_with_level_tree(tree, x)
    for i in range(x.shape[0]):
        labels[i] = stats.mode(prd[i, :])[0][0]
        p[i] = np.mean(prd[i, :] == labels[i])
        
    return labels, p

def report_measures(x, y, ig, h, base_clf, x_tst=None, y_tst=None):
    if x_tst is None:
        trn_x, tst_x, trn_y, tst_y = train_test_split(x, y, train_size = 0.7, stratify = y)
    else:
        trn_x = x
        trn_y = y
        tst_x = x_tst
        tst_y = y_tst

    unq_values, count = np.unique(y, return_counts=True)
    i = count.argmin()
    trees = feating(trn_x, trn_y, ig, h, base_clf)
    prd, p = classify_with_feating(trees, tst_x)
    prc, recall, fscore, _ = precision_recall_fscore_support(tst_y, prd, labels=[unq_values[i]])
    auc = roc_auc_score(tst_y==unq_values[i], p)
    g_mean = geometric_mean_score(tst_y, prd)
    return prc[0], recall[0], fscore[0], auc, g_mean


print('sat_image')
sat_trn_file_adr = 'satimage/sat.trn'
sat_tst_file_adr = 'satimage/sat.tst'
x_sat_trn, y_sat_trn, ig = load_satimage(sat_trn_file_adr)
x_sat_tst, y_sat_tst, _ = load_satimage(sat_tst_file_adr)

base_clfs = [tree.DecisionTreeClassifier(), KNeighborsClassifier(), LinearSVC(max_iter=10000)]
n_itr = 1

for h in [1, 2, 3]:
    for base_clf in base_clfs:
        prc, recall, fscore, auc, g_mean = (0, 0, 0, 0, 0)
        for i in range(n_itr):
            p, r, f, a, g = report_measures(x_sat_trn, y_sat_trn, ig, h, base_clf, x_tst=x_sat_tst, y_tst=y_sat_tst)
            prc += p
            recall += r
            fscore += f
            auc += a
            g_mean += g

        prc /= n_itr
        recall /= n_itr
        fscore /= n_itr
        auc /= n_itr
        g_mean /= n_itr

        print('h: {}, base clf : {}, Precision: {:.2f}, Recall: {:.2f}, F-Score: {:.2f}, AUC: {:.2f}, G-Mean: {:.2f}'.format(h, str(base_clf), prc, recall, fscore, auc, g_mean))


file_adr = 'dna/splice.data'
print('dna')
x, y, ig = load_dna(file_adr)


n_itr = 10

for h in [1, 2, 3]:
    for base_clf in base_clfs:
        prc, recall, fscore, auc, g_mean = (0, 0, 0, 0, 0)
        for i in range(n_itr):
            p, r, f, a, g = report_measures(x, y, ig, h, base_clf)
            prc += p
            recall += r
            fscore += f
            auc += a
            g_mean += g

        prc /= n_itr
        recall /= n_itr
        fscore /= n_itr
        auc /= n_itr
        g_mean /= n_itr

        print('h: {}, base clf : {}, Precision: {:.2f}, Recall: {:.2f}, F-Score: {:.2f}, AUC: {:.2f}, G-Mean: {:.2f}'.format(h, str(base_clf), prc, recall, fscore, auc, g_mean))



file_adr = 'coding/letter-recognition.data'
print('coding')
x, y, ig = load_coding(file_adr)

n_itr = 10

for h in [1, 2, 3]:
    for base_clf in base_clfs:
        prc, recall, fscore, auc, g_mean = (0, 0, 0, 0, 0)
        for i in range(n_itr):
            p, r, f, a, g = report_measures(x, y, ig, h, base_clf)
            prc += p
            recall += r
            fscore += f
            auc += a
            g_mean += g

        prc /= n_itr
        recall /= n_itr
        fscore /= n_itr
        auc /= n_itr
        g_mean /= n_itr

        print('h: {}, base clf : {}, Precision: {:.2f}, Recall: {:.2f}, F-Score: {:.2f}, AUC: {:.2f}, G-Mean: {:.2f}'.format(h, str(base_clf), prc, recall, fscore, auc, g_mean))
