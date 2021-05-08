import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from imblearn.metrics import geometric_mean_score
import numpy as np
import scipy.stats as ss

def add_noise(x, r_n, r_c):
    n_c = int(np.floor(r_c * x.shape[0]))
    dv = int(np.floor(x.shape[1] * r_n))
    f_inds = np.random.choice([i for i in range(x.shape[1])], size= dv, replace=False)
    x_c = x.copy()
    a = np.arange(2)
    xU, xL = a + 0.5, a - 0.5
    for f in f_inds:
        prob = ss.norm.cdf(xU, scale = 3) - ss.norm.cdf(xL, scale = 3)
        prob = prob / prob.sum()
        vals = np.random.choice(a, size = n_c, p = prob)
        s_inds = np.random.choice([i for i in range(x.shape[0])], size = n_c, replace=False)
        x_c[s_inds, f] = vals

    return x_c

def load_data(file_name):
    df = pd.read_csv(file_name)
    df = df.fillna(-1)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    return x, y

def get_a_bootstrap(x, y):
    mjr_x = x[y == -1]
    mjr_y = y[y == -1]
    
    mnr_x = x[y == 1]
    mnr_y = y[y == 1]
    
    n_mnr = mnr_x.shape[0]
    n_mjr = mjr_x.shape[0]
    
    r_mnr = np.random.randint(0, n_mnr, n_mnr)
    r_mjr = np.random.randint(0, n_mjr, n_mnr)
    
    x_bs = np.concatenate((mjr_x[r_mjr], mnr_x[r_mnr]), axis = 0)
    y_bs = np.concatenate((mjr_y[r_mjr], mnr_y[r_mnr]), axis = 0)
    
    return x_bs, y_bs


def train_bagging(n_trees, x, y):
    x, y = get_a_bootstrap(x, y)
    tree_list = []
    
    for _ in range(n_trees):
        tree = DecisionTreeClassifier()
        tree.fit(x, y)
        tree_list.append(tree)
    return tree_list

def predict_bagging(tree_list, x):
    p = 0
    for tree in tree_list:
        p += tree.predict_proba(x)
    p /= len(tree_list)
    return p

def report_measures_for_bagging(x, y, n_trees, trn_ratio, noise_dst, r_n, r_c):
    trn_x, tst_x, trn_y, tst_y = train_test_split(x, y, train_size = trn_ratio, stratify = y)

    if noise_dst == 'train':
        trn_x = add_noise(trn_x, r_n, r_c)
    elif noise_dst == 'test':
        tst_x = add_noise(tst_x, r_n, r_c)

    bag_trees = train_bagging(n_trees, trn_x, trn_y)
    p = predict_bagging(bag_trees, tst_x)
    prd = np.argmax(p, axis=1)
    prd[prd == 0] = -1
    prc, recall, fscore, _ = precision_recall_fscore_support(tst_y, prd, labels=[1])
    auc = roc_auc_score(tst_y, p[:, 1])
    g_mean = geometric_mean_score(tst_y, prd)
    return prc[0], recall[0], fscore[0], auc, g_mean

file_name = 'Covid-19.csv'
trn_ratio = 0.7
x, y = load_data(file_name)
n_itr = 10

for noise_dst in ['train', 'test']:
    for r_n in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for r_c in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for n_trees in [11, 31, 51, 101]:
                prc, recall, fscore, auc, g_mean = (0, 0, 0, 0, 0)
                for i in range(n_itr):
                    p, r, f, a, g = report_measures_for_bagging(x, y, n_trees, trn_ratio, noise_dst, r_n, r_c)
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
                
                print('Noise Added to: {}, Rn: {}, Rc: {}#Trees: {}, Precision: {:.2f}, Recall: {:.2f}, F-Score: {:.2f}, AUC: {:.2f}, G-Mean: {:.2f}'.format(noise_dst, r_n, r_c, n_trees, prc, recall, fscore, auc, g_mean))
    

