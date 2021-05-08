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

def get_a_balanced_set(x, y):
    mjr_x = x[y == -1]
    mjr_y = y[y == -1]
    
    mnr_x = x[y == 1]
    mnr_y = y[y == 1]
    
    n_mnr = mnr_x.shape[0]
    n_mjr = mjr_x.shape[0]
    
    ind = [i for i in range(mjr_x.shape[0])]
    r = np.random.choice(ind, size=mnr_x.shape[0], replace=False)

    x_bs = np.concatenate((mjr_x[r], mnr_x), axis = 0)
    y_bs = np.concatenate((mjr_y[r], mnr_y), axis = 0)
    
    return x_bs, y_bs

def train_adaboost(x, y, n_trees, max_depth):
    n_samples = x.shape[0]
    d = np.ones((n_samples,)) / n_samples
    
    tree_list = []
    alpha = np.zeros((n_trees))
    inds = [i for i in range(n_samples)]
    for t in range(n_trees):
        sel_inds = np.random.choice(inds, size=n_samples, replace=True, p=d)
        tree = DecisionTreeClassifier(max_depth=max_depth)
        tree.fit(x[sel_inds], y[sel_inds])
        l = tree.predict(x)
        match = (l != y)
        err = (d * match).sum()
    
        if err > 0.5:
            return False
        
        alpha[t] = 0.5 * np.log((1 - err) / err)
        tree_list.append(tree)
        
        d = d * np.exp(-alpha[t] * l * y)
        d /= d.sum()
        
    return tree_list, alpha

def predict_adaboost(tree_list, alpha, x):
    s = alpha[0] * tree_list[0].predict(x)
    for i in range(1, len(tree_list)):
        s += alpha[i] * tree_list[i].predict(x)
        
    label = np.sign(s)
    s = s.astype(np.float64)
    p = 1. / (1 + np.exp(-2 * s))
    return label, p

def report_measures_for_adaboost(x, y, n_trees, n_ensembles, trn_ratio, noise_dst, r_n, r_c):
    trn_x, tst_x, trn_y, tst_y = train_test_split(x, y, train_size = trn_ratio, stratify = y)

    if noise_dst == 'train':
        trn_x = add_noise(trn_x, r_n, r_c)
    elif noise_dst == 'test':
        tst_x = add_noise(tst_x, r_n, r_c)

    all_trees = []
    all_alpha = []
    
    for _ in range(n_ensembles):
        bs_x, bs_y = get_a_balanced_set(trn_x, trn_y)
        max_depth = 1
        while True:
            out = train_adaboost(bs_x, bs_y, n_trees, max_depth)
            if out:
                ada_trees, alpha = out
                break
            max_depth += 1
            if max_depth > 10:
                bs_x, bs_y = get_a_balanced_set(trn_x, trn_y)
                max_depth = 1

        all_trees.extend(ada_trees)
        all_alpha.extend(alpha)
        
    prd, p = predict_adaboost(all_trees, all_alpha, tst_x)
    prc, recall, fscore, _ = precision_recall_fscore_support(tst_y, prd, labels=[1])
    auc = roc_auc_score(tst_y, p)
    g_mean = geometric_mean_score(tst_y, prd)
    return prc[0], recall[0], fscore[0], auc, g_mean

file_name = 'Covid-19.csv'
trn_ratio = 0.7
x, y = load_data(file_name)
n_itr = 10

for noise_dst in ['train', 'test']:
    for r_n in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for r_c in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for ensemble_size in [10, 15]:
                for n_trees in [11, 31, 51, 101]:
                    prc, recall, fscore, auc, g_mean = (0, 0, 0, 0, 0)
                    for i in range(n_itr):
                        p, r, f, a, g = report_measures_for_adaboost(x, y, n_trees, ensemble_size, trn_ratio, noise_dst, r_n, r_c)
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

                    print('ensemble size: {}, Noise Added to: {}, Rn: {}, Rc: {}, #Trees: {}, Precision: {:.2f}, Recall: {:.2f}, F-Score: {:.2f}, AUC: {:.2f}, G-Mean: {:.2f}'.format(ensemble_size, noise_dst, r_n, r_c, n_trees, prc, recall, fscore, auc, g_mean))
        

