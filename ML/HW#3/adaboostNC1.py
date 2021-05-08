import pandas as pd
import numpy as np
import scipy.stats as ss
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import RandomOverSampler


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
    y[y==-1]=0
    
    return x, y

def abs_ambiguty(trees, alpha, x, y):
    prds = np.zeros((x.shape[0], len(trees)))
    
    votes = np.zeros((x.shape[0], 2))
    for i, tree in enumerate(trees):
        prds[:, i] = tree.predict(x)
        for j in range(x.shape[0]):
            votes[j, int(prds[j, i])] += alpha[i]
        
    H = votes.argmax(axis=1)
    H_y = np.float64(H == y).reshape((-1, 1))
    
    h_y = np.float64(prds == np.reshape(y, (-1, 1)))
    
    amb = (H_y - h_y).mean(axis=1)
    return np.abs(amb)

def train_adaboost(x, y, n_trees, lamb):
    n_samples = x.shape[0]
    d = np.ones((n_samples,)) / n_samples
    p = np.ones((n_samples,))
    
    tree_list = []
    alpha = np.zeros((n_trees))
    inds = [i for i in range(n_samples)]
    for t in range(n_trees):
        if t > 0:
            amb = abs_ambiguty(tree_list[:t], alpha[:t], x, y)
            p = 1 - amb
        
        sel_inds = np.random.choice(inds, size=n_samples, replace=True, p=d)
        tree = DecisionTreeClassifier()
        tree.fit(x[sel_inds], y[sel_inds])
        l = tree.predict(x)
        match = (l == y).astype('float64')
        
        alpha[t] = 0.5 * np.log((d * p ** lamb * match).sum() / ((d * p ** lamb * np.logical_not(match)).sum()))
        tree_list.append(tree)
        
        d = p ** lamb * d * np.exp(-alpha[t] * match)
        d /= d.sum()
        
    return tree_list, alpha

def predict_adaboost(trees, alpha, x):
    votes = np.zeros((x.shape[0], 2))
    
    for i, tree in enumerate(trees):
        prd = tree.predict(x)
        for j in range(x.shape[0]):
            votes[j, int(prd[j])] += alpha[i]
            
    label = votes.argmax(axis=1)

    v_min = votes.min(axis=1).reshape((-1, 1))
    v_max = votes.max(axis=1).reshape((-1, 1))
    
    votes = (votes - v_min) / (v_max - v_min)
    p = votes / votes.sum(axis=1).reshape((-1, 1))
    return label, p[:, 1]

def report_measures_for_adaboost(x, y, n_trees, trn_ratio, lamb, noise_dst='none', r_n=0, r_c=0):
    trn_x, tst_x, trn_y, tst_y = train_test_split(x, y, train_size = trn_ratio, stratify = y)
    
    if noise_dst == 'train':
        trn_x = add_noise(trn_x, r_n, r_c)
    elif noise_dst == 'test':
        tst_x = add_noise(tst_x, r_n, r_c)

    sampler = RandomOverSampler()
    s_x, s_y = sampler.fit_resample(trn_x, trn_y)
    
    tree_list, alpha = train_adaboost(s_x, s_y, n_trees, lamb)
        
    prd, p = predict_adaboost(tree_list, alpha, tst_x)
    prc, recall, fscore, _ = precision_recall_fscore_support(tst_y, prd, labels=[1])
    auc = roc_auc_score(tst_y, p)
    g_mean = geometric_mean_score(tst_y, prd)
    return prc[0], recall[0], fscore[0], auc, g_mean

file_name = 'Covid-19.csv'
trn_ratio = 0.7
x, y = load_data(file_name)
n_itr = 10

for lamb in np.arange(0.1, 4, 0.5):
    print('lambda: {}'.format(lamb))
    for n_trees in [11, 21, 31, 41, 51]:
        prc, recall, fscore, auc, g_mean = (0, 0, 0, 0, 0)
        for i in range(n_itr):
            p, r, f, a, g = report_measures_for_adaboost(x, y, n_trees, trn_ratio, lamb)
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

        print('#Trees: {}, Precision: {:.2f}, Recall: {:.2f}, F-Score: {:.2f}, AUC: {:.2f}, G-Mean: {:.2f}'.format(n_trees, prc, recall, fscore, auc, g_mean))


for noise_dst in ['train', 'test']:
    for r_n in np.arange(0.1, 0.51, 0.1):
        for r_c in np.arange(0.1, 0.51, 0.1):
            for lamb in np.arange(0.1, 4, 0.5):
                print('lambda: {}'.format(lamb))
                for n_trees in [11, 21, 31, 41, 51]:
                    prc, recall, fscore, auc, g_mean = (0, 0, 0, 0, 0)
                    for i in range(n_itr):
                        p, r, f, a, g = report_measures_for_adaboost(x, y, n_trees, trn_ratio, lamb, noise_dst, r_n, r_c)
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

                    print('Noise Added to: {}, Rn: {}, Rc: {}, #Trees: {}, Precision: {:.2f}, Recall: {:.2f}, F-Score: {:.2f}, AUC: {:.2f}, G-Mean: {:.2f}'.format(noise_dst, r_n, r_c, n_trees, prc, recall, fscore, auc, g_mean))