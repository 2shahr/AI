import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from imblearn.metrics import geometric_mean_score
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def load_data(file_name):
    df = pd.read_csv(file_name)
    df = df.fillna(-1)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    return x, y

def report_measures_for_clf(x, y, clf, sampler):
    trn_x, tst_x, trn_y, tst_y = train_test_split(x, y, train_size = trn_ratio, stratify = y)
    s_x, s_y = sampler.fit_resample(trn_x, trn_y)
    clf.fit(s_x, s_y)
    p = clf.predict_proba(tst_x)
    prd = np.argmax(p, axis=1)
    prd[prd == 0] = -1
    prc, recall, fscore, _ = precision_recall_fscore_support(tst_y, prd, labels=[1])
    auc = roc_auc_score(tst_y, p[:, 1])
    g_mean = geometric_mean_score(tst_y, prd)
    return prc[0], recall[0], fscore[0], auc, g_mean


clfs = {'NB': GaussianNB(), 'Linear-SVM': SVC(kernel='linear', probability=True), 
        'RBF-SVM': SVC(probability=True), '1NN': KNeighborsClassifier(n_neighbors=1)}
samplers = {'SMOTE': SMOTE(), 'UnderSampler': RandomUnderSampler(), 'OverSampler': RandomOverSampler()}


file_name = 'Covid-19.csv'
trn_ratio = 0.7
x, y = load_data(file_name)
n_itr = 10

for clf_name in clfs:
    for sampler_name  in samplers:
        prc, recall, fscore, auc, g_mean = (0, 0, 0, 0, 0)
        for i in range(n_itr):
            p, r, f, a, g = report_measures_for_clf(x, y, clfs[clf_name], samplers[sampler_name])
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

        print('{}, {}, Precision: {:.2f}, Recall: {:.2f}, F-Score: {:.2f}, AUC: {:.2f}, G-Mean: {:.2f}'.format(clf_name, sampler_name, prc, recall, fscore, auc, g_mean))

