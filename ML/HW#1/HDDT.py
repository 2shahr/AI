#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


# In[2]:


class TreeNode:
    def __init__(self, feature_index, is_leaf = False, prob = None):
        self.feature_index = feature_index
        self.is_leaf = is_leaf
        self.childs = []
        self.child_values = []
        self.prob = prob
        
    def add_child(self, child, value):
        self.childs.append(child)
        self.child_values.append(value)


# In[3]:


def Binary_Hellinger(z, pos_class):
    Tf = np.unique(z[:, 0])
    helinger = -1
    z_plus = sum(z[:, 1] == pos_class)
    z_minus = z.shape[0] - z_plus
    
    for t in Tf:
        z_f_t_plus = sum((z[:, 0] == t) * (z[:, 1] == pos_class))
        z_f_t_minus = sum((z[:, 0] == t) * (z[:, 1] != pos_class))
        
        z_f_p_plus = sum((z[:, 0] != t) * (z[:, 1] == pos_class))
        z_f_p_minus = sum((z[:, 0] != t) * (z[:, 1] != pos_class))
        
        hd_value = ((z_f_t_plus / z_plus) ** 0.5                     - (z_f_t_minus / z_minus) ** 0.5) ** 2 +                     ((z_f_p_plus / z_plus) ** 0.5                     - (z_f_p_minus / z_minus) ** 0.5) ** 2
        
        if hd_value > helinger:
            helinger = hd_value
    
    return helinger ** 0.5


# In[4]:


def HDDT(z, c, unused_features, pos_class):
    if z.shape[0] <= c or len(np.unique(z[:, -1])) == 1:
        prob = np.mean(z[:, -1] == pos_class)
        return TreeNode([], is_leaf = True, prob = prob)
    
    best_helinger = -1
    for i, f_ind in enumerate(unused_features):
        z_f = z[:, [f_ind, -1]]
        helinger = Binary_Hellinger(z_f, pos_class)
        if helinger > best_helinger:
            best_helinger = helinger
            best_f_ind = f_ind
            
    node = TreeNode(best_f_ind)
    Tf = np.unique(z[:, best_f_ind])
    unused_features.discard(best_f_ind)
    for v in Tf:
        z_v = z[z[:, best_f_ind] == v, :]
        node.add_child(HDDT(z_v, c, unused_features, pos_class), v)
    
    return node


# In[5]:


def classify(tree, data):
    prd = np.zeros(data.shape[0])
    for i, sample in enumerate(data):
        node = tree
        while node.is_leaf == False:
            child_ind = np.nonzero(sample[node.feature_index] == node.child_values)[0][0]
            node = node.childs[child_ind]
        prd[i] = node.prob
        
    return prd


# In[6]:


df_dis = pd.read_csv('Dis_Covid-19.csv')
df_con = pd.read_csv('Con_Covid-19.csv')


# In[7]:


data_con = df_con.values
np.random.shuffle(data_con)
data_dis = df_dis.values
np.random.shuffle(data_dis)

data_dis_bin = data_dis.copy()
data_dis_bin[:, -1] = data_dis[:, -1] == 2

train_ratio = 0.7
c = 100
n_trn = int(train_ratio * data_dis.shape[0])

trn_dis = data_dis[:n_trn]
tst_dis = data_dis[n_trn:]

trn_con = data_dis[:n_trn]
tst_con = data_dis[n_trn:]

trn_dis_bin = data_dis_bin[:n_trn]
tst_dis_bin = data_dis_bin[n_trn:]


# In[8]:


tree = HDDT(trn_dis_bin, c, {i for i in range(trn_dis_bin.shape[1] - 1)}, 1)


# In[9]:


prd = classify(tree, tst_dis_bin)
prd = prd > 0

fpr, tpr, thresholds = metrics.roc_curve(tst_dis_bin[:, -1], prd, pos_label=1)
auc = metrics.auc(fpr, tpr)
precision, recall, f1, _ = metrics.precision_recall_fscore_support(tst_dis_bin[:, -1], prd, average=None, labels = [1])

print('Binary HDDT')
print("AUC: {}".format(auc))
print("Precison: {}".format(precision[0]))
print("Recall: {}".format(recall[0]))
print("f1: {}".format(precision[0]))


# In[10]:


cls = GaussianNB()
prd = cls.fit(trn_dis_bin[:, :-1], trn_dis_bin[:, -1]).predict(tst_dis_bin[:, :-1])
fpr, tpr, thresholds = metrics.roc_curve(tst_dis_bin[:, -1], prd, pos_label=1)
auc = metrics.auc(fpr, tpr)
precision, recall, f1, _ = metrics.precision_recall_fscore_support(tst_dis_bin[:, -1], prd, average=None, labels = [1])

print('Binary HDDT')
print("AUC: {}".format(auc))
print("Precison: {}".format(precision[0]))
print("Recall: {}".format(recall[0]))
print("f1: {}".format(precision[0]))


# In[11]:


cls = KNeighborsClassifier()
prd = cls.fit(trn_dis_bin[:, :-1], trn_dis_bin[:, -1]).predict(tst_dis_bin[:, :-1])
fpr, tpr, thresholds = metrics.roc_curve(tst_dis_bin[:, -1], prd, pos_label=1)
auc = metrics.auc(fpr, tpr)
precision, recall, f1, _ = metrics.precision_recall_fscore_support(tst_dis_bin[:, -1], prd, average=None, labels = [1])

print('Binary HDDT')
print("AUC: {}".format(auc))
print("Precison: {}".format(precision[0]))
print("Recall: {}".format(recall[0]))
print("f1: {}".format(precision[0]))


# In[12]:


cls = LinearSVC()
prd = cls.fit(trn_dis_bin[:, :-1], trn_dis_bin[:, -1]).predict(tst_dis_bin[:, :-1])
fpr, tpr, thresholds = metrics.roc_curve(tst_dis_bin[:, -1], prd, pos_label=1)
auc = metrics.auc(fpr, tpr)
precision, recall, f1, _ = metrics.precision_recall_fscore_support(tst_dis_bin[:, -1], prd, average=None, labels = [1])

print('Binary HDDT')
print("AUC: {}".format(auc))
print("Precison: {}".format(precision[0]))
print("Recall: {}".format(recall[0]))
print("f1: {}".format(precision[0]))


# In[13]:


cls = SVC(gamma='auto')
prd = cls.fit(trn_dis_bin[:, :-1], trn_dis_bin[:, -1]).predict(tst_dis_bin[:, :-1])
fpr, tpr, thresholds = metrics.roc_curve(tst_dis_bin[:, -1], prd, pos_label=1)
auc = metrics.auc(fpr, tpr)
precision, recall, f1, _ = metrics.precision_recall_fscore_support(tst_dis_bin[:, -1], prd, average=None, labels = [1])

print('Binary HDDT')
print("AUC: {}".format(auc))
print("Precison: {}".format(precision[0]))
print("Recall: {}".format(recall[0]))
print("f1: {}".format(precision[0]))


# In[ ]:




