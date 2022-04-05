import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

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
        
        hd_value = ((z_f_t_plus / z_plus) ** 0.5 - (z_f_t_minus / z_minus) ** 0.5) ** 2 + ((z_f_p_plus / z_plus) ** 0.5 - (z_f_p_minus / z_minus) ** 0.5) ** 2
        
        if hd_value > helinger:
            helinger = hd_value
    if helinger < 0:
        print(helinger)
        pdb.set_trace()
    return helinger ** 0.5


def HDDT(z, c, unused_features, pos_class, current_height, max_height=1000000):
    if z.shape[0] <= c or len(np.unique(z[:, -1])) == 1 or current_height >= max_height:
        prob = np.mean(z[:, -1] == pos_class)
        return TreeNode([], is_leaf = True, prob = prob)
    
    best_helinger = -1
    for i, f_ind in enumerate(unused_features):
        z_f = z[:, [f_ind, -1]]
        if np.max(np.unique(z_f[:, -1])) > 1:
            pdb.set_trace()
            
        helinger = Binary_Hellinger(z_f, pos_class)
        if helinger > best_helinger:
            best_helinger = helinger
            best_f_ind = f_ind
            
    node = TreeNode(best_f_ind)
    Tf = np.unique(z[:, best_f_ind])
    unused_features.discard(best_f_ind)
    for v in Tf:
        z_v = z[z[:, best_f_ind] == v, :]
        node.add_child(HDDT(z_v, c, unused_features, pos_class, current_height + 1, max_height), v)
    
    return node


def classify(tree, data):
    prd = np.zeros(data.shape[0])
    for i, sample in enumerate(data):
        node = tree
        while node.is_leaf == False:
            child_ind = np.nonzero(sample[node.feature_index] == node.child_values)[0][0]
            node = node.childs[child_ind]
        prd[i] = node.prob
        
    return prd


def load_dataset(trn_ratio=0.7, minority_class = 2, dis_file_name='Dis_Covid-19.csv', con_file_name='Con_Covid-19.csv'):
    
    df_dis = pd.read_csv(dis_file_name)
    df_con = pd.read_csv(con_file_name)
    
    data_con = df_con.values
    np.random.shuffle(data_con)
    
    data_dis = df_dis.values
    np.random.shuffle(data_dis)

    data_dis_bin = data_dis.copy()
    data_dis_bin[:, -1] = data_dis[:, -1] == minority_class
    
    data_con_bin = data_con.copy()
    data_con_bin[:, -1] = data_con[:, -1] == minority_class

    n_trn = int(trn_ratio * data_dis.shape[0])

    trn_dis = data_dis[:n_trn]
    tst_dis = data_dis[n_trn:]

    trn_con = data_dis[:n_trn]
    tst_con = data_dis[n_trn:]

    trn_dis_bin = data_dis_bin[:n_trn]
    tst_dis_bin = data_dis_bin[n_trn:]
    
    trn_con_bin = data_con_bin[:n_trn]
    tst_con_bin = data_con_bin[n_trn:]
    
    return trn_dis_bin, tst_dis_bin, trn_con_bin, tst_con_bin, trn_dis, tst_dis, trn_con, tst_con


def one_vs_all(trn, c, max_height=100000):
    unq = np.unique(trn[:, -1])
    trees = []
    for t in unq:
        new_trn = trn.copy()
        new_trn[:, -1] = trn[:, -1] == t
        tree = HDDT(new_trn, c, {i for i in range(trn.shape[1] - 1)}, 1, 0, max_height)
        trees.append(tree)
    return trees


def classify_one_vs_all(trees, tst):
    prds = np.zeros((tst.shape[0], len(trees)))
    
    for i, tree in enumerate(trees):
        prds[:, i] = classify(tree, tst)
    
    prds = prds / np.sum(prds, axis=1)[:, np.newaxis]
    return prds

def one_vs_one(trn, c, max_height=100000):
    max_label = max(trn[:, -1])
    trees = []
    for t1 in range(int(max_label) + 1):
        for t2 in range(t1 + 1, int(max_label) + 1):
            ind = np.logical_or(trn[:, -1] == t1, trn[:, -1] == t2)
            new_trn = trn[ind, :]
            new_trn[:, -1] = new_trn[:, -1] == t1
            tree = HDDT(new_trn, c, {i for i in range(trn.shape[1] - 1)}, 1, 0, max_height)
            tree.pos_label = t1
            tree.neg_label = t2
            trees.append(tree)
    return trees

def classify_one_vs_one(trees, tst):
    n_class = 3
    prds = np.zeros((tst.shape[0], n_class))
    for tree in trees:
        prd = classify(tree, tst)
        prds[:, tree.pos_label] += prd
        prds[:, tree.neg_label] += 1 - prd
        
    prds = prds / np.sum(prds, axis=1)[:, np.newaxis]
    return prds


def step1_HDDT(num_itr, max_height, file, c):
    sum_auc = 0
    sum_precision = 0
    sum_recall = 0
    sum_f1 = 0
    
    for _ in range(num_itr):
        trn, tst, *_ = load_dataset()
        
        tree = HDDT(trn, c, {i for i in range(trn.shape[1] - 1)}, 1, 0, max_height)
        
        prd = classify(tree, tst)

        fpr, tpr, _ = metrics.roc_curve(tst[:, -1], prd, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        precision, recall, f1, _ =  metrics.precision_recall_fscore_support(tst[:, -1], prd, average=None, labels = [1])
        sum_auc += auc
        sum_precision += precision[0]
        sum_recall += recall[0]
        sum_f1 += f1[0]

        print('one done!')
    
    file.write('HDDT\n')
    file.write("Average AUC: {}\n".format(sum_auc / num_itr))
    file.write("Average Precison: {}\n".format(sum_precision / num_itr))
    file.write("Average Recall: {}\n".format(sum_recall / num_itr))
    file.write("Average f1: {}\n".format(sum_f1 / num_itr))


def other_classifiers(num_itr, is_binary, file):
    
    clfs = [GaussianNB(), KNeighborsClassifier(n_neighbors=1), SVC(kernel='linear', gamma='auto', probability=True), SVC(kernel='rbf', gamma='auto', probability=True)]
    clf_names = ['NB', '1-NN', 'LinearSVM', 'rbf_SVM']
    
    for clf, clf_name in zip(clfs, clf_names):
        sum_auc = 0
        sum_precision = 0
        sum_recall = 0
        sum_f1 = 0
        for _ in range(num_itr):
            if is_binary:
                _, _, trn, tst, *_ = load_dataset()
                pos_label = 1
            else:
                _, _, _, _, _, _, trn, tst = load_dataset()
                pos_label = 2
            
            prd = clf.fit(trn[:, :-1], trn[:, -1]).predict_proba(tst[:, :-1])
            fpr, tpr, thresholds = metrics.roc_curve(tst[:, -1], prd[:, -1], pos_label=pos_label)
            auc = metrics.auc(fpr, tpr)
            precision, recall, f1, _ =  metrics.precision_recall_fscore_support(tst[:, -1], prd.argmax(axis=1), average=None, labels = [pos_label])
            sum_auc += auc
            sum_precision += precision[0]
            sum_recall += recall[0]
            sum_f1 += f1[0]
            
            print('one done!')
    
        file.write('\n{}\n'.format(clf_name))
        file.write("Average AUC: {}\n".format(sum_auc / num_itr))
        file.write("Average Precison: {}\n".format(sum_precision / num_itr))
        file.write("Average Recall: {}\n".format(sum_recall / num_itr))
        file.write("Average f1: {}\n".format(sum_f1 / num_itr))


def step2_HDDT(num_itr, max_height, file, c):
    train_funs = [one_vs_all, one_vs_one]
    test_funs = [classify_one_vs_all, classify_one_vs_one]
    fun_names = ['One VS All', 'One VS One']
    
    for trn_fun, tst_fun, fun_name in zip(train_funs, test_funs, fun_names):
        sum_auc = 0
        sum_precision = 0
        sum_recall = 0
        sum_f1 = 0

        for _ in range(num_itr):
            _, _, _, _, trn, tst, *_ = load_dataset()

            trees = trn_fun(trn, c)

            prd = tst_fun(trees, tst)

            fpr, tpr, _ = metrics.roc_curve(tst[:, -1], prd[:, -1], pos_label=2)
            auc = metrics.auc(fpr, tpr)
            precision, recall, f1, _ =  metrics.precision_recall_fscore_support(tst[:, -1], prd.argmax(axis=1), average=None, labels = [2])
            sum_auc += auc
            sum_precision += precision[0]
            sum_recall += recall[0]
            sum_f1 += f1[0]
            
            print('one done!')

        file.write('\nHDDT {}\n'.format(fun_name))
        file.write("Average AUC: {}\n".format(sum_auc / num_itr))
        file.write("Average Precison: {}\n".format(sum_precision / num_itr))
        file.write("Average Recall: {}\n".format(sum_recall / num_itr))
        file.write("Average f1: {}\n".format(sum_f1 / num_itr))


cutoff_size = 100
num_itr = 10
max_height = float('inf')
is_binary = True

file = open('Step 1.txt', 'w')
step1_HDDT(num_itr, max_height, file, cutoff_size)
other_classifiers(num_itr, is_binary, file)
file.close()

is_binary = False

file = open('Step 2.txt', 'w')
step2_HDDT(num_itr, max_height, file, cutoff_size)
other_classifiers(num_itr, is_binary, file)
file.close()


for height in range(2, 6):
    file = open('Step 1 - MAX HEIGHT = ' + str(height) + '.txt', 'w')
    step1_HDDT(num_itr, height, file, cutoff_size)
    file.close()
    
    file = open('Step 2 - MAX HEIGHT = ' + str(height) + '.txt', 'w')
    step2_HDDT(num_itr, height, file, cutoff_size)
    file.close()

