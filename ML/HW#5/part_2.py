import pandas as pd
import numpy as np


def forward_alg_1(a, b, p, o):
    N = a.shape[0]
    T = o.shape[0]

    forward = np.zeros((N, T))
    forward[:, 0] = p * b[:, o[0]]

    for t in range(1, T):
        for s in range(N):
            for sp in range(N):
                forward[s, t] += forward[sp, t - 1] * a[sp, s] * b[s, o[t]]

    return forward


def forward_alg_2(a, b, p, o):
    N = a.shape[0]
    T = o.shape[0]

    forward = np.zeros((N, T))
    forward[:, 0] = np.log(p) + np.log(b[:, o[0]])

    for t in range(1, T):
        for s in range(N):
            tmp = forward[0, t - 1] + np.log(a[0, s]) + np.log(b[s, o[t]])
            for sp in range(1, N):
                tmp1 = forward[sp, t - 1] + np.log(a[sp, s]) + np.log(b[s, o[t]])
                if tmp1 <= tmp:
                    tmp += np.log(1 + np.exp(tmp1 - tmp))
                else:
                    tmp = tmp1 + np.log(1 + np.exp(tmp - tmp1))
            forward[s, t] = tmp
    return forward


def backward_alg_1(a, b, o):
    N = a.shape[0]
    T = o.shape[0]

    backward = np.zeros((N, T))
    backward[:, T - 1] = 1

    for t in range(T - 2, -1, -1):
        for s in range(N):
            for sp in range(N):
                backward[s, t] += backward[sp, t + 1] * a[s, sp] * b[sp, o[t + 1]]

    return backward


def backward_alg_2(a, b, o):
    N = a.shape[0]
    T = o.shape[0]

    backward = np.zeros((N, T))
    for t in range(T - 2, -1, -1):
        for s in range(N):
            tmp = backward[0, t + 1] + np.log(a[s, 0]) + np.log(b[0, o[t + 1]])
            for sp in range(1, N):
                tmp1 = backward[sp, t + 1] + np.log(a[s, sp]) + np.log(b[sp, o[t + 1]])
                if tmp1 <= tmp:
                    tmp += np.log(1 + np.exp(tmp1 - tmp))
                else:
                    tmp = tmp1 + np.log(1 + np.exp(tmp - tmp1))
            backward[s, t] = tmp
    return backward


def forward_backward(a, b, p, trn_seqs, v, q, forward_fcn, backward_fcn, max_iter=10):
    for iter in range(max_iter):
        print(iter)
        for pr, o in enumerate(trn_seqs):
            # print(pr)
            alpha = np.exp(forward_fcn(a, b, p, o))
            beta = np.exp(backward_fcn(a, b, o))

            N, T = alpha.shape

            s = alpha.sum(axis=0)
            alpha = alpha / s[np.newaxis, :]

            s = beta.sum(axis=0)
            s[s == 0] = 1
            beta = beta / s[np.newaxis, :]

            denum = alpha[:, -1].sum()
            gamma = alpha * beta / denum

            xee = np.zeros((N, N, T))
            for i in range(N):
                for j in range(N):
                    for t in range(T - 1):
                        xee[i, j, t] = (
                            alpha[i, t]
                            * a[i, j]
                            * b[j, o[t + 1]]
                            * beta[j, t + 1]
                            / denum
                        )
            for i in range(N):
                for j in range(N):
                    a[i, j] = xee[i, j, :].sum() / xee[i, :-1, :].sum()
            for j in range(b.shape[0]):
                for vk in range(b.shape[1]):
                    b[j, vk] = gamma[j, o == vk].sum() / gamma[j, :].sum()

    return a, b


def load_data(trn_file_name, tst_file_name):
    df_trn = pd.read_csv(trn_file_name, header=None, sep="/")
    df_tst = pd.read_csv(tst_file_name, header=None, sep="/")

    df = pd.concat([df_trn, df_tst])
    words = df.iloc[:, 0].factorize()
    tags = df.iloc[:, 1].factorize()

    n_trn = df_trn.shape[0]

    trn_words = words[0][:n_trn]
    tst_words = words[0][n_trn:]

    trn_tags = tags[0][:n_trn]
    tst_tags = tags[0][n_trn:]

    words_ind = words[1]
    tags_ind = tags[1]

    return trn_words, trn_tags, tst_words, tst_tags, words_ind, tags_ind


def load_raw(file_name, words_ind):
    df = pd.read_csv(file_name, header=None, sep=" ")
    words = df.values
    ind = np.zeros(words.shape)
    for word in words_ind:
        ind = np.logical_or(ind, words == word)

    words = words[ind]
    coded_words = np.zeros((words.shape[0],), dtype=int)
    for i, word in enumerate(words_ind):
        coded_words[words == word] = i

    ind = np.where(words == "###")[0]
    ind = np.insert(ind, 0, -1) + 1
    seqs = []
    for i in range(ind.shape[0] - 1):
        seqs.append(coded_words[ind[i] : ind[i + 1]])

    return seqs


def add_one_smoothing(words, tags, sharp, n_tags, n_words):
    p = np.zeros((n_tags,))
    ind = np.where(tags == sharp)[0]
    n_sentences = ind.shape[0]
    ind = np.insert(ind, 0, -1)[:-1] + 1
    for i in range(n_tags):
        p[i] = (np.count_nonzero(tags[ind] == i) + 1) / (n_sentences + n_tags)

    a = np.zeros((n_tags, n_tags))
    b = np.zeros((n_tags, n_words))
    for i in range(tags.shape[0] - 1):
        if tags[i] != sharp:
            a[tags[i], tags[i + 1]] += 1
            b[tags[i], words[i]] += 1

    c_q = a.sum(axis=1)
    a = (a + 1) / (c_q[:, np.newaxis] + n_tags)

    b = (b + 1) / (c_q[:, np.newaxis] + n_words)
    return a, b, p


def viterbi(a, b, p, o):
    N = a.shape[0]
    T = o.shape[0]

    vit = np.zeros((N, T))
    back_pointer = np.zeros((N, T))

    # initialization step
    vit[:, 0] = np.log(p) + np.log(b[:, o[0]])
    back_pointer[:, 0] = 0

    for t in range(1, T):
        for s in range(N):
            a_max = np.argmax(vit[:, t - 1] + np.log(a[:, s]))
            back_pointer[s, t] = a_max
            vit[s, t] = np.log(b[s, o[t]]) + vit[a_max, t - 1] + np.log(a[a_max, s])

    best_path_pointer = np.argmax(vit[:, T - 1])

    best_path = np.zeros((1,), dtype=int)
    best_path[0] = best_path_pointer
    for t in range(T - 1, 0, -1):
        best_path = np.insert(best_path, 0, back_pointer[best_path[0], t])

    return best_path


def apply_viterbi(a, b, p, tst_words, tags_ind, words_ind, word_sharp):
    ind = np.where(tst_words == word_sharp)[0]
    n_sentences = ind.shape[0]
    ind = np.insert(ind, 0, -1) + 1

    tags = []
    for i in range(n_sentences):
        snt = tst_words[ind[i] : ind[i + 1]]
        path = viterbi(a, b, p, snt)
        tags = tags + [tags_ind[j] for j in path]
    return tags


data_sets = {
    "IC": ("ictrain.txt", "ictest.txt", "icraw.txt"),
    "EN": ("entrain.txt", "entest.txt", "enraw.txt"),
    # "CZ": ("cztrain.txt", "cztest.txt"),
}

for ds in data_sets:
    trn_file_name = "Dataset/" + data_sets[ds][0]
    tst_file_name = "Dataset/" + data_sets[ds][1]
    raw_file_name = "Dataset/" + data_sets[ds][2]
    trn_words, trn_tags, tst_words, tst_tags, words_ind, tags_ind = load_data(
        trn_file_name, tst_file_name
    )

    seqs = load_raw(raw_file_name, words_ind)[:100]

    tag_sharp = np.where(tags_ind == "###")[0][0]
    word_sharp = np.where(words_ind == "###")[0][0]
    n_tags = tags_ind.shape[0]
    n_words = words_ind.shape[0]
    a, b, p = add_one_smoothing(trn_words, trn_tags, tag_sharp, n_tags, n_words)
    a, b = forward_backward(
        a, b, p, seqs, tags_ind.shape[0], a.shape[0], forward_alg_1, backward_alg_1
    )
    prd_tags = apply_viterbi(a, b, p, tst_words, tags_ind, words_ind, word_sharp)
    tst_actual_tags = [tags_ind[i] for i in tst_tags]
    tst_actual_words = [words_ind[i] for i in tst_words]

    s = sum([prd_tags[i] == tst_actual_tags[i] for i in range(len(prd_tags))])
    acc = s / len(prd_tags)
    print("{} accuracy is {}".format(ds, acc))

    f = open("{}_add_one_smoothing.txt".format(ds), "w")
    for word, tag, prd in zip(tst_actual_words, tst_actual_tags, prd_tags):
        f.writelines(word + "/" + tag + "/" + prd + "\n")
    f.close()
