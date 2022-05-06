import pandas as pd
import numpy as np
import minisom
import matplotlib.pyplot as plt

def data_winner_distance(som, data):
    d = 0
    w = som.get_weights()
    for i in range(data.shape[0]):
        winner = som.winner(data[i])
        d += np.linalg.norm(data[i] - w[winner[0], winner[1]])
    return d


df = pd.read_excel('Data.xlsx')

data = df.iloc[:, :-1].values
np.random.shuffle(data)
n_trn = int(np.round(0.8 * data.shape[0]))
trn = data[:n_trn]
tst = data[n_trn:]


d = float('inf')
num_features = 2
epoch = 100000
for sigma in [i / 10.0 for i in range(1, 51, 5)]:
    for lr in [i / 10.0 for i in range(1, 10)]:
        som = minisom.MiniSom(10, 10, num_features, sigma=sigma, learning_rate=lr)
        som.train(trn, epoch)
        new_d = data_winner_distance(som, trn)
        if new_d < d:
            d = new_d
            best_sigma = sigma
            best_lr = lr

som = minisom.MiniSom(10, 10, num_features, sigma=best_sigma, learning_rate=best_lr)
w = som.get_weights()
plt.figure()
plt.plot(trn[:, 0], trn[:, 1], 'r*')
for i in range(10):
    plt.plot(w[i, :, 0], w[i, :, 1], '-ks', markersize=10, markerfacecolor='yellow', markeredgecolor='green')
    
for i in range(10):
    plt.plot(w[:, i, 0], w[:, i, 1], '-ks', markersize=10, markerfacecolor='yellow', markeredgecolor='green')
    
plt.show(block=False)


som.train(trn, epoch)
w = som.get_weights()
d = data_winner_distance(som, tst)
print('The Overall Distance is {}'.format(d))

plt.figure()
plt.plot(trn[:, 0], trn[:, 1], 'r*')
for i in range(10):
    plt.plot(w[i, :, 0], w[i, :, 1], '-ks', markersize=10, markerfacecolor='yellow', markeredgecolor='green')
    
for i in range(10):
    plt.plot(w[:, i, 0], w[:, i, 1], '-ks', markersize=10, markerfacecolor='yellow', markeredgecolor='green')

plt.show(block=True)
