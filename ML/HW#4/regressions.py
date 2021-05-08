import numpy as np
from numpy import dot
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from functools import partial

def get_A(n_samples=100):
    x = np.random.multivariate_normal(np.zeros(4,), np.eye(4), n_samples)
    eps = np.random.normal(0, 1, n_samples)
    y_n = x[:, 0] / (0.5 + (x[:, 1] + 1.5) ** 2) + (1 + x[:, 1]) ** 2 + 0.5 * eps
    y = x[:, 0] / (0.5 + (x[:, 1] + 1.5) ** 2) + (1 + x[:, 1]) ** 2
    return x, y, y_n

def get_B(n_samples=100):
    x = np.random.multivariate_normal(np.zeros(10,), np.eye(10), n_samples)
    eps = np.random.normal(0, 1, n_samples)
    y_n = x[:, 0] ** 2 / 2 * eps
    y = x[:, 0] ** 2 / 2
    return x, y, y_n

def get_C(n_samples=100):
    x = np.random.multivariate_normal(np.zeros(4,), np.eye(4), n_samples)
    eps = np.random.normal(0, 1, n_samples)
    y_n = x[:, 0] ** 2 + x[:, 1] + 0.5 * eps
    y = x[:, 0] ** 2 + x[:, 1]
    return x, y, y_n

def get_D(n_samples=100):
    x = np.random.multivariate_normal(np.zeros(10,), np.eye(10), n_samples)
    eps = np.random.normal(0, 1, n_samples)
    y_n = np.cos(3 * x[:, 0] / 2) + x[:, 1] ** 3 / 2 + 0.5 * eps
    y = np.cos(3 * x[:, 0] / 2) + x[:, 1] ** 3 / 2
    return x, y, y_n

def linear_regression(x, y, n_trn=70):
    x_trn = x[:n_trn]
    y_trn = y[:n_trn]

    x_tst = x[n_trn:]
    y_tst = y[n_trn:]

    w = dot(dot(inv(dot(x_trn.T, x_trn)), x_trn.T), y_trn)
    y_hat = dot(w.T, x_tst.T)

    return np.sqrt(mean_squared_error(y_tst, y_hat))

def ridge_regression(x, y, lmb, n_trn=70):
    x_trn = x[:n_trn]
    y_trn = y[:n_trn]

    x_tst = x[n_trn:]
    y_tst = y[n_trn:]

    w = dot(dot(inv(dot(x_trn.T, x_trn) + lmb * np.eye(x_trn.shape[1])), x_trn.T), y_trn)
    y_hat = dot(w.T, x_tst.T)
    
    return np.sqrt(mean_squared_error(y_tst, y_hat))

def kernerlized_ridge_regression(x, y, lmb, kernel, n_trn=70):
    x_trn = x[:n_trn]
    y_trn = y[:n_trn]

    x_tst = x[n_trn:]
    y_tst = y[n_trn:]

    k = np.zeros((x_trn.shape[0], x_trn.shape[0]))

    for i in range(k.shape[0]):
        for j in range(k.shape[0]):
            k[i, j] = kernel(x_trn[i, :], x_trn[j, :])

    alpha = inv(k + lmb * np.eye(k.shape[0])) @ y_trn
    y_hat = np.zeros(y_tst.shape)

    for i in range(y_hat.shape[0]):
        for j in range(alpha.shape[0]):
            y_hat[i] += alpha[j] * kernel(x_tst[i, :], x_trn[j, :])
    
    return np.sqrt(mean_squared_error(y_tst, y_hat))

def gaussian_kernel(x_i, x_j):
    return np.exp(-np.sum((x_i - x_j) ** 2) / 2)

def polynomial_kernel(x_i, x_j, d):
    return (x_i @ x_j.T + 1) ** d

def coordinate_descent_for_lasso(x, y, lmb, max_itr=200, n_trn=70):
    x_trn = x[:n_trn]
    y_trn = y[:n_trn]

    x_tst = x[n_trn:]
    y_tst = y[n_trn:]

    w = dot(dot(inv(dot(x_trn.T, x_trn) + lmb * np.eye(x_trn.shape[1])), x_trn.T), y_trn)

    for _ in range(max_itr):
        for j in range(x_trn.shape[1]):
            a_j = 2 * np.sum(x_trn[:, j] ** 2)
            c_j = 0
            for i in range(x_trn.shape[1]):
                c_j += x_trn[i, j] * (y_trn[i] - dot(w.T, x_trn[i, :].T) + w[j] * x_trn[i, j])
            a = c_j / a_j
            delta = lmb / a_j
            w[j] = np.sign(a) * np.maximum(np.abs(a) - delta, 0)

    y_hat = dot(w.T, x_tst.T)
    
    return np.sqrt(mean_squared_error(y_tst, y_hat))

data_set = {'A': get_A, 'B': get_B, 'C': get_C, 'D': get_D}
methods = {'linear_regression': linear_regression}

for lmb in [0.5, 1, 10, 100, 1000]:
    method_name = 'ridge regression' + ', lambda = ' + str(lmb)
    fcn = partial(ridge_regression, lmb=lmb)
    methods[method_name] = fcn

    method_name = 'coordinate descent for lasso' + ', lambda = ' + str(lmb)
    fcn = partial(coordinate_descent_for_lasso, lmb=lmb)
    methods[method_name] = fcn

    method_name = 'kernerlized ridge regression gaussian kernel' + ', lambda = ' + str(lmb)
    fcn = partial(kernerlized_ridge_regression, lmb=lmb, kernel=gaussian_kernel)
    methods[method_name] = fcn

    for d in [2, 5, 10]:
        method_name = 'kernerlized ridge regression polynomial kernel, d = ' + str(d) + ', lambda = ' + str(lmb)
        fcn = partial(kernerlized_ridge_regression, lmb=lmb, kernel=partial(polynomial_kernel, d=d))
        methods[method_name] = fcn


rmse = np.zeros((50,))
rmse_n = np.zeros((50,))
f = open('output.txt', 'w')
for ds in data_set:
    f.write(ds + '\n')
    for m in methods:
        for i in range(50):
            x, y, y_n = data_set[ds]()
            rmse[i] = methods[m](x, y)
            rmse_n[i] = methods[m](x, y_n)
        output_1 = '{}, {}, RMSE = {}\n'.format(m, 'without noise', rmse.mean())
        output_2 = '{}, {}, RMSE = {}\n'.format(m, 'with noise', rmse_n.mean())
        f.write(output_1)
        f.write(output_2)
        f.flush()

f.close()