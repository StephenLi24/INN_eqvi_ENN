# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from Gaussian_Integration import *
import data_prepare
import argparse
def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--n', type=int, default = '400', help='dimension of gmm_data')
    parser.add_argument('--phi',type = str, default = 'relu', help='tanh or relu')
    parser.add_argument('--asquare', type=float, default=0.2, help='coefficients sa')
    args = parser.parse_args()
    return args

def main():
    args = parse_args_and_config()
    m = 1000
    K_maxiter = 100
    fp_maxiter = 25

    p = 784
    n = args.n
    cn = 2  # class number
    sa = 0.2
    sb = 1 - sa
    tau_maxiter = 20

    name = args.phi
    # name = 'relu'

    def phi_t(x):
        if name == 'tanh':
            return np.tanh(x)
        if name == 'relu':
            return np.maximum(0, x)


    def dphi(x):
        if name == 'tanh':
            return 1 - np.tanh(x) ** 2
        if name == 'relu':
            return np.where(x >= 0, 1, 0)

    res = data_prepare.gen_data(
        'MNIST',
        selected_target=[6, 8],
        T=n,
        p=p,
        cs=[0.5, 0.5]
    )
    X = res[0]
    # Calculate C0, the average of the covariance matrices of all classes
    C0 = np.cov(X) / cn

    # Calculate the covariance matrix and mean vector of each class
    C_l = []
    mu_l = []
    for i in range(cn):
        # Extract the data of the i-th class from X
        Xi = X[:, i * (n // cn): (i + 1) * (n // cn)]
        # Calculate the covariance matrix of the i-th class
        Ci = np.cov(Xi)
        # Calculate the mean vector of the i-th class
        mui = np.mean(Xi, axis=1, keepdims=True)
        # Add Ci and mui to the lists
        C_l.append(Ci)
        mu_l.append(mui)

        # Calculate C0_l of each class, which is the covariance matrix minus C0
    C0_l = []
    for i in range(cn):
        C0_l.append(C_l[i] - C0)

    # Calculate Z of each class, which is the data minus the mean vector
    Z_l = []
    for i in range(cn):
        # Extract the data of the i-th class from X
        Xi = X[:, i * (n // cn): (i + 1) * (n // cn)]
        # Calculate Z of the i-th class
        Zi = Xi - mu_l[i]
        # Add Zi to the list
        Z_l.append(Zi)

    t = np.zeros((cn, 1))
    # for i in range(cn):
    #     t[i, :] = np.trace(C0_l[i]) / np.sqrt(p)
    T = np.zeros((cn, cn))
    for i in range(cn):
        for j in range(cn):
            T[i, j] = np.trace(C0_l[i] * C0_l[j]) / p
    Psi_l = []
    for i in range(cn):
        VZN = np.sum(Z_l[i] ** 2, axis=0) / p
        CC = np.tile(np.trace(C_l[i]) / p, (1, n // cn))
        Psi_l.append(VZN - CC)
    Psi = np.concatenate(Psi_l, axis=1)

    J_l = []
    for i in range(cn):
        one_hot = np.eye(cn)[:, i]
        Jt = np.tile(one_hot, (n // cn, 1))
        J_l.append(Jt)
    J = np.concatenate(J_l, axis=0)
    V = np.concatenate([J / np.sqrt(p), Psi.T], axis=1)

    # tau0 = np.sqrt(np.trace(C0) / p)
    tau0 = np.sqrt(np.sum(X ** 2) / n)    # Calculate C0, the average of the covariance matrices of all classes
    C0 = np.cov(X) / cn

    # Calculate the covariance matrix and mean vector of each class
    C_l = []
    mu_l = []
    for i in range(cn):
        # Extract the data of the i-th class from X
        Xi = X[:, i * (n // cn): (i + 1) * (n // cn)]
        # Calculate the covariance matrix of the i-th class
        Ci = np.cov(Xi)
        # Calculate the mean vector of the i-th class
        mui = np.mean(Xi, axis=1, keepdims=True)
        # Add Ci and mui to the lists
        C_l.append(Ci)
        mu_l.append(mui)

        # Calculate C0_l of each class, which is the covariance matrix minus C0
    C0_l = []
    for i in range(cn):
        C0_l.append(C_l[i] - C0)

    # Calculate Z of each class, which is the data minus the mean vector
    Z_l = []
    for i in range(cn):
        # Extract the data of the i-th class from X
        Xi = X[:, i * (n // cn): (i + 1) * (n // cn)]
        # Calculate Z of the i-th class
        Zi = Xi - mu_l[i]
        # Add Zi to the list
        Z_l.append(Zi)

    t = np.zeros((cn, 1))
    # for i in range(cn):
    #     t[i, :] = np.trace(C0_l[i]) / np.sqrt(p)
    T = np.zeros((cn, cn))
    for i in range(cn):
        for j in range(cn):
            T[i, j] = np.trace(C0_l[i] * C0_l[j]) / p
    Psi_l = []
    for i in range(cn):
        VZN = np.sum(Z_l[i] ** 2, axis=0) / p
        CC = np.tile(np.trace(C_l[i]) / p, (1, n // cn))
        Psi_l.append(VZN - CC)
    Psi = np.concatenate(Psi_l, axis=1)

    J_l = []
    for i in range(cn):
        one_hot = np.eye(cn)[:, i]
        Jt = np.tile(one_hot, (n // cn, 1))
        J_l.append(Jt)
    J = np.concatenate(J_l, axis=0)
    V = np.concatenate([J / np.sqrt(p), Psi.T], axis=1)

    # tau0 = np.sqrt(np.trace(C0) / p)
    tau0 = np.sqrt(np.sum(X ** 2) / n)


    tau = 0
    for i in range(tau_maxiter):
        z, bias = Ef2(phi_t, tau)
        tau_ = np.sqrt(sa * z + sb * tau0 ** 2)
        print('%d' % i + '-th iteration eror of tau: %.5f' % abs(tau_ - tau))
        tau = tau_


    def phi(x):
        return phi_t(x) - bias


    # def dphi(x):
    #     return np.where(x >= 0, 1, 0)

    g4 = ED2f2(phi, tau)
    g1 = ED1f1(phi, tau) ** 2
    g2 = ED2f1(phi, tau) ** 2

    alpha1 = (1 - sa * g1) ** (-1) * (sb * g1)
    alpha4 = (1 - sa / 2 * g4) ** (-1) * sb
    alpha2 = 1 / 4 * (1 - sa * g1) ** (-1) * g2 * alpha4 ** 2
    alpha3 = 1 / 2 * (1 - sa * g1) ** (-1) * g2 * (sa * alpha1 + sb) ** 2

    dtau = ED2(dphi, tau) ** 0.5

    dalpha0 = sa * g1
    dalpha1 = sa * g2 * (sa * alpha1 + sb)

    kappa = ((1 - sa * dtau ** 2) ** (-1) * z) ** 0.5
    beta1 = alpha1 / (1 - dalpha0)
    beta2 = alpha2 / (1 - dalpha0)
    beta3 = (alpha3 + beta1 * dalpha1) / (1 - dalpha0)

    D = np.concatenate([np.concatenate([beta2 * np.dot(t, t.T) + beta3 * T, beta2 * t], axis=1),
                        np.concatenate([beta2 * t.T, beta2 * np.eye(1)], axis=1)], axis=0)

    K_ = beta1 * X.T @ X + V @ D @ V.T + \
         (kappa ** 2 - tau0 ** 2 * beta1) * np.eye(n)

    evsK_, _ = np.linalg.eig(K_)
    print('evsK_:', evsK_)
    plt.figure(1)
    plt.hist(evsK_, bins=50, density=True, edgecolor='white')

    K = 0
    onematrix = np.ones((n, n))
    for k in range(K_maxiter):
        A = np.random.randn(m, m)
        B = np.random.randn(m, p)
        Z = np.zeros((m, n))
        for i in range(fp_maxiter):
            Z_ = phi(np.sqrt(sa) * A @ Z + np.sqrt(sb) * B @ X) / np.sqrt(m)
            if np.linalg.norm(Z - Z_, 2) > 1e-5:
                Z = Z_
            else:
                print('%d' % k + '-th realization, %d' % i
                      + '-th iteration error: %f' % np.linalg.norm(Z - Z_, 2))
                break
        G = Z.T @ Z
        dZ = dphi(np.sqrt(sa) * A @ Z + np.sqrt(sb) * B @ X) / np.sqrt(m)
        dG = sa * dZ.T @ dZ
        K += np.divide(G, onematrix - dG) / K_maxiter

    evsK, _ = np.linalg.eig(K)
    print('evsK', evsK)
    plt.figure(2)
    plt.hist(evsK, bins=50, density=True, color='m', edgecolor='white')
    print('the approximation error: %f' % np.linalg.norm(K - K_, 2))

if __name__ == '__main__':
    main()