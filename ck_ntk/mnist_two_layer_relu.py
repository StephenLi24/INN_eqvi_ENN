# -*- coding: utf-8 -*-
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from Gaussian_Integration import *
from scipy.optimize import *
import data_prepare
from scipy.special import *

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--n', type=int, default = '400', help='dimension of gmm_data')
    parser.add_argument('--phi',type = str, default = 'tanh', help='tanh or relu')
    args = parser.parse_args()
    return args

def main():
    args = parse_args_and_config()
    p = 784
    n = args.n
    cn = 2  # class number
    sa = 0.2
    sb = 1 - sa
    tau_maxiter = 20

    def phi_t(x):
        if args.phi == 'tanh':
            return np.tanh(x)
        elif args.phi == 'relu':
            return np.maximum(0, x)

    res = data_prepare.gen_data(
        'MNIST',
        selected_target=[6, 8],
        T=n,
        p=p,
        cs=[0.5, 0.5]
    )
    X = res[0]
    m = 1000
    # G_maxiter = (int)(p * p / m)
    fp_maxiter = 25
    G_maxiter = 10
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
    tau0 = np.sqrt(np.sum(X ** 2) / n)
    name = 'ReLU'

    #################### The coefficients of equiv CK of the INN ####################
    def phi_t(x):
        if name == 'tanh':
            return np.tanh(x)
        if name == 'ReLU':
            return np.maximum(0, x)

    tau = 0
    for i in range(tau_maxiter):
        z, bias = Ef2(phi_t, tau)
        tau_ = np.sqrt(sa * z + sb * tau0 ** 2)
        print('%d' % i + '-th iteration eror of tau: %.5f' % abs(tau_ - tau))
        tau = tau_

    def phi(x):
        return phi_t(x) - bias

    g4 = ED2f2(phi, tau)
    g1 = ED1f1(phi, tau) ** 2
    g2 = ED2f1(phi, tau) ** 2

    alpha1 = (1 - sa * g1) ** (-1) * (1 - sa)
    alpha4 = (1 - sa / 2 * g4) ** (-1) * (1 - sa)
    alpha2 = sa / 4 * (1 - sa * g1) ** (-1) * g2 * alpha4 ** 2
    alpha3 = sa / 2 * (1 - sa * g1) ** (-1) * g2 * alpha1 ** 2

    # Gaussian integration for centered L-ReLU:
    # Max(ax,bx)- (a - b)t / (2pi)^(1/2)
    def G4(a, b):
        return ((a ** 2 + b ** 2) * (np.pi - 1) + 2 * a * b) / np.pi

    def G1(a, b):
        return (a + b) ** 2 / 4

    def G2(a, b, t):
        return (a - b) ** 2 / 2 / np.pi / t ** 2

    def Tau(a, b, t):
        return (((a ** 2 + b ** 2) * (np.pi - 1) + 2 * a * b)
                / 2 / np.pi) ** 0.5 * t

    #################### The equation for solving the activation for the equiv ENN ####################
    def equations(vars):

        a1, b1, a2, b2 = vars

        talpha_11 = G1(a1, b1)
        talpha_12 = G2(a1, b1, tau0) / 4
        talpha_13 = G2(a1, b1, tau0) / 2
        talpha_14 = G4(a1, b1) / 2
        tau1 = Tau(a1, b1, tau0)

        talpha_21 = G1(a2, b2) * talpha_11
        talpha_22 = G1(a2, b2) * talpha_12 \
                    + G2(a2, b2, tau1) / 4 * talpha_14 ** 2
        talpha_23 = G1(a2, b2) * talpha_13 \
                    + G2(a2, b2, tau1) / 2 * talpha_11 ** 2
        tau2 = Tau(a2, b2, tau1)

        eq1 = (alpha1 - (1 - sa)) / sa - talpha_21
        eq2 = alpha2 / sa - talpha_22
        eq3 = alpha3 / sa - talpha_23
        eq4 = (tau ** 2 - tau0 ** 2 * alpha1) / sa \
              - (tau2 ** 2 - tau0 ** 2 * talpha_21)

        print('error: ', [eq1, eq2, eq3, eq4],
              'vars: ', vars)
        return eq1 ** 2 + eq2 ** 2 + eq3 ** 2 + eq4 ** 2

    if name == 'tanh':
        initial_guess = [1, 1, 1, 1]
    if name == 'ReLU':
        initial_guess = [1, 0.1, 1, 0.1]

    try:
        result = minimize(equations, initial_guess,
                          method='SLSQP', tol=1e-10)
        print("Numerical solution:", result)
    except Exception as e:
        print("Optimization failed. Message:", str(e))

    [a1, b1, a2, b2] = result.x

    #################### The activation ####################
    # max(ax, bx) - (a - b)t / (2pi)^(1/2)
    def sigma1(x):
        return np.maximum(a1 * x, b1 * x) - (a1 - b1) / (2 * np.pi) ** 0.5 * tau0

    def sigma2(x):
        return np.maximum(a2 * x, b2 * x) - (a2 - b2) / (2 * np.pi) ** 0.5 * tau1

    #################### Check the result ####################
    talpha_11 = G1(a1, b1)
    talpha_12 = G2(a1, b1, tau0) / 4
    talpha_13 = G2(a1, b1, tau0) / 2
    talpha_14 = G4(a1, b1) / 2
    tau1 = Tau(a1, b1, tau0)

    talpha_21 = G1(a2, b2) * talpha_11
    talpha_22 = G1(a2, b2) * talpha_12 \
                + G2(a2, b2, tau1) / 4 * talpha_14 ** 2
    talpha_23 = G1(a2, b2) * talpha_13 \
                + G2(a2, b2, tau1) / 2 * talpha_11 ** 2
    tau2 = Tau(a2, b2, tau1)

    # equivalent CK of the two-layer ENN
    tC2 = np.concatenate([np.concatenate([talpha_22 * np.dot(t, t.T) + talpha_23 * T, talpha_22 * t], axis=1),
                          np.concatenate([talpha_22 * t.T, talpha_22 * np.eye(1)], axis=1)], axis=0)

    G2_ = talpha_21 * X.T @ X + V @ tC2 @ V.T \
          + (tau2 ** 2 - tau0 ** 2 * talpha_21) * np.eye(n)
    evsG2_, _ = np.linalg.eig(G2_)

    # empirical CK of the two-layer ENN
    nbins = 40

    G = 0
    G2 = 0
    for k in range(G_maxiter):
        W1 = np.random.randn(m, p)
        W2 = np.random.randn(m, m)
        Y1 = sigma1(W1 @ X) / m ** 0.5
        Y2 = sigma2(W2 @ Y1) / m ** 0.5
        G2 += Y2.T @ Y2 / G_maxiter

        Z = np.zeros((m, n))
        for i in range(fp_maxiter):
            Z_ = phi(np.sqrt(sa) * W2 @ Z + np.sqrt(sb) * W1 @ X) / np.sqrt(m)
            if np.linalg.norm(Z - Z_, 2) > 5e-6:
                Z = Z_
            else:
                print('%d' % k + '-th realization, %d' % i
                      + '-th iteration error: %f' % np.linalg.norm(Z - Z_, 2))
                break
        G += Z.T @ Z / G_maxiter  # empirical CK of the INN

    # equivalent CK of the INN
    C = np.concatenate([np.concatenate([alpha2 * np.dot(t, t.T) + alpha3 * T, alpha2 * t], axis=1),
                        np.concatenate([alpha2 * t.T, alpha2 * np.eye(1)], axis=1)], axis=0)
    G1_ = alpha1 * X.T @ X + V @ C @ V.T \
          + (tau ** 2 - tau0 ** 2 * alpha1) * np.eye(n)
    G_ = (G1_ - (1 - sa) * X.T @ X) / sa
    evsG_, _ = np.linalg.eig(G_)

    evsG, _ = np.linalg.eig(G)
    print('evsG', evsG)
    plt.figure(1)
    plt.hist(evsG, bins=50, density=True, color='m', edgecolor='white')
    plt.show()
    evsG2, _ = np.linalg.eig(G2)
    print('evsG2', evsG2)
    plt.figure(2)
    plt.hist(evsG2, bins=50, density=True, color='r', edgecolor='white')
    plt.show()
    print('|G-G2|:', np.linalg.norm(G - G2, 2))
    print('|G_|:', np.linalg.norm(G_, 2))

    # print('|G_-G2_|:', np.linalg.norm(G_ - G2_, 2))
    # print('|G2-G2_|:', np.linalg.norm(G2 - G2_, 2))
    # print('|G-G2|:', np.linalg.norm(G - G2, 2))
    # print('|G-G_|:', np.linalg.norm(G - G_, 2))
    # print('|G-G2_|:', np.linalg.norm(G - G2_, 2))
    result_list = []
    result_dict = {}
    result_dict["p"] = args.p
    result_dict["|G-G2|"] = np.linalg.norm(G - G2, 2)
    result_dict["|G_|"] = np.linalg.norm(G_, 2)
    result_dict["phi"] = args.phi
    result_dict["m"] = m
    result_list.append(result_dict)
    output_path = './twolayer_matching_result'
    target_folder = output_path
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)
    # result_json_file = 'phi_'+ args.phi +'p_' + str(p) + "_n_" + str(n) +'.json'
    result_json_file = 'phi_' + args.phi + '_twolayer_matching' + '.json'
    result_json_file = os.path.join(target_folder, result_json_file)
    # Read existing data from the JSON file, or initialize as an empty list
    try:
        with open(result_json_file, 'r') as json_file:
            existing_data = json.load(json_file)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        existing_data = []

    # Append the new data list to the existing data
    existing_data.append(result_dict)

    # Write the combined data back to the JSON file
    with open(result_json_file, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)


if __name__ == '__main__':
    main()