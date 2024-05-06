# -*- coding: utf-8 -*-
import json

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from Gaussian_Integration import *
from scipy.optimize import *
from scipy.special import *


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--p', type=int, default = '2000', help='dimension of gmm_data')
    parser.add_argument('--phi',type = str, default = 'tanh', help='just tanh in this file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args_and_config()
    p = args.p
    n = (int)(p * 0.8)

    def phi_t(x):
        if args.phi == 'tanh':
            return np.tanh(x)
        elif args.phi == 'relu':
            return np.maximum(0, x)

    cn = 2  # class number
    sa = 0.2
    sb = 1 - sa
    tau_maxiter = 20

    m = 1000
    G_maxiter = (int)(p * p / m)
    fp_maxiter = 25

    mu_l = []
    c_l = []
    C_l = []
    X_l = []
    Z_l = []
    C0 = 0
    C0_l = []
    for i in range(cn):
        mu_l.append(np.zeros((p, 1)))
        mu_l[i][8 * (i), 0] = 8

        c_l.append(1 + 8 * i / np.sqrt(p))
        C_l.append(c_l[i] * np.eye(p))
        C0 += C_l[i] / cn

        Z_l.append(np.random.randn(p, n // cn) * np.sqrt(c_l[i]))
        X_l.append(Z_l[i] / np.sqrt(p) + np.tile(mu_l[i] / np.sqrt(p), (1, n // cn)))

    for i in range(cn):
        C0_l.append(C_l[i] - C0)

    X = np.concatenate(X_l, axis=1)

    t = np.zeros((cn, 1))
    for i in range(cn):
        t[i, :] = np.trace(C0_l[i]) / np.sqrt(p)
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

    g4 = ED2f2(phi, tau)
    g1 = ED1f1(phi, tau) ** 2
    g2 = ED2f1(phi, tau) ** 2

    alpha1 = (1 - sa * g1) ** (-1) * (1 - sa)
    alpha4 = (1 - sa / 2 * g4) ** (-1) * (1 - sa)
    alpha2 = sa / 4 * (1 - sa * g1) ** (-1) * g2 * alpha4 ** 2
    alpha3 = sa / 2 * (1 - sa * g1) ** (-1) * g2 * alpha1 ** 2

    def G4(a, c, t):
        return 2 * a ** 2 * (-(2 / np.pi) ** 0.5 * c / t * np.exp(-c ** 2 / 2 / t ** 2) \
                             + erf(c / 2 ** 0.5 / t))

    def G1(a, c, t):
        return a ** 2 * erf(c / 2 ** 0.5 / t) ** 2

    def G2(a, c, t):
        return 0

    def Tau(a, c, t):
        return a * (t ** 2 * erf(c / 2 ** 0.5 / t) \
                    + c * (c * erfc(c / 2 ** 0.5 / t) \
                           - np.exp(-c ** 2 / 2 / t ** 2) * (2 / np.pi) ** 0.5 * t)) ** 0.5

    def equations(vars):

        a1, c1 = vars

        talpha_11 = G1(a1, c1, tau0)
        talpha_12 = G2(a1, c1, tau0) / 4
        talpha_13 = G2(a1, c1, tau0) / 2
        talpha_14 = G4(a1, c1, tau0) / 2
        tau1 = Tau(a1, c1, tau0)

        eq1 = (alpha1 - (1 - sa)) / sa - talpha_11
        eq2 = alpha2 / sa - talpha_12
        eq3 = alpha3 / sa - talpha_13
        eq4 = (tau ** 2 - tau0 ** 2 * alpha1) / sa - (tau1 ** 2 - tau0 ** 2 * talpha_11)

        print('error: ', [eq1, eq2, eq3, eq4],
              'vars：', vars)
        print('error: ', eq1 ** 2 + eq2 ** 2 + eq3 ** 2 + eq4 ** 2)
        return eq1 ** 2 + eq2 ** 2 + eq3 ** 2 + eq4 ** 2

    initial_guess = [0.1, 0.1]
    try:
        # Use fsolve to solve the system of equations
        # result = fsolve(equations, initial_guess)
        result = minimize(equations, initial_guess,
                          method='SLSQP', tol=1e-10)
        print("Numerical solution:", result)
    except Exception as e:
        print("Optimization failed. Message:", str(e))

    [a1, c1] = result.x

    talpha_11 = G1(a1, c1, tau0)
    talpha_12 = G2(a1, c1, tau0) / 4
    talpha_13 = G2(a1, c1, tau0) / 2
    talpha_14 = G4(a1, c1, tau0) / 2
    tau1 = Tau(a1, c1, tau0)

    C = np.concatenate([np.concatenate([alpha2 * np.dot(t, t.T) + alpha3 * T, alpha2 * t], axis=1),
                        np.concatenate([alpha2 * t.T, alpha2 * np.eye(1)], axis=1)], axis=0)
    G1_ = alpha1 * X.T @ X + V @ C @ V.T + \
          (tau ** 2 - tau0 ** 2 * alpha1) * np.eye(n)
    G_ = (G1_ - (1 - sa) * X.T @ X) / sa

    tC2 = np.concatenate([np.concatenate([talpha_12 * np.dot(t, t.T) + talpha_13 * T, talpha_12 * t], axis=1),
                          np.concatenate([talpha_12 * t.T, talpha_12 * np.eye(1)], axis=1)], axis=0)

    G2_ = talpha_11 * X.T @ X + V @ tC2 @ V.T + \
          (tau1 ** 2 - tau0 ** 2 * talpha_11) * np.eye(n)

    # empirical implicit CK G and  empirical equivalent single-layer htanh CK: G2
    def htanh(x):
        return a1 * np.maximum(-c1, np.minimum(c1, x))

    G2 = 0

    G = 0
    for k in range(G_maxiter):
        A = np.random.randn(m, m)
        B = np.random.randn(m, p)
        Z = np.zeros((m, n))
        for i in range(fp_maxiter):
            Z_ = phi(np.sqrt(sa) * A @ Z + np.sqrt(sb) * B @ X) / np.sqrt(m)
            if np.linalg.norm(Z - Z_, 2) > 5e-6:
                Z = Z_
            else:
                print('%d' % k + '-th realization, %d' % i
                      + '-th iteration error: %f' % np.linalg.norm(Z - Z_, 2))
                break
        G += Z.T @ Z / G_maxiter
        tZ = htanh(B @ X) / np.sqrt(m)
        G2 += tZ.T @ tZ / G_maxiter

    print('|G-G2|:', np.linalg.norm(G - G2, 2))
    print('|G_|:', np.linalg.norm(G_, 2))

    result_list = []
    result_dict = {}
    result_dict["p"] = args.p
    result_dict["|G-G2|"] = np.linalg.norm(G - G2, 2)
    result_dict["|G_|"] = np.linalg.norm(G_, 2)
    result_dict["phi"] = args.phi
    result_dict["m"] = m
    result_list.append(result_dict)
    output_path = './onelayer_matching_result'
    target_folder = output_path
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)
    # result_json_file = 'phi_'+ args.phi +'p_' + str(p) + "_n_" + str(n) +'.json'
    result_json_file = 'phi_' + args.phi + '_onelayer_matching' + '.json'
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