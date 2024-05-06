import json
import numpy as np
import matplotlib.pyplot as plt
from Gaussian_Integration import *
import os
import argparse
def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--p', type=int, default = '500', help='dimension of gmm_data')
    parser.add_argument('--phi',type = str, default = 'relu', help='tanh or relu')
    parser.add_argument('--output_path', type = str, default = './gmm_result', help='output path of json file')
    args = parser.parse_args()
    return args
def main():
    args = parse_args_and_config()
    p = args.p
    n = (int) (p * 0.8)
    cn = 2  # class number
    sa = 0.2
    sb = 1-sa
    tau_maxiter = 20
    m = 1000
    G_maxiter = (int)(p * p / m)
    fp_maxiter = 25

    def phi_t(x):
        if args.phi == 'tanh':
            return np.tanh(x)
        elif args.phi == 'relu':
            return np.maximum(0, x)
        elif args.phi == 'leaky_relu':
            return np.maximum(0.01 * x, x)
        elif args.phi == 'elu':
            return np.where(x >= 0, x, 1.0 * (np.exp(x) - 1))
        elif args.phi == 'swish':
            return x * (1.0 / (1.0 + np.exp(-1.0 * x)))


    mu_l = []
    c_l = []
    C_l = []
    X_l = []
    Z_l = []
    C0 = 0
    C0_l = []
    for i in range(cn):
        mu_l.append(np.zeros((p, 1)))
        mu_l[i][8*(i), 0] = 8

        c_l.append(1 + 8 * i / np.sqrt(p))
        C_l.append(c_l[i] * np.eye(p))
        C0 += C_l[i] / cn

        Z_l.append(np.random.randn(p, n//cn) * np.sqrt(c_l[i]))
        X_l.append(Z_l[i] / np.sqrt(p) + np.tile(mu_l[i] / np.sqrt(p), (1, n//cn)))
    for i in range(cn):
        C0_l.append(C_l[i]-C0)

    X = np.concatenate(X_l, axis=1)
    # X : the data we neeed
    t = np.zeros((cn, 1))
    for i in range(cn):
        t[i, :] = np.trace(C0_l[i]) / np.sqrt(p)
    T = np.zeros((cn, cn))
    for i in range(cn):
        for j in range(cn):
            T[i, j] = np.trace(C0_l[i] * C0_l[j]) / p
    Psi_l = []
    for i in range(cn):
        VZN = np.sum(Z_l[i]**2, axis=0) / p
        CC = np.tile(np.trace(C_l[i]) / p, (1, n//cn))
        Psi_l.append(VZN - CC)
    Psi = np.concatenate(Psi_l, axis=1)

    J_l = []
    for i in range(cn):
        one_hot = np.eye(cn)[:, i]
        Jt = np.tile(one_hot, (n//cn, 1))
        J_l.append(Jt)
    J = np.concatenate(J_l, axis=0)
    V = np.concatenate([J/np.sqrt(p), Psi.T], axis = 1)

    # tau0 = np.sqrt(np.trace(C0) / p)
    tau0 = np.sqrt(np.sum(X**2) / n)

    tau = 0
    for i in range(tau_maxiter):
        z, bias = Ef2(phi_t, tau)
        tau_ = np.sqrt(sa * z + sb * tau0 **2)
        print('%d'%i + '-th iteration eror of tau: %.5f' % abs(tau_-tau))
        tau = tau_

    def phi(x):
        return phi_t(x) - bias

    g4 = ED2f2(phi, tau)
    g1 = ED1f1(phi, tau)**2
    g2 = ED2f1(phi, tau)**2


    alpha1 =(1-sa*g1)**(-1)*(1-sa)
    alpha4 = (1-sa/2*g4)**(-1)*(1-sa)
    alpha2 = sa/4 * (1-sa*g1)**(-1) * g2 * alpha4**2
    alpha3 = sa/2 * (1-sa*g1)**(-1) * g2 * alpha1**2

    C = np.concatenate([np.concatenate([alpha2*np.dot(t, t.T)+alpha3*T,alpha2*t], axis=1 ),
                      np.concatenate([alpha2 *t.T, alpha2 * np.eye(1)], axis=1)], axis=0)

    G_ = (alpha1-(1-sa)) * X.T @ X + V @ C @ V.T +\
        (tau**2 - tau0**2 * alpha1- tau0**4 * 0) * np.eye(n)
    G_ = G_ / sa
    evsG_, _ = np.linalg.eig(G_)
    print('evsG_', evsG_)
    # plt.figure(1)
    # plt.hist(evsG_, bins=50, density=True, edgecolor='white')
    # plt.show()

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
                print('%d'%k + '-th realization, %d'%i
                      + '-th iteration error: %f' % np.linalg.norm(Z - Z_, 2) )
                break
        G += Z.T @ Z / G_maxiter

    evsG, _ = np.linalg.eig(G)
    print('evsG', evsG)
    # plt.figure(2)
    # plt.hist(evsG, bins=50, density=True, color='m', edgecolor='white')
    # plt.show()
    print('the approximation error: %f'% np.linalg.norm(G - G_, 2))
    result_list = []
    result_dict = {}
    result_dict["p"] = args.p
    result_dict["np.linalg.norm(G - G_, 2) = "] = np.linalg.norm(G - G_, 2)
    result_dict["np.linalg.norm(G, 2) = "] = np.linalg.norm(G, 2)
    result_dict["np.linalg.norm(G_, 2)"] = np.linalg.norm(G_, 2)
    result_dict["phi"] = args.phi
    result_dict["m"] = m
    result_list.append(result_dict)
    output_path = args.output_path
    target_folder = output_path
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)
    # result_json_file = 'phi_'+ args.phi +'p_' + str(p) + "_n_" + str(n) +'.json'
    result_json_file = 'phi_'+ args.phi +'.json'
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


    # with open(result_json_file, "w") as f:
    #     json.dump(result_dict, f)
if __name__ == '__main__':
    main()