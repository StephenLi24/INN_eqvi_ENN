import numpy as np
import matplotlib.pyplot as plt
from Gaussian_Integration import *
from scipy.optimize import *
from scipy.special import *
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


class one_layer_tanh:
    def __init__(self, data, name='tanh'):
        self.X = data
        self.sa = 0.1
        self.sb = 1 - self.sa
        self.p = self.X.shape[1]
        self.nbins = 40
        # self.tau0 = np.sqrt(np.sum(self.X**2)/ self.X.shape[0])
        self.tau0 = np.mean(np.diag(self.X @ self.X.T))
        self.name = name
        self.tau_maxiter = 20
        self.tau, self.bias = self.estimate_tau()

    #################### The coefficients of equiv CK of the INN ####################
    def phi_t(self, x):
        if self.name == 'tanh':
            return np.tanh(x)
        if self.name == 'ReLU':
            return np.maximum(0, x)

    def estimate_tau(self):
        tau = 0
        bias = 0
        for i in range(self.tau_maxiter):
            z, bias = Ef2(self.phi_t, tau)
            tau_ = np.sqrt(self.sa * z + self.sb * self.tau0 ** 2)
            print('%d' % i + '-th iteration eror of tau: %.5f' % abs(tau_ - tau))
            tau = tau_
        return tau, bias

    def phi(self, x):
        return self.phi_t(x) - self.bias

    #################### The equation for solving the activation for the equiv ENN ####################
    def equations(self, vars):
        g4 = ED2f2(self.phi, self.tau)
        g1 = ED1f1(self.phi, self.tau) ** 2
        g2 = ED2f1(self.phi, self.tau) ** 2

        alpha1 = (1 - self.sa * g1) ** (-1) * (1 - self.sa)
        alpha4 = (1 - self.sa / 2 * g4) ** (-1) * (1 - self.sa)
        alpha2 = self.sa / 4 * (1 - self.sa * g1) ** (-1) * g2 * alpha4 ** 2
        alpha3 = self.sa / 2 * (1 - self.sa * g1) ** (-1) * g2 * alpha1 ** 2

        a1, c1 = vars

        talpha_11 = G1(a1, c1, self.tau0)
        talpha_12 = G2(a1, c1, self.tau0) / 4
        talpha_13 = G2(a1, c1, self.tau0) / 2
        talpha_14 = G4(a1, c1, self.tau0) / 2
        tau1 = Tau(a1, c1, self.tau0)

        eq1 = (alpha1 - (1 - self.sa)) / self.sa - talpha_11
        eq2 = alpha2 / self.sa - talpha_12
        eq3 = alpha3 / self.sa - talpha_13
        eq4 = (self.tau ** 2 - self.tau0 ** 2 * alpha1) / self.sa - (tau1 ** 2 - self.tau0 ** 2 * talpha_11)

        print('error: ', [eq1, eq2, eq3, eq4],
                'vars: ', vars)
        print('error: ', eq1 ** 2 + eq2 ** 2 + eq3 ** 2 + eq4 ** 2)
        return eq1 ** 2 + eq2 ** 2 + eq3 ** 2 + eq4 ** 2

    def get_activation(self):
        initial_guess = [0.1, 0.1]

        try:
            # Use fsolve to solve the system of equations
            # result = fsolve(equations, initial_guess)
            result = minimize(self.equations, initial_guess,
                              method='SLSQP', tol=1e-10)
            print("Numerical solution:", result)
        except Exception as e:
            print("Optimization failed. Message:", str(e))

        [a1, c1] = result.x
        # [a1, b1, a2, b2] = result.x
        return result.x
