import math
from scipy.integrate import quad
import numpy as np
def get_tau(tau0):
    tau = 0
    s = 0.1
    for i in range(1,20):
        tau = math.sqrt(s*Ef2(tau)+(1-s)*tau0**2)
    return tau
def g(x):
    return np.exp(-x**2/2) / np.sqrt(2*np.pi)

def Ef2(t):
    integrand = lambda x: (np.maximum(t*x, 0) - t/np.sqrt(2*np.pi))**2 * g(x)
    y, _ = quad(integrand, -np.inf, np.inf)
    return y

def estim_tau_tensor(X):
    tau = np.mean(np.diag(X @ X.T))
    return tau
