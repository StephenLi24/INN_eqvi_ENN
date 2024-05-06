# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import quad

def g(x, mu=0, sigma=1):  # gaussian_pdf
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def ED2(f, t):
    def d2(x):
        return (f(t * x))**2 * g(x)
    result, _ = quad(d2, -np.inf, np.inf)
    return result

def Ef2(f, t):  # E[phi_t^2(tx)]
    def f1(x):
        return f(t * x) * g(x)  # E[phi_t(tx)]
    bias, _ = quad(f1, -np.inf, np.inf)

    def f2(x):
        return (f(t * x) - bias)**2 * g(x)
    result, _ = quad(f2, -np.inf, np.inf)
    return result, bias


def ED2f2(f, t):  # E[(f^2(tx))''] = E[(x^2-1)f^2(tx)]/t^2
    def d2f2(x):
        return (x**2 - 1) * f(t * x)**2 * g(x)
    result, _ = quad(d2f2, -np.inf, np.inf) 
    return result / t**2


def ED1f1(f, t):  # E[f'(tx)] = E[xf(tx)]/t
    def d1f1(x):
        return x * f(t * x) * g(x)
    result, _ = quad(d1f1, -np.inf, np.inf)
    return result / t


def ED2f1(f, t):  # E[f''(tx)] = E[(x^2-1)f(tx)]/t^2
    def d2f1(x):
        return (x**2 - 1) * f(t * x) * g(x)
    result, _ = quad(d2f1, -np.inf, np.inf) 
    return result / t**2