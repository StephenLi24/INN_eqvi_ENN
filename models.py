import torch
import torch.nn as nn
import math
import torch.autograd as autograd
import torch.nn.functional as F

class L_relu(nn.Module):
    def __init__(self, input_dim, m, w_1, w_2, phi_list, tau0, tau1):
        super(L_relu, self).__init__()
        self.input_dim = input_dim
        self.m = m
        self.w_1 = w_1
        self.w_2 = w_2
        [self.a1, self.b1, self.a2, self.b2] = phi_list
        self.tau0 = tau0
        self.tau1 = tau1

    def forward(self, x):
        x = x.to(torch.float32)
        y1 = self.sigma1(x @ self.w_1, self.a1, self.b1, self.a2, self.b2, self.tau0) / (self.m ** 0.5)
        y2 = self.sigma2(y1 @ self.w_2, self.a1, self.b1, self.a2, self.b2, self.tau1) / (self.m ** 0.5)
        return y2

    def sigma1(self, x, a1, b1, a2, b2, tau0):
        return torch.max(a1 * x, b1 * x) - (a1 - b1) / (2 * math.pi)**0.5 * tau0

    def sigma2(self, x, a1, b1, a2, b2, tau1):
        return torch.max(a2 * x, b2 * x) - (a2 - b2) / (2 * math.pi)**0.5 * tau1

class ol_tanh(nn.Module):
    def __init__(self, a, c, w_matrix, dim):
        super(ol_tanh, self).__init__()
        self.a = a
        self.c = c
        self.w_matrix = w_matrix
        self.m = dim
        
    def forward(self, x):
        x = x.to(torch.float32)
        return self.phi_t(x @ self.w_matrix) / (self.m ** 0.5)
    def phi_t(self, x):
        return self.a * torch.clamp(x, -self.c, self.c)

class tanh(nn.Module):
    def __init__(self, w_matrix, dim):
        super(tanh, self).__init__()
        self.w_matrix = w_matrix
        self.m = dim
    def forward(self, x):
        x = x.to(torch.float32)
        return self.phi_t(x @ self.w_matrix) /  (self.m ** 0.5)
    def phi_t(self, x):
        return torch.tanh(x)

class Explicit_relu(nn.Module):
    def __init__(self, input_dim, output_dim, a, tau0, wmatrix, w2, tau1):
        super(Explicit_relu, self).__init__()
        self.wmatrix = wmatrix
        self.w2 = w2
        self.a = a
        self.tau0 = tau0
        self.output_dim = output_dim
        self.tau1 = tau1


    def forward(self, x):
        x = x.to(torch.float32)
        y = torch.matmul(x, self.wmatrix) - self.a * self.tau0 / ((math.pi * 2) ** 0.5)
        y_ = F.relu(y) / math.sqrt(self.output_dim)
        y_ = y_ @ self.w2 - 1 / (2 * math.pi)**0.5 * self.tau1
        y_ = F.relu(y_) / math.sqrt(self.output_dim)

        return y_



class SingleLayerNN(nn.Module):

    def __init__(self, input_dim, m, s, tau, device, batch_size):
        super(SingleLayerNN, self).__init__()
        # self.input_dim = input_dim
        self.m = torch.tensor(m).to(device)
        self.s = torch.tensor(s).to(device)
        self.tau = torch.tensor(tau).to(device)
        self.w_A = torch.randn(m, m).to(device)
        self.w_B = torch.randn(m, input_dim).to(device)
        self.device = device
        self.bsz = batch_size

    def forward(self, z, x):
        # Z_ = phi(sqrt(s) * A * Z + sqrt(1 - s) * B * X) / sqrt(m);
        x = x.to(torch.float32)
        y = torch.rand(self.bsz, self.m)
        for i in range(x.size()[0]):
            # y_ = torch.matmul(self.w_A, z[i])
            y_ = torch.sqrt(self.s ** 2) * torch.matmul(self.w_A, z[i].to(self.device)) + torch.sqrt(
                1 - self.s ** 2) * torch.matmul(self.w_B, x[i].to(self.device))
            y[i] = y_
        output = self.shift_relu(y, self.tau) / torch.sqrt(self.m)
        return output
    def shift_relu(self, x, tau):
        x = torch.tensor(x).to(self.device)
        tau = torch.tensor(tau).to(self.device)
        return torch.relu(x) - tau / torch.sqrt(torch.tensor(2).to(self.device)*torch.pi)

class tanhNN(nn.Module):
    def __init__(self, input_dim, m, s, tau, device, batch_size):
        super(tanhNN, self).__init__()
        # self.input_dim = input_dim
        self.m = torch.tensor(m).to(device)
        self.s = torch.tensor(s).to(device)
        self.tau = torch.tensor(tau).to(device)
        self.w_A = torch.randn(m, m).to(device)
        self.w_B = torch.randn(m, input_dim).to(device)
        self.device = device
        self.bsz = batch_size

    def forward(self, z, x):
        # Z_ = phi(sqrt(s) * A * Z + sqrt(1 - s) * B * X) / sqrt(m);
        # print("sizez,x",x.size(),z.size())
        x = x.to(torch.float32)
        y = torch.rand(self.bsz, self.m)
        for i in range(x.size()[0]):
            # print(self.w_A.shape(),z[i].shape())
            # y_ = torch.matmul(self.w_A, z[i])
            y_ = torch.sqrt(self.s ** 2) * torch.matmul(self.w_A, z[i].to(self.device)) + torch.sqrt(
                1 - self.s ** 2) * torch.matmul(self.w_B, x[i].to(self.device))
            y[i] = y_
        output = self.tanh(y) / torch.sqrt(self.m)
        return output
    def tanh(self, x):
        x = torch.tensor(x).to(self.device)
        return torch.tanh(x)

def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0):
    """ Anderson acceleration for fixed point iteration. """
    bsz, D = x0.shape
    # print('bsz,D',bsz,D)
    X = torch.zeros(bsz, m, D, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, D, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].view_as(x0)).view(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]
        # alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)
        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].view_as(x0)).view(bsz, -1)
        res.append((F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item()))
        if (res[-1] < tol):
            break
    return X[:, k % m].view_as(x0), res


class DEQFixedPoint(nn.Module):
    def __init__(self, f, dim, **kwargs):
        super().__init__()
        self.f = f
        self.solver = anderson
        self.kwargs = kwargs
        self.dim = dim


    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z: self.f(z, x), torch.zeros(x.size()[0],self.dim), **self.kwargs)
        z = self.f(z, x)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)

        def backward_hook(grad):
            g, self.backward_res = self.solver(lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g
        # z.register_hook(backward_hook)
        return z

