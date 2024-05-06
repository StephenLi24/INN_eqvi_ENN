# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import data_prepare
import models
from train import model_train
import sys
from two_layer_lrelu import *
from one_layer_tanh import *

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--dataset', type=str, default = 'cifar10', help='Choose from mnist, fashion_mnist, cifar10')
    parser.add_argument('--model', type=str, default='l_relu_enn', help='Choose your model:l_relu_enn, inn, relu_enn, tanh_inn, h_tanh_enn, tanh_enn')
    parser.add_argument('--dim', type=int, default= 1024, help='The dimension of our exp')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--batch_size', type=int, default=128,help = 'batch size')
    parser.add_argument('--save_path', type=str, default='./check_points', help='save checkpoints path')
    parser.add_argument('--dataset_path', type = str, default='/data', help='dataset path')
    parser.add_argument('--device',type=str, default='cuda:0', help='cuda:0 or cpu')
    parser.add_argument('--vgg_path',type=str, default='./pretrained/cifar10_vgg.pth', help = 'pretrained vgg model path')
    parser.add_argument('--output_path', type=str, default='./result/', help='output path')
    parser.add_argument('--asquare', type=float, default=0.2, help='square of varience of w matrix in relu')
    args = parser.parse_args()
    return args

# standard training or evaluation loop


def main():
    args = parse_args_and_config()
    if args.dataset == 'mnist' :
        train, test, tau = data_prepare.mnist_prepared(args.dataset_path)
    elif args.dataset == 'fashion_mnist':
        train, test, tau = data_prepare.fashionmnist_prepared(args.dataset_path)
    elif args.dataset == 'cifar10':
        train, test, tau = data_prepare.vgg_cifa10_feature(args.vgg_path, args.dataset_path)
    else:
        print('Wrong dataset, choose from mnist, fashion_mnist, cifar10')
        return False
    if(args.model == 'l_relu_enn' or args.model == 'relu_enn'):
    #get l_relu two layer matching coefficients
        phi_list = two_layer_lrelu(train.X).get_activation()
        [a1, b1, a2, b2] = phi_list
        tau0 = two_layer_lrelu(train.X).tau0
        tau1 = Tau(a1, b1, tau0)
        tau = two_layer_lrelu(train.X).tau
        print("tau:", tau)
        print("tau0: ", tau0, "tau1: ",tau1)
        print("[a1, b1, a2, b2]", "[", a1, b1, a2, b2,"]")
    if(args.model == 'h_tanh_enn'):
        [a, c] = one_layer_tanh(train.X).get_activation()
        tau0 = two_layer_lrelu(train.X).tau0
        tau1 = Tau(a, c, tau0)
        tau = two_layer_lrelu(train.X).tau
        print("tau:", tau)
        print("tau0: ", tau0, "tau1: ", tau1)
        print("[a, c]", "[", a, c, "]")
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=8,drop_last=True)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=8,drop_last=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    dim = args.dim
    if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        input_dimension = 784
        varience = args.asquare ** 0.5
        b = (1 - varience ** 2) ** 0.5 # coeficients of w matrix
        w_matrix = torch.randn(input_dimension, dim).to(args.device)
        w_1 = torch.randn(input_dimension, dim).to(args.device)
        w_2 = torch.randn(dim, dim).to(args.device)
        if args.model == 'inn':
            f = models.SingleLayerNN(input_dimension, tau= tau, s=varience, m=dim, device = device, batch_size = args.batch_size)
            model = nn.Sequential(models.DEQFixedPoint(f = f, tol=1e-2, max_iter=25, m=5, dim=dim),
                                                       nn.Linear(dim, out_features=10, bias=True)).to(device)
        elif args.model == 'l_relu_enn':
            model = nn.Sequential(models.L_relu(input_dimension, dim, w_1, w_2, phi_list=[a1, b1, a2, b2], tau0=tau0, tau1=tau1),
                                  nn.Linear(dim, 10)).to(device)
        elif args.model == 'relu_enn':
            model = nn.Sequential(models.Explicit_relu(input_dimension, dim, a = varience, tau0=tau0, wmatrix = b * w_matrix, w2= w_2, tau1= tau1),
                                  nn.Linear(dim, 10)).to(device)
        elif args.model == 'h_tanh_enn':
            model = nn.Sequential(models.ol_tanh(torch.tensor(a).to(args.device), torch.tensor(c).to(device), w_matrix, dim),
                                  nn.Linear(dim, 10)).to(device)
        elif args.model == 'tanh_enn':
            model = nn.Sequential(models.tanh(varience * w_matrix, dim),
                                  nn.Linear(dim, 10)).to(device)
        elif args.model == 'tanh_inn':
            f = models.tanhNN(input_dimension, tau= tau, s=varience, m=dim, device = device, batch_size = args.batch_size)
            model = nn.Sequential(models.DEQFixedPoint(f = f, tol=1e-2, max_iter=25, m=5, dim=dim),
                                                       nn.Linear(dim, out_features=10, bias=True)).to(device)
        else:
            print('Wrong model name! Choose your model:l_relu_enn, inn, relu_enn, tanh_inn, h_tanh_enn, tanh_enn')
            return False
        mnist = model_train(model = model, model_name = args.model, train_loader = train_loader, test_loader = test_loader,
                                dim = dim, epoch_number= args.epoch, lr=args.lr, device=args.device, output_path = args.output_path, dataset_name = args.dataset)
        mnist.mnist_train()
        print('Successfully!')
        return 0
    elif args.dataset == 'cifar10':
        input_dimension = 256
        varience = args.asquare ** 0.5
        w_matrix = torch.randn(input_dimension, dim).to(args.device)
        w_1 = torch.randn(input_dimension, dim).to(args.device)
        w_2 = torch.randn(dim, dim).to(args.device)
        print(w_matrix.shape)

        if args.model == 'inn':
            f = models.SingleLayerNN(input_dimension, tau=tau, s=varience, m=dim, device= args.device, batch_size = args.batch_size)
            model = nn.Sequential(models.DEQFixedPoint(f = f, tol= 1e-2, max_iter=25, m=5, dim=dim),
                                  nn.Linear(dim, out_features=10, bias=True)).to(device)
        elif args.model == 'l_relu_enn':
            model = nn.Sequential(
                models.L_relu(input_dimension, dim, w_1, w_2, phi_list=[a1, b1, a2, b2], tau0=tau0, tau1=tau1),
                nn.Linear(dim, 10)).to(device)
        elif args.model == 'relu_enn':
            b = (1 - args.asquare) ** 0.5 # coeficients of w matrix to make exp in same condition
            model = nn.Sequential(
                models.Explicit_relu(input_dimension, dim, a = varience, tau0=tau0, wmatrix = b * w_matrix, w2=w_2, tau1= tau1),
                nn.Linear(dim, 10)).to(device)

        elif args.model == 'h_tanh_enn':
            model = nn.Sequential(models.ol_tanh(torch.tensor(a).to(args.device), torch.tensor(c).to(device), w_matrix, dim),
                                  nn.Linear(dim, 10)).to(device)
        elif args.model == 'tanh_enn':
            b = (1 - args.asquare) ** 0.5 # coeficients of w matrix
            model = nn.Sequential(models.tanh(b * w_matrix, dim),
                                  nn.Linear(dim, 10)).to(device)
        elif args.model == 'tanh_inn':
            f = models.tanhNN(input_dimension, tau= tau, s=varience, m=dim, device = device, batch_size = args.batch_size)
            model = nn.Sequential(models.DEQFixedPoint(f = f, tol=1e-2, max_iter=25, m=5, dim=dim),
                                                       nn.Linear(dim, out_features=10, bias=True)).to(device)
        else:
            print('Wrong model name, choose your model from:l_relu_enn, inn, relu_enn, tanh_inn, h_tanh_enn, tanh_enn')
            return False
        cifar10 = model_train(model = model, model_name = args.model, train_loader = train_loader, test_loader = test_loader,
                                dim = dim, epoch_number= args.epoch, lr=args.lr, device=args.device, output_path = args.output_path, dataset_name = args.dataset)
        cifar10.cifar10_train()
        print('Successfully!')
        return 0
if __name__ == '__main__':
    sys.exit(main())
