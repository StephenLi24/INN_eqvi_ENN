import os
import numpy as np
import torchvision.datasets as dset
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import utils

def fashionmnist_prepared(dataset_path):
    cs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    K = len(cs)
    res = my_dataset_custome('fashion_mnist',
                             T_train=50000,
                             T_test=8000,
                             cs=cs,
                             selected_target=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dataset_path= dataset_path)
    dataset_train, dataset_test = res[0], res[1]
    'to be done here new net configuration here just binaryzero and binary last'
    tau_zero = np.sqrt(utils.estim_tau_tensor(dataset_train.X))
    print("tau_zero=",tau_zero)
    tau = utils.get_tau(tau_zero)
    # print("tau=",tau)
    dataset_train.Y = F.one_hot(torch.tensor(dataset_train.Y).long(), 10)
    dataset_train.Y = dataset_train.Y.float()
    # print(mnist_train.Y.dtype)
    dataset_test.Y = F.one_hot(torch.tensor(dataset_test.Y).long(), 10)
    dataset_test.Y = dataset_test.Y.float()
    return dataset_train, dataset_test, tau

def mnist_prepared(dataset_path):
    cs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    K = len(cs)


    res = my_dataset_custome('MNIST',
                             T_train=50000,
                             T_test=8000,
                             cs=cs,
                             selected_target=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dataset_path = dataset_path)
    dataset_train, dataset_test = res[0], res[1]
    'to be done here new net configuration here just binaryzero and binary last'
    tau_zero = np.sqrt(utils.estim_tau_tensor(dataset_train.X))
    # print("tau_zero=",tau_zero)
    tau = utils.get_tau(tau_zero)
    # print("tau=",tau)
    dataset_train.Y = F.one_hot(torch.tensor(dataset_train.Y).long(), 10)
    dataset_train.Y = dataset_train.Y.float()
    # print(dataset_train.Y.dtype)
    dataset_test.Y = F.one_hot(torch.tensor(dataset_test.Y).long(), 10)
    dataset_test.Y = dataset_test.Y.float()
    return dataset_train, dataset_test, tau
def gen_data(testcase,
             selected_target=[6, 8],
             T=None,
             p=None,
             cs=None,
             means=None,
             covs=None,
             mode='train',
             dataset_path = '/data'):
    '''Generate GMM data from existing datasets or self sampling datasets.

    Arguments:
        testcase -- 'MNIST'/'FashionMnist‘
        selected_traget -- list[xx, xx], only used for testcase=='MNIST'/'FashionMnist‘'
    Returns:
        X -- data
        Omega -- data - means
        y -- targets
        means -- means
        covs -- covs
        K -- number of class
        p -- dimension of data
        T -- number of data

    '''
    root = dataset_path
    if not os.path.isdir(root):
        os.makedirs(root)
    if testcase == 'MNIST':
        if mode == 'train':
            mnist = dset.MNIST(root=os.path.join(root, 'train'),
                               train=True,
                               download=True)
        else:
            mnist = dset.MNIST(root=os.path.join(root, 'test'),
                               train=False,
                               download=True)
        data, labels = mnist.data.view(mnist.data.shape[0], -1), mnist.targets

        # feel free to choose the number you like
        selected_target = selected_target
        p = 784
        K = len(selected_target)

        # get the whole set of selected number
        data_full = []
        data_full_matrix = np.array([]).reshape(p, 0)
        ind = 0
        for i in selected_target:
            locate_target_train = np.where(labels == i)[0]
            data_full.append(data[locate_target_train].T)
            data_full_matrix = np.concatenate(
                (data_full_matrix, data[locate_target_train].T), axis=1)
            ind += 1

        # recentering and normalization
        T_full = data_full_matrix.shape[1]
        mean_selected_data = np.mean(data_full_matrix, axis=1).reshape(p, 1)
        norm2_selected_data = np.sum(
            (data_full_matrix -
             np.mean(data_full_matrix, axis=1).reshape(p, 1))**2,
            (0, 1)) / T_full
        for i in range(K):
            data_full[i] = data_full[i] - mean_selected_data
            data_full[i] = data_full[i] * np.sqrt(p) / np.sqrt(
                norm2_selected_data)

        # get the statistics of MNIST data
        means = []
        covs = []
        for i in range(K):
            data_tmp = data_full[i]
            T_tmp = data_tmp.shape[1]
            means.append(np.mean(data_tmp.numpy(), axis=1).reshape(p, 1))
            covs.append((data_tmp @ (data_tmp.T) / T_tmp -
                         means[i] @ (means[i].T)).reshape(p, p))

        # data for train

        X = np.array([]).reshape(p, 0)
        Omega = np.array([]).reshape(p, 0)
        y = []

        ind = 0
        for i in range(K):
            data_tmp = data_full[i]
            X = np.concatenate((X, data_tmp[:, range(int(cs[ind] * T))]),
                               axis=1)
            Omega = np.concatenate(
                (Omega, data_tmp[:, range(int(cs[ind] * T))] -
                 np.outer(means[ind], np.ones((1, int(T * cs[ind]))))),
                axis=1)
            y = np.concatenate((y, ind * np.ones(int(T * cs[ind]))))
            ind += 1

        X = X / np.sqrt(p)
        Omega = Omega / np.sqrt(p)

    elif testcase == 'fashion_mnist':
        if mode == 'train':
            mnist = dset.FashionMNIST(root=os.path.join(root, 'train'),
                               train=True,
                               download=True)
        else:
            mnist = dset.FashionMNIST(root=os.path.join(root, 'test'),
                               train=False,
                               download=True)
        data, labels = mnist.data.view(mnist.data.shape[0], -1), mnist.targets

        # feel free to choose the number you like :)
        selected_target = selected_target
        p = 784
        K = len(selected_target)

        # get the whole set of selected number
        data_full = []
        data_full_matrix = np.array([]).reshape(p, 0)
        ind = 0
        for i in selected_target:
            locate_target_train = np.where(labels == i)[0]
            data_full.append(data[locate_target_train].T)
            data_full_matrix = np.concatenate(
                (data_full_matrix, data[locate_target_train].T), axis=1)
            ind += 1

        # recentering and normalization
        T_full = data_full_matrix.shape[1]
        mean_selected_data = np.mean(data_full_matrix, axis=1).reshape(p, 1)
        norm2_selected_data = np.sum(
            (data_full_matrix -
             np.mean(data_full_matrix, axis=1).reshape(p, 1)) ** 2,
            (0, 1)) / T_full
        for i in range(K):
            data_full[i] = data_full[i] - mean_selected_data
            data_full[i] = data_full[i] * np.sqrt(p) / np.sqrt(
                norm2_selected_data)

        # get the statistics of MNIST data
        means = []
        covs = []
        for i in range(K):
            data_tmp = data_full[i]
            T_tmp = data_tmp.shape[1]
            means.append(np.mean(data_tmp.numpy(), axis=1).reshape(p, 1))
            covs.append((data_tmp @ (data_tmp.T) / T_tmp -
                         means[i] @ (means[i].T)).reshape(p, p))

        # data for train

        X = np.array([]).reshape(p, 0)
        Omega = np.array([]).reshape(p, 0)
        y = []

        ind = 0
        for i in range(K):
            data_tmp = data_full[i]
            X = np.concatenate((X, data_tmp[:, range(int(cs[ind] * T))]),
                               axis=1)
            Omega = np.concatenate(
                (Omega, data_tmp[:, range(int(cs[ind] * T))] -
                 np.outer(means[ind], np.ones((1, int(T * cs[ind]))))),
                axis=1)
            y = np.concatenate((y, ind * np.ones(int(T * cs[ind]))))
            ind += 1

        X = X / np.sqrt(p)
        Omega = Omega / np.sqrt(p)

    return X, Omega, y, means, covs, K, p, T

def my_dataset_custome(testcase,
                       selected_target=[6, 8],
                       T_train=None,
                       T_test=None,
                       p=None,
                       cs=None,
                       dataset_path = './data',
                       means=None,
                       covs=None):
    '''Generate GMM data generate the data matrix with respect to different test cases.

    Arguments:
        testcase -- 'MNIST'/'fashion_mnist'
        selected_traget -- list[xx, xx], only used for testcase=='MNIST'/'CIFAR10'
        T_train -- len of train datasets
        T_test -- len of test datasets
        p -- dimension of data, only used for get tau
        cs -- list[0.xx, 0.xx], ratio for diff classes, len(cs) is number of class of the dataset
        means -- matrix, means for diff classes, only used for testcase=='iid'/'means'/'vars'/'mixed'
        covs -- matrix, covs for diff classes, only used for testcase=='iid'/'means'/'vars'/'mixed'
    Returns:
        train_dataset -- train_dataset(packed as torch.utils.data.Dataset)
        test_dataset -- test_dataset(packed as torch.utils.data.Dataset)
        means -- means for different classes
        covs -- covs for different classes
        K -- number of class
        p -- dimension of data
        train_T -- number of train data
        test_T -- number of test data
        Omega_train -- train_data - means
        Omega_test -- test_data - means
    '''
    if testcase == 'MNIST':
        # get train and test dataset and then packed as torch.Dataset
        X_train, Omega_train, Y_train, means, covs, K, p, train_T = gen_data(
            testcase,
            selected_target=selected_target,
            T=T_train,
            cs=cs,
            mode='train',
            dataset_path = dataset_path)
        train_dataset = my_dataset(X_train, Y_train)

        X_test, Omega_test, Y_test, _, _, _, _, test_T = gen_data(
            testcase,
            selected_target=selected_target,
            T=T_test,
            cs=cs,
            mode='test',
            dataset_path = dataset_path)
        test_dataset = my_dataset(X_test, Y_test)
    elif testcase == 'fashion_mnist':
        X_train, Omega_train, Y_train, means, covs, K, p, train_T = gen_data(
            testcase,
            selected_target=selected_target,
            T=T_train,
            cs=cs,
            mode='train',
            dataset_path = dataset_path)
        train_dataset = my_dataset(X_train, Y_train)

        X_test, Omega_test, Y_test, _, _, _, _, test_T = gen_data(
            testcase,
            selected_target=selected_target,
            T=T_test,
            cs=cs,
            mode='test',
            dataset_path = dataset_path)
        test_dataset = my_dataset(X_test, Y_test)
    return train_dataset, test_dataset, means, covs, K, p, train_T, test_T, Omega_train, Omega_test

class my_dataset(Dataset):
    '''Packed mnist,fashionMinst datasets to torch.utils.data.Dataset.
    '''
    def __init__(self, X, Y) -> None:
        super().__init__()
        self.X, self.Y = X.T, Y

    def __getitem__(self, idx):
        if self.Y.ndim == 1:
            return self.X[idx, :], self.Y[idx]
        else:
            return self.X[idx, :], self.Y[idx, :]

    def __len__(self):
        return self.X.shape[0]
class Feature_Dataset(Dataset):

    def __init__(self, X, Y) -> None:
        """Packed Features extracted from VGG19 feature layer which will
        be further concatenated with random feature layers and classification layers.
        Arguments:
            X -- Features of data
            Y -- Labels of data
        """
        super().__init__()
        self.X, self.Y = X, Y

    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx]

    def __len__(self):
        return self.X.shape[0]


cfg = {'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
       'lr_epoch': (75, 120)}


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers


class VGG(nn.Module):
    def __init__(self, base):
        super(VGG, self).__init__()
        # self.features = nn.ModuleList(base)
        self.features = nn.Sequential(*base)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 10, bias=False)
        )


        for m in self.modules():
            # if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512)
        x = self.classifier(x)
        return x

def vgg_cifa10_feature(pretrained_checkpoints_path, data_path):
    if pretrained_checkpoints_path == 'default':
        model_vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT)
        new_classifier = torch.nn.Sequential(*list(model_vgg.children())[-1][:6])
        model_vgg.classifier = new_classifier

    else:
        ##use pretrained VGG16 trained by cifar10
        model_vgg = VGG(vgg(cfg['VGG16'], 3, False))
        # print(model_vgg)
        new_classifier = torch.nn.Sequential(*list(model_vgg.children())[-1][:6])
        model_vgg.classifier = new_classifier
        model_dict = model_vgg.state_dict()
        pretrained_dict = torch.load(pretrained_checkpoints_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_vgg.load_state_dict(pretrained_dict)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    train_data = torchvision.datasets.CIFAR10(root=data_path,
                                              train=True,
                                              download=True,
                                              transform=transform_train)
    trainloader = torch.utils.data.DataLoader(train_data,
                                              batch_size=50000,
                                              shuffle=False)

    test_data = torchvision.datasets.CIFAR10(root=data_path,
                                             train=False,
                                             download=True,
                                             transform=transform_test)
    testloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=10000,
                                             shuffle=False)

    classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog',
               'Horse', 'Ship', 'Truck')

    # ------------------------------Feature Extration----------------------------------
    train_data, train_label = next(iter(trainloader))
    test_data, test_label = next(iter(testloader))
    with torch.no_grad():
        feature_train = model_vgg(train_data)
        feature_train = feature_train.view(feature_train.shape[0], -1)
        feature_test = model_vgg(test_data)
        feature_test = feature_test.view(feature_test.shape[0], -1)
        p = feature_train.shape[1]
        N = feature_train.shape[0]
        mean_selected_data = torch.mean(feature_train, dim=0)
        norm2_selected_data = torch.sum(
            (feature_train - mean_selected_data) ** 2, (0, 1)) / N
        feature_train = feature_train - mean_selected_data
        feature_train = feature_train / np.sqrt(norm2_selected_data)

        p = feature_test.shape[1]
        N = feature_test.shape[0]
        mean_selected_data = torch.mean(feature_test, dim=0)
        norm2_selected_data = torch.sum((feature_test - mean_selected_data) ** 2,
                                        (0, 1)) / N
        feature_test = feature_test - mean_selected_data
        feature_test = feature_test / np.sqrt(norm2_selected_data)

    # dataset for future training testing
    feature_train_dataset = Feature_Dataset(feature_train, train_label)
    feature_test_dataset = Feature_Dataset(feature_test, test_label)

    tau_zero = torch.sqrt(
        torch.mean(torch.diag(torch.mm(feature_train,
                                       feature_train.t())))).detach().numpy()
    print(tau_zero)
    tau = utils.get_tau(tau_zero)
    print("tau=",tau)
    return  feature_train_dataset, feature_test_dataset, tau

