import torch.optim as optim
import json
import torch.nn as nn
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class model_train():
    def __init__(self, model, model_name, train_loader, test_loader, dim=1024, epoch_number=150, lr=1e-3, device='cuda:0', output_path = './result', dataset_name = 'mnist', log_dir = './result_sgd_second/'):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.dim = dim
        self.epoch_number = epoch_number
        self.lr = lr
        self.device = device
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.fileName = str(self.dim) + str(self.model_name) + str(self.lr) + str(dataset_name)
        self.writer = SummaryWriter(log_dir= (log_dir + self.fileName))
    def epoch(self, loader, opt=None):
        total_loss, total_err = 0., 0.
        self.model.eval() if opt is None else self.model.train()

        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            yp = self.model(X)
            loss = nn.CrossEntropyLoss()(yp, y)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()
                # lr_scheduler.step()
            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item() * X.shape[0]

        return total_err / len(loader.dataset), total_loss / len(loader.dataset)

    def cifar10_train(self):
        # opt = optim.Adam(self.model.parameters(), lr=self.lr)
        opt = optim.SGD(self.model.parameters(),
                                    lr=self.lr,
                                    momentum=0)
        print("# Parmeters: ", sum(a.numel() for a in self.model.parameters()))

        # create a summary writer with automatically generated folder name.
        epochs = self.epoch_number
        max_epochs = 500
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs * len(train_loader), eta_min= lr)
        Training_acc_list = []
        Test_acc_list = []
        root = self.output_path
        if not os.path.isdir(root):
            os.makedirs(root)
        Train_json_file = str(self.dim) + "_train_" + str(self.model_name) + "_cifar10.json"
        Test_json_file = str(self.dim) + "_test_" + str(self.model_name) + "_cifar10.json"
        Train_json_file = os.path.join(root, Train_json_file)
        Test_json_file = os.path.join(root, Test_json_file)
        for i in range(epochs):
            # Training_err, Training_loss = epoch(train_loader, model, opt, scheduler)
            Training_err, Training_loss = self.epoch(loader=self.train_loader, opt = opt)
            Test_err, Test_loss = self.epoch(loader=self.test_loader, opt = None)
            Training_acc_list.append(1 - Training_err)
            Test_acc_list.append(1 - Test_err)
            self.writer.add_scalar('Loss/train', Training_loss, i)
            self.writer.add_scalar('Loss/test', Test_loss, i)
            self.writer.add_scalar('Accuracy/Train', 1 - Training_err, i)
            self.writer.add_scalar('Accuracy/Test', 1 - Test_err, i)
            print(
                f'DIM:{self.dim},Epoch [{i + 1}/{epochs}]: Train loss: {Training_loss:.4f},Train accuracy: {(1 - Training_err):.4f}, Valid loss: {Test_loss:.4f}, Valid accuracy: {(1 - Test_err):.4f}'
            )

            with open(Train_json_file, "w") as f:
                json.dump(Training_acc_list, f)
            with open(Test_json_file, "w") as f:
                json.dump(Test_acc_list, f)

    def mnist_train(self):
        lr = self.lr
        epochs = self.epoch_number
        # optimizer = optim.Adam(self.model.parameters(), lr=lr)
        optimizer = optim.SGD(self.model.parameters(),
                                    lr=lr,
                                    momentum=0)
        criterion = nn.CrossEntropyLoss()
        target_folder = self.output_path
        if not os.path.isdir(target_folder):
            os.makedirs(target_folder)
        Train_json_file = str(self.dim) + "_train_" + str(self.model_name) + '_' + str(self.dataset_name)+'.json'
        Test_json_file = str(self.dim) + "_test_" + str(self.model_name) + '_' + str(self.dataset_name)+'.json'
        Train_json_file = os.path.join(target_folder, Train_json_file)
        Test_json_file = os.path.join(target_folder, Test_json_file)
        train_json = []
        test_json = []
        # --------------------------Trainning and Validation-----------------------------------
        for epoch in range(epochs):
            self.model.train()
            loss_record = []
            train_accuracy_record = []
            for train_data, train_label in self.train_loader:
                optimizer.zero_grad()
                train_data, train_label = train_data.to(self.device), train_label.to(
                    self.device)
                pred = self.model(train_data)
                loss = criterion(pred, train_label)
                loss.backward()
                optimizer.step()
                loss_record.append(loss.item())
                # accuracy
                _, index = pred.data.cpu().topk(1, dim=1)
                _, index_label = train_label.data.cpu().topk(1, dim=1)
                accuracy_batch = np.sum(
                    (index.squeeze(dim=1) == index_label.squeeze(dim=1)).numpy())
                accuracy_batch = accuracy_batch / len(train_label)
                train_accuracy_record.append(accuracy_batch)

            train_loss = sum(loss_record) / len(loss_record)
            train_accuracy = sum(train_accuracy_record) / len(train_accuracy_record)
            train_json.append(train_accuracy)

            # validation
            self.model.eval()
            loss_record = []
            test_accuracy_record = []
            for val_data, val_label in self.test_loader:
                val_data, val_label = val_data.to(self.device), val_label.to(self.device)
                with torch.no_grad():
                    pred = self.model(val_data)
                    loss = criterion(pred, val_label)
                loss_record.append(loss.item())
                # accuracy
                _, index = pred.data.cpu().topk(1, dim=1)
                _, index_label = val_label.data.cpu().topk(1, dim=1)
                accuracy_batch = np.sum(
                    (index.squeeze(dim=1) == index_label.squeeze(dim=1)).numpy())
                accuracy_batch = accuracy_batch / len(val_label)
                test_accuracy_record.append(accuracy_batch)

            val_loss = sum(loss_record) / len(loss_record)
            val_accuracy = sum(test_accuracy_record) / len(test_accuracy_record)
            test_json.append(val_accuracy)
            with open(Train_json_file, "w") as f:
                json.dump(train_json, f)
            with open(Test_json_file, "w") as f:
                json.dump(test_json, f)
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/test', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
            self.writer.add_scalar('Accuracy/Test', val_accuracy, epoch)
            print(
                f'DIM={self.dim},Epoch [{epoch + 1}/{epochs}]:Train loss: {train_loss:.4f},Train accuracy: {train_accuracy:.4f}, Valid loss: {val_loss:.4f}, Valid accuracy: {val_accuracy:.4f}'
            )
