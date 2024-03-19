"""
@file: main.py
@author: wak
@date: 2024/2/4 10:17 
"""
import warnings

# 取消所有警告
warnings.filterwarnings("ignore")

import time

import numpy as np
import torch
from termcolor import colored

import configration.config as cf
from preprocess.get_data import MyDataSet, HybridDataset, read_origin_data, collate1
from models.models import NewModel, CNN, ContrastiveLoss, MyModel3, MyModel4, BFD, MyModel5, MambaTest, MyModel6

import torch.nn as nn
from torch.utils.data import DataLoader

from utils import get_prediction, evaluate_accuracy


def main(args, model, name):
    batch_size = args.batch_size
    dataset = args.dataset
    lr = args.lr
    device = args.device
    epochs = args.epochs
    train_seq, train_label, test_seq, test_label = read_origin_data(f'AntiACP2.0_{dataset}')
    train_dataset = MyDataSet(train_seq, train_label)
    test_dataset = MyDataSet(test_seq, test_label)
    train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   drop_last=True), DataLoader(test_dataset, batch_size=batch_size,
                                                                               shuffle=True, drop_last=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion_model = nn.CrossEntropyLoss(reduction='sum').to(device)
    best_acc = 0.0
    best_metrics = []
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        loss_ls = []
        for data, label in train_dataloader:
            output = model.trainModel(data)
            loss = criterion_model(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_ls.append(loss.item())

        model.eval()
        with torch.no_grad():
            train_acc = evaluate_accuracy(train_dataloader, model)
            metrics, roc_data, prc_data = get_prediction(test_dataloader, model)
            metrics_list = metrics.tolist()
            test_acc = metrics_list[0]

        results = f"epoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
        results += f'\ttrain_acc: {train_acc:.4f}, test_acc: {colored(test_acc, "red")}, time: {time.time() - t0:.2f}'
        print(results)

        if test_acc > best_acc:
            best_acc = test_acc
            best_metrics = metrics_list
            torch.save({"best_acc": best_acc, "metric": metrics, "model": model.state_dict()},
                       f'./result/{dataset}/{name}.pl')
            print(f"best_acc: {best_acc},metric:{metrics_list}")
    print(f"best_acc: {best_acc},metric:{best_metrics}")


def main_contrastive(args, model, name):
    batch_size = args.batch_size
    dataset = args.dataset
    lr = args.lr
    device = args.device

    train_seq, train_label, test_seq, test_label = read_origin_data(f'AntiACP2.0_{dataset}')
    train_dataset = MyDataSet(train_seq, train_label)
    test_dataset = MyDataSet(test_seq, test_label)
    train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   drop_last=True), DataLoader(test_dataset, batch_size=batch_size,
                                                                               shuffle=True, drop_last=True)
    contrastive_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                        collate_fn=collate1)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion_model = nn.CrossEntropyLoss().to(device)
    criterion = ContrastiveLoss().to(device)
    epochs = args.epochs
    best_acc = 0.0
    best_metrics = []
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        loss_ls = []
        contrastive_loss_ls = []
        model_loss_ls = []
        for data1, data2, label, label1, label2 in contrastive_dataloader:
            output1 = model(data1)
            output2 = model(data2)
            output3 = model.trainModel(data1)
            output4 = model.trainModel(data2)

            contrastive_loss = criterion(output1, output2, label)
            model_loss1 = criterion_model(output3, label1)
            model_loss2 = criterion_model(output4, label2)
            loss = contrastive_loss + model_loss1 + model_loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ls.append(loss.item())
            contrastive_loss_ls.append(contrastive_loss.item())
            model_loss_ls.append((model_loss1 + model_loss2).item())

        model.eval()
        with torch.no_grad():
            train_acc = evaluate_accuracy(train_dataloader, model)
            metrics, roc_data, prc_data = get_prediction(test_dataloader, model)
            metrics_list = metrics.tolist()
            test_acc = metrics_list[0]

        results = f"epoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}, contrastive_loss: {np.mean(contrastive_loss_ls):.5f}, model_loss: {np.mean(model_loss_ls):.5f}\n"
        results += f'\ttrain_acc: {train_acc:.4f}, test_acc: {colored(test_acc, "red")}, time: {time.time() - t0:.2f}'
        print(results)

        if test_acc > best_acc:
            best_acc = test_acc
            best_metrics = metrics_list
            torch.save({"best_acc": best_acc, "metric": metrics, "model": model.state_dict()},
                       f'./result/{dataset}/{name}.pl')
            print(f"best_acc: {best_acc},metric:{metrics_list}")
    print(f"best_acc: {best_acc},metric:{best_metrics}")


if __name__ == '__main__':
    args = cf.get_config()

    # main_contrastive(args, model, 'basic_CL')
    # for i in range(10):
    #     model = MyModel5()
    #     main_contrastive(args, model, f'basic_PM_CL_{i + 1}')
    # main(args, model, 'basic_PM')
    # for i in range(16):
    #     print(f'd_model=512, d_state={i + 1}, d_conv=2, expand=8')
    #     model = MambaTest(d_state=i + 1)
    #     main(args, model, f'mamba_mlp:expand={i + 1}')
    #     print('---------------------------------------------------------')
    # model = MambaTest(d_conv=2)
    # main(args, model, 'mamba_mlp:d_conv=2')
    # print('---------------------------------------------------------')
    # model = MambaTest(d_state=16, d_conv=4, expand=2)
    # main(args, model, 'mamba_mlp')
    for i in range(10):
        print(f'第{i + 1}次')
        model = MyModel6()
        main_contrastive(args, model, f'model6:no_PM_{i + 1}')
