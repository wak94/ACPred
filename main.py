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
from models.models import NewModel
from preprocess.get_data import get_dataloader
import torch.nn as nn

from utils import get_prediction, evaluate_accuracy


def main(args):
    batch_size = args.batch_size
    dataset = args.dataset
    lr = args.lr
    devicenum = args.devicenum
    device = torch.device('cuda', devicenum)

    train_dataloader, test_dataloader = get_dataloader('AntiACP2.0_' + dataset, batch_size)

    net = NewModel().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    criterion_model = nn.CrossEntropyLoss(reduction='sum').to(device)
    epochs = args.epochs
    best_acc = 0.0
    for epoch in range(epochs):
        t0 = time.time()
        net.train()
        loss_ls = []
        for data, label in train_dataloader:
            data, label = data.to(device), label.to(device)
            output = net(data)
            loss = criterion_model(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_ls.append(loss.item())

        net.eval()
        with torch.no_grad():
            train_acc = evaluate_accuracy(train_dataloader, net)
            metrics, roc_data, prc_data = get_prediction(test_dataloader, net)
            metrics_list = metrics.tolist()
            test_acc = metrics_list[0]

        results = f"epoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
        results += f'\ttrain_acc: {train_acc:.4f}, test_acc: {colored(test_acc, "red")}, time: {time.time() - t0:.2f}'
        print(results)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"best_acc": best_acc, "metric": metrics, "model": net.state_dict()},
                       f'./result/{dataset}/res.pl')
            print(f"best_acc: {best_acc},metric:{metrics_list}")


if __name__ == '__main__':
    args = cf.get_config()
    main(args)
