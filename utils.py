"""
@file: utils.py
@author: wak
@date: 2024/2/3 22:16 
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import os.path
import configration.config as cf

cur_path = os.path.abspath(__file__)
devicenum = cf.get_config().devicenum
device = torch.device('cuda', devicenum)


def ROC(fpr, tpr, AUC, path, name):
    path = os.path.join(os.path.dirname(cur_path), 'result', path, name + '.png')
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % AUC)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(path)


def PRC(recall, precision, AP, path, name):
    path = os.path.join(os.path.dirname(cur_path), 'result', path, name + '.png')
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % AP)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig(path)


def calculate_metric(pred_y, labels, pred_prob):
    test_num = len(labels)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for index in range(test_num):
        if int(labels[index]) == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    ACC = float(tp + tn) / test_num

    # precision
    if tp + fp == 0:
        Precision = 0
    else:
        Precision = float(tp) / (tp + fp)

    # SE
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # SP
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # F1-score
    if Recall + Precision == 0:
        F1 = 0
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)

    # ROC and AUC
    labels = list(map(int, labels))
    pred_prob = list(map(float, pred_prob))
    fpr, tpr, thresholds = roc_curve(labels, pred_prob, pos_label=1)  # 默认1就是阳性
    AUC = auc(fpr, tpr)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(labels, pred_prob, pos_label=1)
    AP = average_precision_score(labels, pred_prob, average='macro', pos_label=1, sample_weight=None)
    metric = [ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC]
    metric = [round(x, 5) for x in metric]
    metric = torch.tensor(metric)

    # ROC(fpr, tpr, AUC)
    # PRC(recall, precision, AP)
    roc_data = [fpr, tpr, AUC]
    prc_data = [recall, precision, AP]
    return metric, roc_data, prc_data


def model_output(data, net):
    return net.trainModel(data['bert'], data['oe'], data['aa'])


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for data, label in data_iter:
        outputs = net.trainModel(data)
        acc_sum += (outputs.argmax(dim=1) == label).float().sum().item()
        n += label.shape[0]
    return acc_sum / n


def get_prediction(data_iter, net):
    y_pred, y_true = [], []
    outputs = []
    for x, y in data_iter:
        output = net.trainModel(x)
        outputs.append(output)
        y_pred.append(output.argmax(dim=1).cpu().numpy())
        y_true.append(y.cpu().numpy())
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    y_pred = y_pred.reshape(-1, 1)
    y_true = y_true.reshape(-1, 1)

    df1 = pd.DataFrame(y_pred, columns=['y_pred'])
    df2 = pd.DataFrame(y_true, columns=['y_true'])
    df4 = pd.concat([df1, df2], axis=1)

    outputs = torch.cat(outputs, dim=0)
    pred_prob = outputs[:, 1]
    pred_prob = np.array(pred_prob.cpu().detach().numpy())
    pred_prob = pred_prob.reshape(-1)
    df3 = pd.DataFrame(pred_prob, columns=['pred_prob'])
    df5 = pd.concat([df4, df3], axis=1)
    y_true1 = df5['y_true']
    y_pred1 = df5['y_pred']
    pred_prob1 = df5['pred_prob']

    metrics, roc_data, prc_data = calculate_metric(y_pred1, y_true1, pred_prob1)
    return metrics, roc_data, prc_data
