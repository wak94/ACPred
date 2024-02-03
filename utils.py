"""
@file: utils.py
@author: wak
@date: 2024/2/3 22:16 
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import os.path

cur_path = os.path.abspath(__file__)


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


def caculate_metric(pred_y, labels, pred_prob):
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

    metric = torch.tensor([ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC])

    # ROC(fpr, tpr, AUC)
    # PRC(recall, precision, AP)
    roc_data = [fpr, tpr, AUC]
    prc_data = [recall, precision, AP]
    return metric, roc_data, prc_data
