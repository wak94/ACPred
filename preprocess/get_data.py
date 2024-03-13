"""
@file: get_data.py
@author: wak
@date: 2024/3/1 15:19 
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from Bio import SeqIO
from Bio.Align import substitution_matrices

from configration.config import get_config

cur_path = os.path.dirname(os.path.abspath(__file__))
device = get_config().device
dataset = get_config().dataset


# Create custom dataset
class MyDataSet(Data.Dataset):
    def __init__(self, seq, label):
        self.label = torch.tensor(label, device=device)
        self.bert = bert_encoding(seq)
        self.aaindex = aaindex_encoding(seq)
        # self.blosum = blosum62_encoding(seq)
        self.oe = ordinal_encoding(seq)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data = {'bert': self.bert[idx], 'oe': self.oe[idx], 'aa': self.aaindex[idx]}
        return data, self.label[idx]


class HybridDataset(Data.Dataset):
    def __init__(self, seq, label):
        self.seq = seq
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        seq, label = self.seq[idx], self.label[idx]
        seq2vec = json.load(open(os.path.join(cur_path, f'../seq2vec_{dataset}.emb')))
        data = seq2vec[seq]
        return torch.tensor(data), self.label[idx]


def ordinal_encoding(seq):
    seq2num = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20}
    datas = []
    for s in seq:
        v = [0] * 50
        for i, x in enumerate(s):
            v[i] = seq2num[x]
        datas.append(v)
    return torch.tensor(datas, device=device)


def blosum62():
    matrix = substitution_matrices.load('BLOSUM62')
    blosum62_matrix = np.array(matrix)[:20, :20]
    elements = 'ARNDCQEGHILKMFPSTWYV'
    substitution = {}
    for i in range(20):
        substitution[elements[i]] = blosum62_matrix[i, :].tolist()
    return substitution


def blosum62_encoding(seq):
    datas = []
    for x in seq:
        v = np.array([blosum62()[char] for char in x], dtype=np.float32)
        means = np.mean(v, axis=0).tolist()
        datas.append(means)
    return torch.tensor(datas, dtype=torch.float32, device=device)


def aaindex_encoding(seq):
    aa = pd.read_csv(os.path.join(cur_path, 'AAindex.txt'), sep='\t')
    aa = np.array(aa.iloc[:, 1:])
    index = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
             'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 17, 'T': 17, 'W': 18, 'Y': 19, 'V': 20}

    aa_dict = {}
    for k in index.keys():
        v = index[k]
        aa_dict[k] = aa[v - 1, :].tolist()
    datas = []
    for x in seq:
        f = [[0] * 531 for _ in range(50)]
        for i, y in enumerate(x):
            f[i] = aa_dict[y]
        datas.append(f)

    return torch.tensor(datas, dtype=torch.float32, device=device)


def bert_encoding(seq):
    path = os.path.join(cur_path, '..', f'seq2vec_{dataset}.emb')
    seq2vec = json.load(open(path))
    datas = []
    for x in seq:
        datas.append(seq2vec[x])
    return torch.tensor(datas, dtype=torch.float32, device=device)


def read_origin_data(dir_name):
    train_path = os.path.join(cur_path, '..', 'dataset', dir_name, 'train.txt')
    test_path = os.path.join(cur_path, '..', 'dataset', dir_name, 'test.txt')
    train_data = list(SeqIO.parse(train_path, format="fasta"))
    test_data = list(SeqIO.parse(test_path, format="fasta"))

    train_seq, train_label = [], []
    for record in train_data:
        seq = str(record.seq)
        label = int(record.id[-1])
        train_seq.append(seq)
        train_label.append(label)

    test_seq, test_label = [], []
    for record in test_data:
        seq = str(record.seq)
        label = int(record.id[-1])
        test_seq.append(seq)
        test_label.append(label)

    return train_seq, train_label, test_seq, test_label


def collate(batch):
    data1_ls = []
    data2_ls = []
    label1_ls = []
    label2_ls = []
    label_ls = []
    batch_size = len(batch)
    for i in range(batch_size // 2):
        data1, label1 = batch[i][0], batch[i][1]
        data2, label2 = batch[i + batch_size // 2][0], batch[i + batch_size // 2][1]

        label1_ls.append(label1.unsqueeze(0))
        label2_ls.append(label2.unsqueeze(0))
        label = (label1 ^ label2)
        data1_ls.append(data1.unsqueeze(0))
        data2_ls.append(data2.unsqueeze(0))
        label_ls.append(label.unsqueeze(0))
    data1 = torch.cat(data1_ls).to(device)
    data2 = torch.cat(data2_ls).to(device)
    label = torch.cat(label_ls).to(device)
    label1 = torch.cat(label1_ls).to(device)
    label2 = torch.cat(label2_ls).to(device)
    return data1, data2, label, label1, label2


def collate1(batch):
    bert1_ls = []
    oe1_ls = []
    aa1_ls = []
    bert2_ls = []
    oe2_ls = []
    aa2_ls = []
    label1_ls = []
    label2_ls = []
    label_ls = []
    batch_size = len(batch)
    for i in range(batch_size // 2):
        data1, label1 = batch[i][0], batch[i][1]
        data2, label2 = batch[i + batch_size // 2][0], batch[i + batch_size // 2][1]

        label1_ls.append(label1.unsqueeze(0))
        label2_ls.append(label2.unsqueeze(0))
        label = (label1 ^ label2)
        bert1_ls.append(data1['bert'].unsqueeze(0))
        oe1_ls.append(data1['oe'].unsqueeze(0))
        aa1_ls.append(data1['aa'].unsqueeze(0))
        bert2_ls.append(data2['bert'].unsqueeze(0))
        oe2_ls.append(data2['oe'].unsqueeze(0))
        aa2_ls.append(data2['aa'].unsqueeze(0))
        label_ls.append(label.unsqueeze(0))
    data1 = {'bert': torch.cat(bert1_ls).to(device), 'oe': torch.cat(oe1_ls).to(device),
             'aa': torch.cat(aa1_ls).to(device)}
    data2 = {'bert': torch.cat(bert2_ls).to(device), 'oe': torch.cat(oe2_ls).to(device),
             'aa': torch.cat(aa2_ls).to(device)}
    label = torch.cat(label_ls).to(device)
    label1 = torch.cat(label1_ls).to(device)
    label2 = torch.cat(label2_ls).to(device)
    return data1, data2, label, label1, label2


if __name__ == '__main__':
    # cfg = get_config()
    # dataset = cfg.dataset
    # train_data, test_data = get_original_data(f'AntiACP2.0_{dataset}')
    # train_seq, train_label = get_seq_label(train_data)
    # train_dataset = HybridDataset(train_seq, train_label)
    # train_dataloader = Data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    # for X, y in train_dataloader:
    #     print(X)
    #     print(X.shape)
    # print(blosum62())
    # b = blosum62_encoding('YDDFAELGSTETTGFSFQNVFQLAGVPKDFIASPRSPVQELNQKQENREN')
    # print(b)
    # print(b.size())
    # X1, y1, X2, y2 = read_origin_data('AntiACP2.0_Main')
    # print(X1)
    # print(y1)
    # print(X2)
    # print(y2)
    # read_origin_data('AntiACP2.0_Alternate')
    aaindex_encoding('AA')
