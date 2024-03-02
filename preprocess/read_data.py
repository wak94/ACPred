"""
@file: read_data.py
@author: wak
@date: 2024/2/3 21:14 
"""
import os.path

import numpy as np
import torch
from Bio import SeqIO

cur_path = os.path.dirname(os.path.abspath(__file__))


# Get original data in the datasets which includes the sequences and corresponding labels
# path: The path of the dateset that you want to use for training and testing
def get_original_data(path):
    train_path = os.path.join(cur_path, '..', 'dataset', path, 'train.txt')
    test_path = os.path.join(cur_path, '..', 'dataset', path, 'test.txt')
    train_data = list(SeqIO.parse(train_path, format="fasta"))
    test_data = list(SeqIO.parse(test_path, format="fasta"))

    return train_data, test_data


def get_seq_label(data):
    X, y = [], []
    for x in data:
        seq = str(x.seq)
        label = int(x.id.split('|')[-1])
        X.append(seq)
        y.append(label)
    return X, y


def ordinal_encoding(seq):
    elements = 'ACDEFGHIKLMNPQRSTVWY'
    mapping = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 17, 'T': 17, 'W': 18, 'Y': 19, 'V': 20}
    v = [0] * 50
    for i, x in enumerate(seq):
        v[i] = mapping[x]
    return v


def read(path):
    train_data, test_data = get_original_data(path)
    train_seq, train_label = get_seq_label(train_data)
    test_seq, test_label = get_seq_label(test_data)

    train_X, test_X = [], []
    for x in train_seq:
        train_X.append(ordinal_encoding(x))

    for x in test_seq:
        test_X.append(ordinal_encoding(x))

    return torch.tensor(train_X), torch.tensor(train_label), torch.tensor(test_X), torch.tensor(test_label)


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = read('AntiACP2.0_Alternate')
    print(train_X.shape, train_y.shape)
    print(test_X.shape, test_y.shape)
