"""
@file: read_data.py
@author: wak
@date: 2024/2/3 21:14 
"""
import os.path

from Bio import SeqIO

cur_path = os.path.abspath(__file__)


def read(path, label):
    X, y = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            X.append(line.strip())
            y.append(label)
    return X, y


# Get original data in the datasets which includes the sequences and corresponding labels
# path: The path of the dateset that you want to use for training and testing
def get_original_data(path):
    train_path = os.path.join(os.path.dirname(cur_path), '..', 'data', path, 'Train.fasta')
    test_path = os.path.join(os.path.dirname(cur_path), '..', 'data', path, 'Test.fasta')
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


if __name__ == '__main__':
    train_data, test_data = get_original_data('Main')
    train_X, train_y = get_seq_label(train_data)
    test_X, test_y = get_seq_label(test_data)
    print(train_X)
    print('==================')
    print(test_X)
