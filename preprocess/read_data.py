"""
@file: read_data.py
@author: wak
@date: 2024/2/3 21:14 
"""
import os.path

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
    path = os.path.join(os.path.dirname(cur_path), '..', 'data', path)
    if path[-1] != '/':
        path += '/'

    train_pos = os.path.join(os.path.dirname(path), 'Internal', 'pos_train.txt')
    train_neg = os.path.join(os.path.dirname(path), 'Internal', 'neg_train.txt')
    test_pos = os.path.join(os.path.dirname(path), 'Validation', 'pos_test.txt')
    test_neg = os.path.join(os.path.dirname(path), 'Validation', 'neg_test.txt')

    X1, y1 = read(train_pos, 1)
    X2, y2 = read(train_neg, 0)
    X3, y3 = read(test_pos, 1)
    X4, y4 = read(test_neg, 0)

    X_train = X1 + X2
    y_train = y1 + y2
    X_test = X3 + X4
    y_test = y3 + y4

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_original_data('AntiCP_Main')
    print('======================train======================')
    for X, y in zip(X_train, y_train):
        print(X + ',' + str(y))
    print('======================test======================')
    for X, y in zip(X_test, y_test):
        print(X + ',' + str(y))
