"""
@file: get_data.py
@author: wak
@date: 2024/3/1 15:19 
"""
import torch.utils.data as Data

from preprocess.read_data import read


# Create custom dataset
class MyDataSet(Data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


# Get dataloader  corresponding to the dataset
# path The name of dataset
# batch_size number of data in a batch
# return dataloaders for training set and testing set
def get_dataloader(path, batch_size):
    train_X, train_y, test_X, test_y = read(path)
    train_dataset = MyDataSet(train_X, train_y)
    test_dataset = MyDataSet(test_X, test_y)

    train_dataloader = Data.DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    test_dataloader = Data.DataLoader(test_dataset, batch_size, shuffle=True, drop_last=True)

    return train_dataloader, test_dataloader

