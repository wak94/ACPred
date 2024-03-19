"""
@file: config.py
@author: wak
@date: 2024/3/1 12:03 
"""
import argparse

import torch


def get_config():
    parse = argparse.ArgumentParser()
    parse.add_argument('-devicenum', type=int, default=1, help='device id')
    parse.add_argument('-dataset', type=str, default='Main', help='name of dataset directory')
    parse.add_argument('-batch-size', type=int, default=64, help='size of a batch')
    parse.add_argument('-lr', type=float, default=0.0001, help='learning rate')
    parse.add_argument('-epochs', type=int, default=300, help='num of epoch')

    config = parse.parse_args()
    config.device = torch.device('cuda', config.devicenum)
    return config
