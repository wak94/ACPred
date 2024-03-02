"""
@file: config.py
@author: wak
@date: 2024/3/1 12:03 
"""
import argparse


def get_config():
    parse = argparse.ArgumentParser()
    parse.add_argument('-devicenum', type=int, default=0, help='device id')
    parse.add_argument('-dataset', type=str, default='Main', help='name of dataset directory')
    parse.add_argument('-batch-size', type=int, default=32, help='size of a batch')
    parse.add_argument('-lr', type=float, default=1e-4, help='learning rate')
    parse.add_argument('-epochs', type=int, default=200, help='num of epoch')

    config = parse.parse_args()
    return config
