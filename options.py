#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # experiment arguments
    parser.add_argument('--mode', type=str, default='fed_GTG_SV'
                        , help="plain, DP, or Paillier,fed_GTG_SV")

    # federated arguments
    parser.add_argument('--rounds', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--num_select_users', type=int, default=4, help="number of selected users: k")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=100, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=100, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # other arguments
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--corruption_type', type=str, default='uniform', help="uniform,flip1,flip2")
    parser.add_argument('--corruption_ratio', type=float, default='0.6', help="0.1,0.2,0.3")

    args = parser.parse_args()
    return args
