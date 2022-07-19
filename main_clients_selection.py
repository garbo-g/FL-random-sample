import os.path

import numpy as np
# from numpy.lib.function_base import gradient
import torch
from torchvision import datasets, transforms

from options import args_parser
from client import *
from server import *
import copy
import matplotlib.pyplot as plt
import time
import random
from resnet import resnet32
def uniform_corruption(corruption_ratio, num_classes):
    eye = np.eye(num_classes)
    noise = np.full((num_classes, num_classes), 1/num_classes)
    corruption_matrix = eye * (1 - corruption_ratio) + noise * corruption_ratio
    return corruption_matrix


def flip1_corruption(corruption_ratio, num_classes):
    corruption_matrix = np.eye(num_classes) * (1 - corruption_ratio)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        corruption_matrix[i][np.random.choice(row_indices[row_indices != i])] = corruption_ratio
    return corruption_matrix


def flip2_corruption(corruption_ratio, num_classes):
    corruption_matrix = np.eye(num_classes) * (1 - corruption_ratio)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        corruption_matrix[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_ratio / 2
    return corruption_matrix

def load_dataset():
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.CIFAR10('./data/CIFAR10/', train = True, download = True, transform = trans_mnist)
    dataset_test = datasets.CIFAR10('./data/CIFAR10/', train = False, download = True, transform = trans_mnist)
    # dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
    # dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
    return dataset_train, dataset_test

def create_client_server():
    num_items = int(len(dataset_train) / args.num_users)
    clients, all_idxs = [], [i for i in range(len(dataset_train))]
    net_glob = resnet32(args.num_classes).to(args.device)
    # net_glob = CNNMnist(args=args).to(args.device)

    #平分训练数据，i.i.d.
    #初始化同一个参数的模型
    for i in range(args.num_users):
        new_idxs = set(np.random.choice(all_idxs, num_items, replace=False))
        print('user{}:,idxs:{}'.format(i,len(new_idxs)))
        all_idxs = list(set(all_idxs) - new_idxs)
        new_client = Client(args=args, dataset=dataset_train_flip, idxs=new_idxs, w=copy.deepcopy(net_glob.state_dict()))
        clients.append(new_client)

    server = Server(args = args, w = copy.deepcopy(net_glob.state_dict()))

    return clients, server


if __name__ == '__main__':

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args.device)
    acc_test_100_epoch = []
    print("load dataset...")
    dataset_train, dataset_test = load_dataset()

    dataset_train_flip =  copy.deepcopy(dataset_train)

    corruption_list = {
        'uniform': uniform_corruption,
        'flip1': flip1_corruption,
        'flip2': flip2_corruption,
    }
    if args.corruption_type is not None:
        corruption_matrix = corruption_list[args.corruption_type](args.corruption_ratio, args.num_classes)
        # print(corruption_matrix)
        for index in range(len(dataset_train_flip.targets)):
            p = corruption_matrix[dataset_train_flip.targets[index]]
            dataset_train_flip.targets[index] = np.random.choice(args.num_classes, p=p)

    print((dataset_train_flip.targets==dataset_train.targets))


    print("clients and server initialization...")
    clients, server = create_client_server()

    print(args.mode)

    Loss = np.zeros(args.rounds)
    Accuracy = np.zeros(args.rounds)
    Iter = np.zeros(args.rounds)

    for i in range(args.rounds):
        Iter[i] = i

    begin = time.time()

    # training
    print("start training...")
    torch.cuda.empty_cache()
    for iter in range(args.rounds):
        torch.cuda.empty_cache()
        client_select_list = random.sample(range(0,args.num_users), args.num_select_users)
        server.clients_update_w, server.clients_loss = [], []
        for idx in client_select_list:   #range(args.num_users):
            delta_w, loss = clients[idx].train()

            server.clients_update_w.append(delta_w)
            server.clients_loss.append(loss)

        # calculate global weights
        w_glob, loss_glob = server.FedAvg()

        # update local weights
        for idx in range(args.num_users):
            clients[idx].update(w_glob)
        torch.cuda.empty_cache()
        # print loss
        acc_train, loss_train = server.test(dataset_train)
        acc_test, loss_test = server.test(dataset_test)

        Loss[iter] = loss_train
        Accuracy[iter] = acc_test

        print('Round {:3d}, Training average loss {:.3f}'.format(iter, loss_glob))
        print('Round {:3d}, Training accuracy: {:.2f}'.format(iter, acc_train))
        print("Round {:3d}, Testing accuracy: {:.2f}".format(iter, acc_test))
        acc_test_100_epoch.append(acc_test)
    end = time.time()
    torch.cuda.empty_cache()
    print('Total training time: ', end - begin)
    
    # testing

    acc_train, loss_train = server.test(dataset_train)
    acc_test, loss_test = server.test(dataset_test)

    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    results_save_path='cifar10_s4_uni_round100_bs100_ep1_lr1-e2_m5e-1_baseline_results'
    if not os.path.exists(results_save_path):
        os.mkdir(results_save_path)

    np.save(os.path.join(results_save_path,'loss_cifar10_s4_uni{}_round100_bs100_ep1_lr1-e2_m5e-1.npy'.format(str(args.corruption_ratio).replace('.',''))),acc_test_100_epoch)



