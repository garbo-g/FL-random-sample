import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import copy
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from resnet import resnet32

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class Client():
    
    def __init__(self, args, dataset = None, idxs = None, w = None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs)
        self.model =resnet32(args.num_classes).to(args.device) #CNNMnist(args=args).to(args.device)
        self.model.load_state_dict(w)
        self.client_train_weight = [1]* (len(self.ldr_train))
        self.dataset=dataset
        self.idxs=idxs



    def train(self):
        import time
        time1=time.time()
        w_old = copy.deepcopy(self.model.state_dict())
        net = copy.deepcopy(self.model)
        self.ldr_train = DataLoader(DatasetSplit(self.dataset, self.idxs), batch_size=self.args.local_bs,sampler=WeightedRandomSampler(self.client_train_weight,int(len(self.dataset) /self.args.num_users),replacement=False))
        # tmp_train_weight=copy.deepcopy(self.client_train_weight)
        self.client_train_weight=[]
        net.train()

        #train and update
        optimizer = torch.optim.SGD(net.parameters(), lr = self.args.lr, momentum = self.args.momentum)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                cost = list(F.cross_entropy(log_probs, labels, reduce=False))
                if iter == self.args.local_ep - 1:
                    self.client_train_weight.extend(cost)

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())



        w_new = net.state_dict()
        torch.cuda.empty_cache()
        delta_w = {}


        '''
        0. part zero
            Baseline
        '''
        if self.args.mode == 'plain':
            for k in w_new.keys():
                delta_w[k] = w_new[k] - w_old[k]
        # print('delta_w',np.shape(delta_w))
        print()
        '''
        1. part one
            DP mechanism
            Here the DP mechanism is the same as the baseline, 
            and we implement the gradient-clipping and noise-adding mechanism in server.py
        '''
        if self.args.mode == 'DP':
            for k in w_new.keys():
                delta_w[k] = w_new[k] - w_old[k]

            
        '''
        2. part two
            Paillier enc/dec
        '''
        return delta_w, sum(batch_loss) / len(batch_loss)

    def update(self, w_glob):
        if self.args.mode == 'plain' or 'DP':
            self.model.load_state_dict(w_glob)
            
        elif self.args.mode == 'Paillier':
            
            '''
            part two: Paillier decryption
            '''
            self.model.load_state_dict(w_glob)