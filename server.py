import torch
import copy
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.Nets import CNNMnist
import numpy as np
from resnet import resnet32
Scaler = np.ones(10000)

# preset parameters
C = 2
Sigma = 0.3

class Server():
    def __init__(self, args, w):
        self.args = args
        self.clients_update_w = []
        self.clients_loss = []
        self.model = resnet32(args.num_classes).to(args.device) #CNNMnist(args=args).to(args.device)
        # self.model = CNNMnist(args=args).to(args.device)
        self.model.load_state_dict(w)

    def FedAvg(self):
        if self.args.mode == 'plain':
            update_w_avg = copy.deepcopy(self.clients_update_w[0])

            for k in update_w_avg.keys():
                for i in range(1, len(self.clients_update_w)):
                    update_w_avg[k] += self.clients_update_w[i][k]
                update_w_avg[k] = torch.div(update_w_avg[k], len(self.clients_update_w))
                # print("11111111")
                # print('update_w_avg type ',update_w_avg[k].type())
                # print('model type ',self.model.state_dict()[k].type())

                aaa=self.model.state_dict()[k]
                bbb=update_w_avg[k]
                if 'num_batches_tracked' in k:
                    continue
                else:
                    self.model.state_dict()[k] += update_w_avg[k]

        if self.args.mode == 'DP':
            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                for i in range(1, len(self.clients_update_w)):

                    # Here we seek to find the clipping parameter and clip the gradient
                    Scaler[i] = max(1, torch.norm(self.clients_update_w[i][k])/C)
                    self.clients_update_w[i][k] = self.clients_update_w[i][k]/Scaler[i]

                    update_w_avg[k] += self.clients_update_w[i][k]

                '''
                Here we define the Gaussian Noise, C and Sigma being preset parameters and
                the shape of the Eye Matrix being the shape of tensor update_w_avg[k], 
                where different ks represent different layers
                '''
                GaussianNoise = torch.normal(mean = torch.zeros(update_w_avg[k].shape), std = C * Sigma * torch.ones(update_w_avg[k].shape))

                update_w_avg[k] = update_w_avg[k] + GaussianNoise   # Add Noise
                update_w_avg[k] = torch.div(update_w_avg[k], len(self.clients_update_w))

                self.model.state_dict()[k] += update_w_avg[k] 
    

        elif self.args.mode == 'Paillier':
            pass
            '''
            part two: Paillier addition
            '''
        return copy.deepcopy(self.model.state_dict()), sum(self.clients_loss) / len(self.clients_loss)
    

    def test(self, datatest):
        self.model.eval()

        # testing
        test_loss = 0
        correct = 0
        data_loader = DataLoader(datatest, batch_size=self.args.bs)
        for idx, (data, target) in enumerate(data_loader):
            if self.args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = self.model(data)

            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()

            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
        return accuracy, test_loss
