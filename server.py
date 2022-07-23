import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gtg_shapley_value import GTGShapleyValue
from resnet import resnet32

Scaler = np.ones(10000)

# preset parameters
C = 2
Sigma = 0.3


class Server:
    def __init__(self, args, w, test_dataset):
        self.args = args
        self.clients_update_w = {}
        self.clients_loss = []
        self.model = resnet32(args.num_classes).to(
            args.device
        )  # CNNMnist(args=args).to(args.device)
        # self.model = CNNMnist(args=args).to(args.device)
        self.model.load_state_dict(w)
        self._gtg_algorithm = None
        self.test_dataset = test_dataset

    def __compute_subset_gtg(self, round_number, worker_ids):
        parameters = self.__fed_avg_algorithm(worker_ids)

        old_model = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(parameters)

        _,acc = self.test(self.test_dataset)

        self.model.load_state_dict(old_model)
        return acc

    def __fed_avg_algorithm(self, worker_ids):
        first_client = self.clients_update_w[0]
        avg_update = {}
        for k in first_client:
            updates = [client_update[k] for client_update in self.clients_update_w]
            avg_update[k] = sum(updates) / len(updates)
        parameters = copy.deepcopy(self.model.state_dict())
        for k in parameters:
            if "num_batches_tracked" in k:
                continue
            parameters[k] += avg_update[k]
        return parameters

    def FedAvg(self):
        if self.args.mode == "fed_GTG_SV":
            # time1=time.time()
            if self._gtg_algorithm is None:
                self._gtg_algorithm = GTGShapleyValue(worker_number=self.args.num_users)
            self._gtg_algorithm.set_metric_function(self.__compute_subset_gtg)
            self._gtg_algorithm.compute()
            best_subset: set = set(
                self._gtg_algorithm.shapley_values_S[
                    self._gtg_algorithm.round_number
                ].keys()
            )
            assert best_subset
            print("use subset", best_subset)
            parameters = self.__fed_avg_algorithm(worker_ids=best_subset)
            self.model.load_state_dict(parameters)

        if self.args.mode == "plain":
            update_w_avg = copy.deepcopy(self.clients_update_w[0])

            for k in update_w_avg.keys():
                for i in range(1, len(self.clients_update_w)):
                    update_w_avg[k] += self.clients_update_w[i][k]
                update_w_avg[k] = torch.div(update_w_avg[k], len(self.clients_update_w))
                # print("11111111")
                # print('update_w_avg type ',update_w_avg[k].type())
                # print('model type ',self.model.state_dict()[k].type())

                aaa = self.model.state_dict()[k]
                bbb = update_w_avg[k]
                if "num_batches_tracked" in k:
                    continue
                else:
                    self.model.state_dict()[k] += update_w_avg[k]

        if self.args.mode == "DP":
            update_w_avg = copy.deepcopy(self.clients_update_w[0])
            for k in update_w_avg.keys():
                for i in range(1, len(self.clients_update_w)):

                    # Here we seek to find the clipping parameter and clip the gradient
                    Scaler[i] = max(1, torch.norm(self.clients_update_w[i][k]) / C)
                    self.clients_update_w[i][k] = (
                        self.clients_update_w[i][k] / Scaler[i]
                    )

                    update_w_avg[k] += self.clients_update_w[i][k]

                """
                Here we define the Gaussian Noise, C and Sigma being preset parameters and
                the shape of the Eye Matrix being the shape of tensor update_w_avg[k],
                where different ks represent different layers
                """
                GaussianNoise = torch.normal(
                    mean=torch.zeros(update_w_avg[k].shape),
                    std=C * Sigma * torch.ones(update_w_avg[k].shape),
                )

                update_w_avg[k] = update_w_avg[k] + GaussianNoise  # Add Noise
                update_w_avg[k] = torch.div(update_w_avg[k], len(self.clients_update_w))

                self.model.state_dict()[k] += update_w_avg[k]

        elif self.args.mode == "Paillier":
            pass
            """
            part two: Paillier addition
            """
        return copy.deepcopy(self.model.state_dict()), sum(self.clients_loss) / len(
            self.clients_loss
        )

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
            test_loss += F.cross_entropy(log_probs, target, reduction="sum").item()

            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
        return accuracy, test_loss
