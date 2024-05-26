import time
import torch
import numpy as np
import os
import copy
import ray
import json
from pathlib import Path
from clients.clientFedAvg import clientFedAvg
from servers.serverBase import ServerBase
from collections import OrderedDict
from utils.logger import LoggerCreator

class FedAvg(ServerBase):

    def __init__(self, cfg, global_model, logger):
        super(FedAvg, self).__init__(cfg, global_model, logger)
        self.cfg = cfg
        self.model_rate = np.ones(self.num_clients)
        self.clients_params_idx = None # stores location information for all clients-side local small models
        self.clients_obj = [clientFedAvg.remote(cfg) for _ in range(self.PROCESS_NUM)]
        # self.label_split = dataset_ref['label_type'] # record the types of labels that each clients has
        self.global_model = global_model.cpu()

        self.optimizer = self.make_optimizer(self.global_model, self.lr)
        self.scheduler = self.make_scheduler(self.optimizer)

        # this zero gradient update is needed to avoid a warning message,
        # https://github.com/ildoonet/pytorch-gradual-warmup-lr/issues/8
        self.optimizer.zero_grad()
        self.optimizer.step()

    def train(self):
        for i in range(self.communication_round):
            t0 = time.time()
            self.user_idx = self.select_client()
            if i % self.evaluate_gap == 0:
                test_evaluation = self.test_model(self.global_model)

            if i % 100 == 0:
                self.save_model(self.global_model, path="./results/temp_model_file/", current_round=i)

            lr = self.optimizer.param_groups[0]['lr']
            total_receive_packets = []
            K = min(self.num_active_clients, self.PROCESS_NUM)
            for m in range(0, self.num_active_clients, K):
                M = min(m + self.PROCESS_NUM, self.num_active_clients)
                print(f"round {i}: client id: {self.user_idx[m: M]}")
                self.broadcast(lr, self.user_idx[m: M], current_round=i)
                train_evaluation = ray.get([self.clients_obj[k].local_train.remote(k,) for k in range((M-m))])
                receive_packets = ray.get([self.clients_obj[k].upload.remote() for k in range((M-m))])
                total_receive_packets.extend(receive_packets)

            self.step(total_receive_packets, self.user_idx)
            self.scheduler.step()

            t1 = time.time()
            self.log(i, t1-t0, train_evaluation, test_evaluation)

        self.save_results()
        self.save_model(self.global_model)



    def step(self, receive_packets, user_idx):
        """
        receive_packets contains clients_weights and clients_params.
        :param clients_weights: the weighted aggregation coefficient obtained according to the amount of data,
        :param clients_params: the model parameters returned by the clients after local training
        :param user_idx: the index of clients participating in the current round
        :return:
        """
        clients_weights = np.array([i[0] for i in receive_packets])
        clients_weights = clients_weights / clients_weights.sum()
        clients_params = [i[1] for i in receive_packets]
        self.aggregation(clients_weights, clients_params, user_idx)
        self.current_round += 1

    def broadcast(self, lr, user_idx, current_round):
        clients_params = [copy.deepcopy(self.global_model.state_dict()) for _ in range(len(user_idx))]
        param_ids = [ray.put(client_param) for client_param in clients_params]
        ray.get([
            self.clients_obj[m].update_config.remote(user_idx[m], current_round, {
                'lr': lr,
                'model_rate': self.model_rate[user_idx[m]],
                'label_type': self.label_split[user_idx[m]],
                'client_params': param_ids[m], # object
            })
            for m in range(len(user_idx))
        ])

    def aggregation(self, clients_weights, local_parameters, user_idx):
        """
        aggregate the local model of the clients on cpu
        """
        new_global_model = self.global_model.cpu().state_dict()
        for k, v in new_global_model.items():
            new_global_model[k] = torch.zeros(v.shape)
        for weight, client_model in zip(clients_weights, local_parameters):
            for k, v in client_model.items():
                new_global_model[k] += weight * v
        self.global_model.load_state_dict(new_global_model)












