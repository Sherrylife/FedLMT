import random
import time
import torch
import numpy as np
import os
import copy
import ray
import json
from pathlib import Path
from clients.clientpFedLMT import clientpFedLMT
from servers.serverBase import ServerBase
from utils.model_utils import generate_model



class pFedLMT(ServerBase):

    def __init__(self, cfg, larger_model, logger,):
        super(pFedLMT, self).__init__(cfg, larger_model, logger,)
        self.cfg = cfg
        self.meta_round = cfg['meta_round']
        self.ratio_LR = self.make_LR_ratio(current_round=0,)
        self.decom_rule = cfg['decom_rule']
        self.model_rate = np.ones(self.num_clients)
        self.clients_params_idx = None # stores location information for all client-side local small models
        self.clients_obj = [clientpFedLMT.remote(cfg) for _ in range(self.PROCESS_NUM)]
        # self.label_split = dataset_ref['label_type'] # record the types of labels that each client has

        self.current_round = 0
        self.global_model = larger_model.cpu()
        self.optimizer = self.make_optimizer(self.global_model, self.lr)
        self.scheduler = self.make_scheduler(self.optimizer)

        self.clients_personalized_params = [
            copy.deepcopy(self.global_model.personalized.state_dict())
            for _ in range(self.num_clients)]

        self.client_trained_indexes = None
        self.clients_test_loss1 = []
        self.clients_test_acc1 = []

        # this zero gradient update is needed to avoid a warning message,
        # https://github.com/ildoonet/pytorch-gradual-warmup-lr/issues/8
        self.optimizer.zero_grad()
        self.optimizer.step()

    def make_LR_ratio(self, current_round,):
        """
        stage 1: use fixed LR ratio for all clients.
        stage 2: for each client, there is a different LR ratio.
        """
        if current_round < self.meta_round: # stage 1
            each_client_rate = [self.cfg['ratio_LR'] for _ in range(self.cfg['num_users'])]
            each_client_rate = np.array(each_client_rate)
            return each_client_rate
        else: # stage 2
            each_client_rate = []
            model_mode = self.cfg['model_mode'].split('-')
            if self.cfg['model_split_mode'] == 'dynamic':
                mode_rate, proportion = [], []
                for m in model_mode:
                    mode_rate.append(self.cfg['model_LR_rate'][m[0]])
                    proportion.append(int(m[1:]))
                proportion = (np.array(proportion) / sum(proportion)).tolist()  # Convert to percentage
                rate_idx = torch.multinomial(torch.tensor(proportion), num_samples=self.cfg['num_users'],
                                             replacement=True).tolist()
                each_client_rate = np.array(mode_rate)[rate_idx]
                return each_client_rate
            elif self.cfg['model_split_mode'] == 'fix':
                mode_rate, proportion = [], []
                for m in model_mode:
                    mode_rate.append(self.cfg['model_LR_rate'][m[0]])
                    proportion.append(int(m[1:]))
                num_users_proportion = self.cfg['num_users'] // sum(proportion)
                for i in range(len(mode_rate)):
                    each_client_rate += np.repeat(mode_rate[i], num_users_proportion * proportion[i]).tolist()
                each_client_rate = each_client_rate + [each_client_rate[-1] for _ in
                                                       range(self.cfg['num_users'] - len(each_client_rate))]
                each_client_rate = np.array(each_client_rate)
                return each_client_rate
            else:
                raise ValueError('Not valid model split mode!')

    def personalized_test_model(self, seleceted_all=True):
        """
        test the personalized model
        :param seleceted_all: if false, only calculate the average accuracy of the clients participating in the current round.
        :return: evaluation: containing mean test accuracy and mean loss.
        """

        if seleceted_all:
            results = []
            for m in range(0, self.num_clients, self.PROCESS_NUM):
                processes = []
                for k in range(m, min(m + self.PROCESS_NUM, self.num_clients)):
                    model_id = ray.put([self.global_model.common.state_dict(), self.clients_personalized_params[k]])
                    processes.append(self.clients_obj[k % self.PROCESS_NUM].test_personalized_model.remote(self.ratio_LR[k], k, [model_id]))
                # results store test set loss and accuracy for each clients
                results.extend(ray.get(processes))
            return results
        else:
            results = []
            for m in range(0, self.num_active_clients, self.PROCESS_NUM):
                processes = []
                for k in range(m, min(m + self.PROCESS_NUM, self.num_active_clients)):
                    model_id = ray.put([self.global_model.common.state_dict(), self.clients_personalized_params[k]])
                    processes.append(self.clients_obj[k % self.PROCESS_NUM].test_model_client.remote(self.ratio_LR[k], k, [model_id]))
                # results store test set loss and accuracy for each clients
                results.extend(ray.get(processes))
            return results

    def change_model(self):
        self.global_model.recover_large_layer()
        model_type = np.unique(self.ratio_LR)
        client_models = {}
        for value in model_type:
            client_model = copy.deepcopy(self.global_model)
            client_model.decom_large_layer(ratio_LR=value)
            client_models[value] = client_model.personalized.state_dict()

        for user_id in range(self.num_clients):
            for value in model_type:
                if self.ratio_LR[user_id] == value:
                    self.clients_personalized_params[user_id] = copy.deepcopy(client_models[value])
                    break


    def train(self):
        for i in range(self.communication_round):
            t0 = time.time()
            self.user_idx = self.select_client()
            self.ratio_LR = self.make_LR_ratio(current_round=i, )
            self.test_comm_comput(self.ratio_LR, self.user_idx)

            if i == self.meta_round:
                self.change_model()

            if i % self.evaluate_gap == 0:
                test_evaluation = self.personalized_test_model()

            lr = self.optimizer.param_groups[0]['lr']

            # if i % 100 == 0:
            #     self.save_model(self.global_model, path="./results/temp_model_file/", current_round=i)

            total_receive_packets = []
            K = min(self.num_active_clients, self.PROCESS_NUM)
            for m in range(0, self.num_active_clients, K):
                M = min(m + self.PROCESS_NUM, self.num_active_clients)
                self.broadcast(lr, self.user_idx[m: M], current_round=i,)
                train_evaluation = ray.get([self.clients_obj[k].local_train.remote(k,) for k in range((M-m))])
                receive_packets = ray.get([self.clients_obj[k].upload_stage.remote() for k in range((M-m))])
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
        :param clients_params: the model parameters returned by the client after local training
        :param user_idx: the index of clients participating in the current round
        :return:
        """
        clients_weights = np.array([i[0] for i in receive_packets])
        clients_weights = clients_weights / clients_weights.sum()

        clients_large_params = [i[1] for i in receive_packets]
        self.aggregation(clients_weights, clients_large_params, user_idx)
        self.current_round += 1


    def broadcast(self, lr, user_idx, current_round):

        clients_params = [
            [copy.deepcopy(self.global_model.common.state_dict()), self.clients_personalized_params[user_idx[i]]]
            for i in range(len(user_idx))]
        param_ids = [ray.put(client_param) for client_param in clients_params]
        ray.get([
            self.clients_obj[m].update_config.remote(user_idx[m], self.ratio_LR, current_round, {
                'lr': lr,
                'model_rate': self.model_rate[user_idx[m]],
                'ratio_LR': self.ratio_LR[user_idx[m]],
                'client_params': param_ids[m],
                # 'label_type': self.label_split[user_idx[m]],
            })
            for m in range(len(user_idx))
        ])

    def aggregation(self, clients_weights, local_parameters, user_idx):
        """
        :param clients_weights:
        :param local_parameters:
        :param user_idx:
        :return:
        """

        new_common = self.global_model.common.cpu().state_dict()
        for k, v in new_common.items():
            new_common[k] = torch.zeros(v.shape)
        for weight, client_model in zip(clients_weights, local_parameters):
            for k, v in client_model['common'].items():
                new_common[k] += weight * v
        self.global_model.common.load_state_dict(new_common)

        if self.current_round < self.meta_round:  # stage 1
            new_personalized = self.global_model.personalized.cpu().state_dict()
            for k, v in new_personalized.items():
                new_personalized[k] = torch.zeros(v.shape)
            for weight, client_model in zip(clients_weights, local_parameters):
                for k, v in client_model['personalized'].items():
                    new_personalized[k] += weight * v
            self.global_model.personalized.load_state_dict(new_personalized)
            # store each client's personalized layers
            self.clients_personalized_params = [
                self.global_model.personalized.state_dict()
                for _ in range(self.num_clients)]

        else: # stage 2
            # store each client's personalized layers
            for user_id, client_model in zip(user_idx, local_parameters):
                self.clients_personalized_params[user_id] = client_model['personalized']

        # print("Done")


















