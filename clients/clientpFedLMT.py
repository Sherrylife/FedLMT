import time

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from clients.clientBase import ClientBase
from utils.model_utils import generate_model



@ray.remote(num_gpus=0.2, num_cpus=1.0)
class clientpFedLMT(ClientBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.model_name = cfg['model_name']
        self.model = None
        self.current_round = 0
        self.meta_round = cfg['meta_round']
        self.warmup = cfg['warmup']
        self.warmupT = cfg['warmupT']
        self.cur_warmup_stage = 0
        self.decom_rule = cfg['decom_rule']


    def upload_stage(self,):

        model_state = {'common': None, 'personalized': None,}
        model_state['common'] = {k: v.detach().clone().cpu() for k, v in self.model.common.state_dict().items()}
        model_state['personalized'] = {k: v.detach().clone().cpu() for k, v in self.model.personalized.state_dict().items()}
        return [self.train_size, model_state]

    def test_personalized_model(self, client_LR_ratio, client_id, model_id):
        """
        Test using the clients's own set of tests
        """
        p_model = generate_model(self.cfg['model_name'], self.cfg['dataset_name'], cfg=self.cfg,
                                    LR_ratio=client_LR_ratio)

        [personalized_params] = ray.get(model_id)
        self.test_size, self.test_loader = self.load_test_data(client_id)

        p_model.common.load_state_dict(personalized_params[0])
        p_model.personalized.load_state_dict(personalized_params[1])

        p_model = p_model.to(self.device)
        p_model.eval()
        correct = 0.
        error = 0.
        total = 0.

        with torch.no_grad():
            for batch_id, data in enumerate(self.test_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = p_model(inputs)
                if self.cfg['dataset_name'] in ['wikitext2', 'wikitext103', 'PennTreebank']:
                    ce = F.cross_entropy(outputs, labels.unsqueeze(0))
                    error += torch.exp(ce).item() * len(labels) # Perplexity
                    # error += F.cross_entropy(outputs, labels.unsqueeze(0)).item() * len(labels)
                    _, predicts = torch.max(outputs, 1)
                    correct += (predicts == labels.unsqueeze(0)).sum().item()
                else:
                    error += F.cross_entropy(outputs, labels).item() * len(labels)
                    _, predicts = torch.max(outputs, 1)
                    correct += (predicts == labels).sum().item()

                total += len(labels)
        acc = correct / total
        loss = error / total
        # todo: save acc and loss
        results = {
            'test_loss': loss,
            'test_acc': acc
        }
        return results



    def update_config(self, client_id, ratio_LR, current_round, model_ref):
        """
        :param client_id： id of the client participating in the training for the current communication round
        :param model_ref: parameters related to model training
        :return:
        """
        self.current_round = current_round
        self.lr = model_ref['lr']
        self.loss_fn = nn.CrossEntropyLoss()
        self.client_id = client_id
        # self.label_type = label_type
        self.model_rate = model_ref['model_rate']

        receive_params = ray.get(model_ref['client_params'])
        self.ratio_LR = model_ref['ratio_LR']

        self.model = generate_model(self.cfg['model_name'], self.cfg['dataset_name'], cfg=self.cfg, LR_ratio=self.ratio_LR)

        self.model.common.load_state_dict(receive_params[0])
        self.model.personalized.load_state_dict(receive_params[1])


    def local_train(self, m):

        self.train_size, self.train_loader = self.load_train_data(self.client_id)
        self.model.train(True)
        self.optimizer = self.make_optimizer(self.model, self.lr)

        start_time = time.time()
        correct = 0.
        total = 0.
        loss_logs = []


        if self.current_round < self.meta_round:         # stage 1
            for local_epoch in range(self.E):
                for i, data in enumerate(self.train_loader):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    if self.cfg['dataset_name'] in ['wikitext2', 'wikitext103', 'PennTreebank']:
                        loss = self.loss_fn(outputs, labels.unsqueeze(0))
                    else:
                        loss = self.loss_fn(outputs, labels)

                    if self.cfg['train_decay'] == 'frobenius':
                        loss += self.cfg['coef_decay'] * self.model.frobenius_decay()

                    loss.backward()
                    self.optimizer.step()
                    # todo: save train loss and train accuracy
                    _, predicts = torch.max(outputs, 1)
                    correct += (predicts == labels).sum().item()
                    total += len(labels)
                    loss_logs.append(loss.mean().item())
        else:  # stage 2
            train_batch_num = len(self.train_loader)
            E_body = int(0.5*self.E*train_batch_num)
            E_tail = int(self.E * train_batch_num - E_body)

            j = 0
            for local_epoch in range(self.E):
                for i, data in enumerate(self.train_loader):
                    if j < E_tail:
                        for p in self.model.common.parameters():
                            p.requires_grad = False
                        for p in self.model.personalized.parameters():
                            p.requires_grad = True
                    else:
                        for p in self.model.common.parameters():
                            p.requires_grad = True
                        for p in self.model.personalized.parameters():
                            p.requires_grad = False

                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    if self.cfg['dataset_name'] in ['wikitext2', 'wikitext103', 'PennTreebank']:
                        loss = self.loss_fn(outputs, labels.unsqueeze(0))
                    else:
                        loss = self.loss_fn(outputs, labels)

                    if self.cfg['train_decay'] == 'frobenius':
                        loss += self.cfg['coef_decay'] * self.model.frobenius_decay()

                    loss.backward()
                    self.optimizer.step()
                    # todo: save train loss and train accuracy
                    _, predicts = torch.max(outputs, 1)
                    correct += (predicts == labels).sum().item()
                    total += len(labels)
                    loss_logs.append(loss.mean().item())
                    j += 1

        end_time = time.time()
        resutls = {
            'train_loss': np.array(loss_logs).mean(),
            'train_acc': np.array(correct / total).item(),
        }
        return resutls




