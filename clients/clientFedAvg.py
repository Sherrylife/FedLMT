import time

import numpy as np
import ray
import torch
from clients.clientBase import ClientBase


@ray.remote(num_gpus=0.2, num_cpus=1.0)
class clientFedAvg(ClientBase):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.model = None
        self.receive_model_params = None

    def local_train(self, m):
        self.train_size, self.train_loader = self.load_train_data(self.client_id)
        self.model.train(True)
        self.optimizer = self.make_optimizer(self.model, self.lr)

        start_time = time.time()
        correct = 0.
        total = 0.1 #
        loss_logs = []
        for local_epoch in range(self.E):
            for i, data in enumerate(self.train_loader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                if self.cfg['mask']:
                    label_mask = torch.zeros(self.cfg['classes_size'], device=self.device)
                    label_mask[self.label_type] = 1
                    outputs = outputs.masked_fill(label_mask == 0, 0)

                if self.cfg['dataset_name'] in ['wikitext2', 'wikitext103', 'PennTreebank']:
                    loss = self.loss_fn(outputs, labels.unsqueeze(0))
                else:
                    loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # todo: save train loss and train accuracy
                _, predicts = torch.max(outputs, 1)
                correct += (predicts == labels).sum().item()
                total += len(labels)
                loss_logs.append(loss.mean().item())
        end_time = time.time()
        resutls = {
            'train_loss': np.array(loss_logs).mean(),
            'train_acc': np.array(correct / total).item(),
        }
        return resutls





