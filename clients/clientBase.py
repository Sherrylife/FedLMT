
import ray
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.data_utils import *
from utils.functions import *
from utils.model_utils import generate_model


class ClientBase(object):
    """
    Base class for clients in federated learning
    """
    def __init__(self, cfg, **kwargs):
        super(ClientBase, self).__init__()
        set_seed(cfg['seed'])
        self.cfg = cfg
        self.batch_size = cfg['batch_size']['train']
        self.batch_size_test = cfg['batch_size']['test']
        self.device = cfg['device']
        self.lr = cfg['lr']
        self.E = cfg['l_epochs']
        self.dataset_name = cfg['dataset_name']
        self.client_id = None
        self.model_name = cfg['model_name']
        self.current_round = 0
        self.model = None
        self.dataset_transforms = None
        self.generate_transforms()

        self.model_params = None
        self.train_set = None
        self.test_set = None
        self.train_size = None
        self.test_size = None
        self.optimizer = None
        self.loss_fn = None
        self.model_rate = None  # Pruning rate of the model
        self.train_loader = None
        self.test_loader = None
        self.label_type = None
        self.receive_model_params = None


    def update_config(self, client_id, current_round, model_ref):
        """
        :param client_id： id of the clients participating in the training for the current communication round
        :param model_ref: parameters related to model training
        :return:
        """
        receive_params = ray.get(model_ref['client_params'])
        self.receive_model_params = receive_params
        self.loss_fn = nn.CrossEntropyLoss()
        self.client_id = client_id

        self.label_type = model_ref['label_type']
        self.lr = model_ref['lr']
        # update model
        self.model_rate = model_ref['model_rate']
        self.current_round = current_round
        self.model = self.generate_model_type(self.model_name, self.model_rate, self.dataset_name)
        self.model.load_state_dict(self.receive_model_params)

    def generate_model_type(self, model_name, model_rate, dataset_name, depth_rate=4, train_module_rule=[0, 0]):
        return generate_model(model_name, model_rate, depth_rate, cfg=self.cfg)

    def generate_transforms(self):
        if 'cifar' in self.dataset_name:
            self.dataset_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])
        elif 'svhn' in self.dataset_name:
            self.dataset_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=2),
            ])
        elif 'tiny_imagenet' in self.dataset_name:
            self.dataset_transforms = None
            # self.dataset_transforms = transforms.Compose([
            #     transforms.RandomCrop(32, padding=4),
            #     transforms.RandomHorizontalFlip(),
            # ])
        elif 'mnist' in self.dataset_name:
            self.dataset_transforms = None
        else:
            self.dataset_transforms = None

    def upload(self):
        """
        upload train data size and local model
        :return:
        """
        model_state = {k: v.detach().clone().cpu() for k, v in self.model.to(self.cfg['device']).state_dict().items()}

        return [self.train_size, model_state]

    def load_train_data(self, client_id):
        train_data = read_client_data(self.dataset_name, client_id, self.cfg, is_train=True)
        if self.cfg['dataset_name'] in ['wikitext2', 'wikitext103', 'PennTreebank']:
            train_loader = BatchDataset(dataset=train_data, seq_length=self.cfg['bptt'])
            return len(train_data), train_loader
        else:
            train_data = MyDataset(train_data, self.dataset_transforms)
            return len(train_data), DataLoader(dataset=train_data, shuffle=self.cfg['shuffle']['train'],
                            batch_size=self.batch_size, pin_memory=True,
                            num_workers=self.cfg['num_workers'])

    def load_test_data(self, client_id):
        test_data = read_client_data(self.dataset_name, client_id, self.cfg, is_train=False)
        if self.cfg['dataset_name'] in ['wikitext2', 'wikitext103', 'PennTreebank']:
            test_loader = BatchDataset(dataset=test_data, seq_length=self.cfg['bptt'])
            return len(test_data), test_loader
        else:
            return len(test_data), DataLoader(dataset=test_data, shuffle=self.cfg['shuffle']['test'],
                            batch_size=self.batch_size_test, pin_memory=True,
                            num_workers=self.cfg['num_workers'])

    def test_model_client(self, client_id, global_model_id):
        """
        Test using the clients's own set of tests
        :param global_model_id: store information such as global model
        """
        [global_model] = ray.get(global_model_id)
        self.test_size, self.test_loader = self.load_test_data(client_id)

        model = global_model.to(self.device)
        model.eval()
        correct = 0.
        error = 0.
        total = 0.

        with torch.no_grad():
            for batch_id, data in enumerate(self.test_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = model(inputs)
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

    def make_optimizer(self, model, lr):
        if self.cfg['optimizer_name'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=self.cfg['momentum'],
                                  weight_decay=self.cfg['weight_decay'])
        elif self.cfg['optimizer_name'] == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=self.cfg['momentum'],
                                      weight_decay=self.cfg['weight_decay'])
        elif self.cfg['optimizer_name'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=self.cfg['weight_decay'])
        elif self.cfg['optimizer_name'] == 'Adamax':
            optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=self.cfg['weight_decay'])
        else:
            raise ValueError('Not valid optimizer name')
        return optimizer

    def make_scheduler(self, optimizer):
        if self.cfg['scheduler_name'] == 'None':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
        elif self.cfg['scheduler_name'] == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg['step_size'], gamma=self.cfg['factor'])
        elif self.cfg['scheduler_name'] == 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg['milestones'], gamma=self.cfg['factor'])
        elif self.cfg['scheduler_name'] == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        elif self.cfg['scheduler_name'] == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg['num_epochs']['global'],
                                                             eta_min=self.cfg['min_lr'])
        elif self.cfg['scheduler_name'] == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.cfg['factor'],
                                                             patience=self.cfg['patience'], verbose=True,
                                                             threshold=self.cfg['threshold'], threshold_mode='rel',
                                                             min_lr=self.cfg['min_lr'])
        elif self.cfg['scheduler_name'] == 'CyclicLR':
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.cfg['lr'], max_lr=10 * self.cfg['lr'])
        else:
            raise ValueError('Not valid scheduler name')
        return scheduler




