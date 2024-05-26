import os
import torch
import numpy as np
import ujson
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class BatchDataset(Dataset):
    """
    For Wikitext2 dataset, NLP task
    """
    def __init__(self, dataset, seq_length):
        super().__init__()
        self.dataset = dataset
        self.seq_length = seq_length
        self.S = dataset[0][0].size(0)
        self.idx = list(range(0, self.S, seq_length))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        seq_length = min(self.seq_length, self.S - index)
        input = self.dataset[0][0][self.idx[index]:self.idx[index] + seq_length]
        label = self.dataset[0][1][self.idx[index]:self.idx[index] + seq_length]
        return input, label


class MyDataset(Dataset):
    def __init__(self, dataset,
                 transforms: transforms.Compose):
        self.transforms = transforms
        self.dataset = dataset
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        image, label = self.dataset[item]
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label


class SplitDataset(Dataset):
    def __init__(self, dataset, idx):
        super().__init__()
        label = dataset.targets

        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        return self.dataset[self.idx[index]]


def read_clients_label(dataset_name, cfg):
    """
    Stores the type and number of labels for the clients training or test set
    :return:
        label_split: the type of label that each clients has for the training set
        label_num: the number of each type of label in the training set that each clients has
    """
    partition_mode = cfg['control_name'].split('_')[3]
    if partition_mode == 'iid':
        dir_path = './datasets/data/' + dataset_name + '/noniid0' + '/config.json'
    elif 'noniid1' in partition_mode:
        dir_path = './datasets/data/' + dataset_name + '/noniid1' + '/config.json'
    elif 'noniid2' in partition_mode:
        dir_path = './datasets/data/' + dataset_name + '/noniid2' + '/config.json'
    else:
        print("Error: data partition mode is wrong!")
        ValueError

    clients_labels = None
    if os.path.exists(dir_path):
        with open(dir_path, 'r') as f:
            config = ujson.load(f)
        clients_labels = config['clients_label']['train']
        label_split = [list(map(int, list(clients_labels[i].keys()))) for i in range(len(clients_labels))]
        label_num = [list(clients_labels[i].values()) for i in range(len(clients_labels))]
        return label_split, label_num
    else:
        print("Error: Can not find config.json !")
        ValueError

def read_experiment_setting(dataset_name, args):
    """
    Modify the experiment name based on the generated data mode.
    :param dataset_name:
    :param cfg:
    :return:
    """
    s = args['control_name'].split('_')
    partition_mode = s[3]
    if partition_mode == 'iid':
        dir_path = './datasets/data/' + dataset_name + '/noniid0' + '/config.json'
    elif 'noniid1' in partition_mode:
        dir_path = './datasets/data/' + dataset_name + '/noniid1' + '/config.json'
    elif 'noniid2' in partition_mode:
        dir_path = './datasets/data/' + dataset_name + '/noniid2' + '/config.json'
    else:
        print("Error: data partition mode is wrong!")
        raise ValueError
    if os.path.exists(dir_path):
        with open(dir_path, 'r') as f:
            config = ujson.load(f)
        num_clients = config['num_users']
        noniid_type = config['noniid_type']
        shard = config['shard']
        dir_alpha = config['alpha']
        s = args['control_name'].split('_')
        s[1] = str(num_clients)
        if noniid_type == "0":
            s[3] = 'iid'
        elif noniid_type == "1":
            s[3] = 'noniid1-'+str(shard)
        elif noniid_type == "2":
            s[3] = 'noniid2-'+str(dir_alpha)
        args['control_name'] = '_'.join(s)
        return args
    else:
        print("Error: Can not find config.json !")
        raise NotImplementedError



def read_client_data(dataset_name, client_id, cfg, is_train=True):
    """
    load clients local dataset
    :param dataset_name:
    :param client_id:
    :param partition_mode: noniid type
    :param is_train:
    :return:
    """
    partition_mode = cfg['control_name'].split('_')[3]

    if is_train:
        train_data = read_data(dataset_name, client_id, partition_mode, is_train)
        if cfg['dataset_name'] in ['wikitext2', 'wikitext103', 'PennTreebank']:
            x_train = torch.Tensor(train_data['x']).type(torch.int64)
        else:
            x_train = torch.Tensor(train_data['x']).type(torch.float32)

        y_train = torch.Tensor(train_data['y']).type(torch.int64)
        train_data = [(x, y) for x, y in zip(x_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset_name, client_id, partition_mode, is_train)
        if cfg['dataset_name'] in ['wikitext2', 'wikitext103', 'PennTreebank']:
            x_test = torch.Tensor(test_data['x']).type(torch.int64)
        else:
            x_test = torch.Tensor(test_data['x']).type(torch.float32)

        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(x_test, y_test)]
        return test_data


def read_data(dataset, idx, partition_mode, is_train=True):
    if partition_mode == 'iid':
        dir_path = './datasets/data/' + dataset + '/noniid0/'
    elif 'noniid1' in partition_mode:
        dir_path = './datasets/data/' + dataset + '/noniid1/'
    elif 'noniid2' in partition_mode:
        dir_path = './datasets/data/' + dataset + '/noniid2/'
    else:
        print("Error: data partition mode is wrong!")
        ValueError
    if is_train:
        train_data_dir = os.path.join(dir_path, 'proc_train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join(dir_path, 'proc_test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data

