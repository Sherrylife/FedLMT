import os.path

import torch
import numpy as np
import random
import ujson
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class SplitDataset(Dataset):
    def __init__(self, dataset, idx):
        super().__init__()
        label = dataset.targets

        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        return self.dataset[self.idx[index]]


def iid(dataset, num_users, dataset_name):
    if dataset_name in ['mnist', 'cifar10', 'cifar100']:
        label = dataset.targets
    elif dataset_name in ['wikitext2', 'wikitext103', 'PennTreebank']:
        label = dataset.token
    else:
        raise ValueError('Not valid data name')
    num_items = int(len(dataset) / num_users)
    data_split, idx = {}, list(range(len(dataset)))
    label_split = {}
    for i in range(num_users):
        num_items_i = min(len(idx), num_items)
        data_split[i] = torch.tensor(idx)[torch.randperm(len(idx))[:num_items_i]].tolist()
        label_split[i] = torch.unique(label[data_split[i]]).tolist()
        idx = list(set(idx) - set(data_split[i]))
    return data_split, label_split

def test_iid(dataset, num_users, dataset_name,):
    if dataset_name in ['mnist', 'cifar10', 'cifar100']:
        label = dataset.targets
    elif dataset_name in ['WikiText2', 'WikiText103', 'PennTreebank']:
        label = dataset.token
    else:
        raise ValueError('Not valid data name')
    num_items = int(len(dataset) / num_users)
    data_split, idx = {}, list(range(len(dataset)))
    label_split = {}

    for i in range(num_users):
        num_items_i = min(len(idx), num_items)
        data_split[i] = sorted(torch.tensor(idx)[torch.arange(0, len(idx))[:num_items_i]].tolist())
        label_split[i] = torch.unique(label[data_split[i]]).tolist()
        idx = list(set(idx) - set(data_split[i]))
    return data_split, label_split

def new_iid(dataset, num_users, dataset_name,):
    """
    new implementation for iid distribution
    :return:
    """
    if dataset_name in ['mnist', 'cifar10', 'cifar100', 'tinyImagenet']:
        label = np.array(dataset.targets)
    elif dataset_name in ['wikitext2', 'wikitext103', 'PennTreebank']:
        label = dataset.token
    else:
        raise ValueError('Not valid data name')
    data_split = {i: [] for i in range(num_users)}
    label_split = {}
    label_idx_split = {}
    for i in range(len(label)):
        label_i = label[i].item()
        if label_i not in label_idx_split:
            label_idx_split[label_i] = []
        label_idx_split[label_i].append(i)

    num_per_class = [int(len(label_idx_split[label_i]) / num_users) for label_i in list(label_idx_split.keys())]
    for i in range(num_users):
        for idx, label_i in enumerate(list(label_idx_split.keys())):
            data_split[i].extend(torch.tensor(label_idx_split[label_i])[torch.randperm(len(label_idx_split[label_i]))[:num_per_class[idx]]].tolist())
            label_idx_split[label_i] = list(set(label_idx_split[label_i]) - set(data_split[i]))

    # shuffle again
    for i, (k, v) in enumerate(data_split.items()):
        data_split[k] = torch.tensor(v)[torch.randperm(len(v))].tolist()
        label_split[i] = torch.unique(torch.tensor(label[data_split[i]])).tolist()
    return data_split, label_split


def non_iid_1(dataset, num_shard, num_users, classes_size, dataset_name, label_split=None,):
    """
    Pathological distribution
    :param num_shard: the number of labels that each clients has
    :param label_split: remember the partition order of training set and apply that order to the test
                set to ensure that the training set and the test set have the same type of label
    :param classes_size: the number of label categories in the data set
    """
    label = np.array(dataset.targets)
    shard_per_user = num_shard #
    data_split = {i: [] for i in range(num_users)}
    label_idx_split = {}
    for i in range(len(label)):
        label_i = label[i].item()
        if label_i not in label_idx_split:
            label_idx_split[label_i] = []
        label_idx_split[label_i].append(i)
    shard_per_class = int(float(shard_per_user) * num_users / classes_size)
    for label_i in label_idx_split:
        label_idx = label_idx_split[label_i]
        num_leftover = len(label_idx) % shard_per_class
        leftover = label_idx[-num_leftover:] if num_leftover > 0 else []
        new_label_idx = np.array(label_idx[:-num_leftover]) if num_leftover > 0 else np.array(label_idx)
        new_label_idx = new_label_idx.reshape((shard_per_class, -1)).tolist()
        for i, leftover_label_idx in enumerate(leftover):
            new_label_idx[i] = np.concatenate([new_label_idx[i], [leftover_label_idx]])
        label_idx_split[label_i] = new_label_idx
    if label_split is None:
        label_split = list(range(classes_size)) * shard_per_class
        label_split = torch.tensor(label_split)[torch.randperm(len(label_split))].tolist()
        label_split = np.array(label_split).reshape((num_users, -1)).tolist()
        for i in range(len(label_split)):
            label_split[i] = np.unique(label_split[i]).tolist()
    for i in range(num_users):
        for label_i in label_split[i]:
            idx = torch.arange(len(label_idx_split[label_i]))[torch.randperm(len(label_idx_split[label_i]))[0]].item()
            data_split[i].extend(label_idx_split[label_i].pop(idx))
    # make data

    return data_split, label_split


def non_iid_2(dataset, alpha, num_users, classes_size, dataset_name, label_split=None, minimum_number=10):
    """
    Dirichlet distribution
    :param alpha: hyper-parameter for dirichlet distribution
    :param label_split: remember the partition order of training set and apply that order to the test
                set to ensure that the training set and the test set have the same type of label
    :param classes_size: the number of label categories in the data set
    """
    label = np.array(dataset.targets)
    dirichlet_alpha = alpha
    data_split = {i: [] for i in range(num_users)}
    idxs = np.arange(len(label))
    # sort labels
    idxs_labels = np.vstack((idxs, label))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

    min_size = 0
    K = classes_size
    N = len(label)
    if label_split is None:
        while min_size < minimum_number:
            label_split = []
            idx_batch = [[] for _ in range(num_users)]
            for k in range(K):
                idx_k = np.where(label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_users))
                label_split.append(proportions)
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(num_users):
            data_split[j] = idx_batch[j]
    else:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(label == k)[0]
            np.random.shuffle(idx_k)
            # proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_users))
            proportions = label_split[k]
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(num_users):
            data_split[j] = idx_batch[j]
    return data_split, label_split


def save_file(args, num_classes, train_set, test_set, clients_label, train_size, test_size):
    """
    :param args:
    :param num_classes:
    :param train_set:
    :param test_set:
    :param clients_label: store the types of labels each clients own and the number of each type
    :return:
    """
    config = {
        'num_users': args.num_users,
        'num_classes': num_classes,
        'noniid_type': args.noniid,
        'shard': args.shard,
        'alpha': args.alpha,
        'train_size': train_size,
        'test_size': test_size,
        'clients_label': clients_label,
    }
    print("Saving to disk.\n")
    for idx, train_list in enumerate(train_set):
        with open(args.train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_list)
            # np.savez(f, data=train_list)
    for idx, test_list in enumerate(test_set):
        with open(args.test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_list)
            # np.savez(f, data=test_list)
    with open(args.config_path, 'w') as f:
        ujson.dump(config, f)
    print("Finish generating dataset.\n")

def check(args, num_classes):

    # check existing dataset
    if os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_users'] == args.num_users and \
            config['num_classes'] == num_classes and \
            config['noniid_type'] == args.noniid and \
            config['shard'] == args.shard and \
            config['alpha'] == args.alpha:
            print("\nDataset already generated.\n")
            return True
    return False


def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)

def batchify(dataset, batch_size):
    num_batch = len(dataset) // batch_size
    dataset.token = dataset.token.narrow(0, 0, num_batch * batch_size)
    dataset.token = dataset.token.reshape(batch_size, -1)
    return dataset




