import numpy as np
import os
import sys
import random
import torch
import torchvision.transforms as transforms
from datasets_utils import check, save_file, iid, non_iid_1, non_iid_2, new_iid, test_iid
import argparse

from torchvision import datasets

seed = 31
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


def generate_cifar10(dir_path, args):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    train_path = os.path.dirname(args.train_path)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    test_path = os.path.dirname(args.test_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    dataset = {}
    dataset['train'] = datasets.CIFAR10(root=dir_path + "rawdata", train=True, download=True,
                                        transform=transforms.Compose([
                                            # transforms.RandomCrop(32, padding=4),
                                            # transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    dataset['test'] = datasets.CIFAR10(root=dir_path + "rawdata", train=False, download=True,
                                       transform=transforms.Compose([
                                           # transforms.RandomCrop(32, padding=4),
                                           # transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

    num_classes = len(dataset['train'].classes)

    # The same data partition mode has been generated. No further action is required.
    if check(args, num_classes):
        return

    trainloader = torch.utils.data.DataLoader(
        dataset['train'], batch_size=len(dataset['train'].data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        dataset['test'], batch_size=len(dataset['test'].data), shuffle=False)

    # Extended data dimension
    for _, train_data in enumerate(trainloader, 0):
        dataset['train'].data, dataset['train'].targets = train_data
    for _, test_data in enumerate(testloader, 0):
        dataset['test'].data, dataset['test'].targets = test_data

    data_split = {}
    dataset_name = "cifar10"
    train_size = len(dataset['train'])
    test_size = len(dataset['test'])
    if args.noniid == "0":
        data_split['train'], label_split = new_iid(dataset['train'], args.num_users, dataset_name)
        data_split['test'], label_split = new_iid(dataset['test'], args.num_users, dataset_name)
    elif args.noniid == "1":
        data_split['train'], label_split = non_iid_1(dataset['train'], args.shard, args.num_users, args.classes_size, dataset_name)
        data_split['test'], _ = non_iid_1(dataset['test'], args.shard, args.num_users,  args.classes_size, dataset_name, label_split)
    elif args.noniid == "2":
        data_split['train'], label_split = non_iid_2(dataset['train'], args.alpha, args.num_users, args.classes_size, dataset_name)
        data_split['test'], _ = non_iid_2(dataset['test'], args.alpha, args.num_users, args.classes_size, dataset_name, label_split)

    train_set, test_set = [], []
    clients_labels = {'train': [], 'test': []}
    for i in range(args.num_users):
        x_train = np.array(dataset['train'].data[data_split['train'][i]], dtype=np.float32)
        y_train = np.array(dataset['train'].targets[data_split['train'][i]], dtype=np.int64)
        x_test = np.array(dataset['test'].data[data_split['test'][i]], dtype=np.float32)
        y_test = np.array(dataset['test'].targets[data_split['test'][i]], dtype=np.int64)
        train_set.append({'x': x_train, 'y': y_train})
        test_set.append({'x': x_test, 'y': y_test})
        l1, count1 = np.unique(y_train, return_counts=True)
        l2, count2 = np.unique(y_test, return_counts=True)
        clients_labels['train'].append({l: int(c) for l, c in zip(l1, count1)}) # json can't dump int64
        clients_labels['test'].append({l: int(c) for l, c in zip(l2, count2)})
        print(f'label types of client {i}: train: {l1}, test: {l2}')
        print(f'count of client {i}: train: {count1}, test: {count2}')


    save_file(args, num_classes, train_set, test_set, clients_labels, train_size, test_size)

    return test_set




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-noniid', type=str, default="2",
                        help="Experiment setting:\
                             0 means iid, 1 means pathological distribution,\
                             2 means dilliclet distribution",
                        choices=["0", "1", "2"])
    parser.add_argument('-num_users', type=int, default=100,
                        help="number of clients in FL")
    parser.add_argument('-shard', type=int, default=2,
                        help="hyper-parameter for pathological distribution")
    parser.add_argument('-alpha', type=float, default=0.5,
                        help="hyper-parameter for dilliclet distribution")
    args = parser.parse_args()
    args.classes_size = 10
    dir_path = "./data/cifar10/"
    args.config_path = dir_path + "noniid" + args.noniid + "/config.json"
    args.train_path = dir_path + "noniid" + args.noniid + "/proc_train/"
    args.test_path = dir_path + "noniid" + args.noniid + "/proc_test/"
    generate_cifar10(dir_path, args)


