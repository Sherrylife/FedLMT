import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from datasets_utils import check, save_file, iid
import argparse
import LanguageModel as datasets

seed = 31
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

def batchify(dataset, num_users):
    num_batch = len(dataset) // num_users
    dataset.token = dataset.token.narrow(0, 0, num_batch * num_users)
    dataset.token = dataset.token.reshape(num_users, -1)
    return dataset


def generate_wikitext2(dir_path, args):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    train_path = os.path.dirname(args.train_path)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    test_path = os.path.dirname(args.test_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    dataset = {}
    dataset['train'] = datasets.WikiText2(root=dir_path + "rawdata", split='train')
    dataset['test'] = datasets.WikiText2(root=dir_path + "rawdata", split='test')

    vocab = dataset['train'].vocab
    num_tokens = len(dataset['train'].vocab)
    for split in dataset:
        dataset[split] = batchify(dataset[split], num_users=args.num_users)


    # The same data partition mode has been generated. No further action is required.
    # if check(args, vocab):
    #     return

    data_split = {}
    dataset_name = "wikitext2"
    train_size = len(dataset['train'])
    test_size = len(dataset['test'])
    if args.noniid == "0":
        data_split['train'], label_split = iid(dataset['train'], args.num_users, dataset_name)
        data_split['test'], label_split = iid(dataset['test'], args.num_users, dataset_name)


    train_set, test_set = [], []
    clients_labels = {'train': [], 'test': []}
    for i in range(args.num_users):
        x_train = np.array(dataset['train'].token[data_split['train'][i]], dtype=np.int64)
        y_train = np.array(dataset['train'].token[data_split['train'][i]], dtype=np.int64)
        x_test = np.array(dataset['test'].token[data_split['test'][i]], dtype=np.int64)
        y_test = np.array(dataset['test'].token[data_split['test'][i]], dtype=np.int64)

        train_set.append({'x': x_train, 'y': y_train})
        test_set.append({'x': x_test, 'y': y_test})
        l1, count1 = np.unique(y_train, return_counts=True)
        l2, count2 = np.unique(y_test, return_counts=True)
        clients_labels['train'].append({l: int(c) for l, c in zip(l1, count1)}) # json can't dump int64
        clients_labels['test'].append({l: int(c) for l, c in zip(l2, count2)})
        # print(f'label types of client {i}: train: {l1}, test: {l2}')
        # print(f'count of client {i}: train: {count1}, test: {count2}')


    save_file(args, num_tokens, train_set, test_set, clients_labels, train_size, test_size)

    return test_set




if __name__ == "__main__":
    """
    Note: only iid distribution can be selected.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-noniid', type=str, default="0",
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
    # args.classes_size = 10
    dir_path = "./data/wikitext2/"
    args.config_path = dir_path + "noniid" + args.noniid + "/config.json"
    args.train_path = dir_path + "noniid" + args.noniid + "/proc_train/"
    args.test_path = dir_path + "noniid" + args.noniid + "/proc_test/"
    generate_wikitext2(dir_path, args)


