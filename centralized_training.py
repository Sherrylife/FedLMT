import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from utils.logger import *
from utils.logger import LoggerCreator

def generate_data_transform(dataset_name):
    if 'cifar' in dataset_name:
        dataset_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    elif 'tiny_imagenet' in dataset_name:
        dataset_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    elif 'mnist' in dataset_name:
        dataset_transforms = None
    else:
        dataset_transforms = None

    return dataset_transforms


def save_json_results(store_data, filename):
    json_path = "./results/centralized_file/json_file/"
    if not os.path.exists(json_path):
        os.makedirs(json_path)
    logs = Path(json_path)
    with (logs / filename).open('w', encoding='utf8') as f:
        json.dump(store_data, f)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='centralized_training')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='dataset: [cifar10|ciafr100]')
    parser.add_argument('--model', default='resnet18', type=str,
                        help='resnet is supported currently')


    parser.add_argument('--optim_mode', type=str, default='sgd', choices=['adam', 'sgd'], help="optimizer mode")
    parser.add_argument('--sche_mode', type=str, default='CosineAnnealingLR', choices=['None', 'CosineAnnealingLR', 'MultiStepLR'],
                        help="schedule mode")

    parser.add_argument('--lr', default=0.1, type=float, help='learning rate ')

    parser.add_argument('--num_clients', type=int, default=5, help="number of clients in FL")
    parser.add_argument('--shard', type=int, default=2,
                        help="hyper-parameter for pathological distribution")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="hyper-parameter for dilliclet distribution")

    parser.add_argument('--epochs', default=200, type=int,
                        help='train epochs for decoders and auxiliary networks')
    parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--schedule', default=[100, 150], nargs='+', type=int)
    parser.add_argument('--devices', default=[4], nargs='+', type=int)

    parser.add_argument('--algo', type=str, default='FedAvg',
                        choices=['FedAvg', 'FedLMT', ], help="algorithm")

    # FedLMT hyper-parameter
    parser.add_argument('--train_decay', type=str, default='frobenius',
                        choices=['frobenius', 'kronecker', 'L2', 'L2-FD', 'L2-KD', 'FD-KD', 'none'], help="decay scheme")
    parser.add_argument('--coef_decay', default=0.001, type=float,
                        help='train decay coefficient')
    parser.add_argument('--coef_L2', default=0.001, type=float,
                        help='train decay coefficient for L2 decay')
    parser.add_argument('--coef_KD', default=0.001, type=float,
                        help='train decay coefficient for krobenius decay')
    parser.add_argument('--coef_FD', default=0.001, type=float,
                        help='train decay coefficient for frobenius decay')
    parser.add_argument('--init', type=str, default='none',
                        choices=['SI', 'none'], help="initialization scheme")
    parser.add_argument('--decom_rule', default=None, nargs='+', type=int)
    parser.add_argument('--LR_ratio', default=0.05, type=float,)


    args = vars(parser.parse_args())

    tree = lambda: defaultdict(tree)
    cfg = tree()
    cfg['seed'] = args['seed']
    cfg['devices'] = args['devices']
    cfg['model_name'] = args['model']
    cfg['dataset_name'] = args['dataset']
    cfg['noniid_type'] = '0'
    cfg['num_clients'] = args['num_clients']
    cfg['num_workers'] = args['num_workers']
    cfg['shard'] = args['shard']
    cfg['dir_alpha'] = args['alpha']
    cfg['lr'] = args['lr']
    cfg['epochs'] = args['epochs']
    cfg['batch_size'] = args['batch_size']
    cfg['optim_mode'] = args['optim_mode']
    if args['schedule'] is not None:
        cfg['milestones'] = args['schedule']
    cfg['momentum'] = 0.9
    cfg['global_model_rate'] = 1.0
    cfg['classes_size'] = 10 if cfg['dataset_name'] == 'cifar10' else 100
    cfg['data_shape'] = [3, 32, 32]
    # cfg['resnet']['hidden_size'] = [16, 16, 32, 64] # for resnet32
    cfg['resnet']['hidden_size'] = [64, 128, 256, 512]
    cfg['min_lr'] = 0.0001
    cfg['sche_mode'] = args['sche_mode']
    cfg['norm'] = 'bn'
    cfg['scale'] = False
    cfg['shuffle'] = True
    cfg['device'] = 'cuda'

    # FedLMT
    cfg['algo_name'] = args['algo']
    cfg['train_decay'] = args['train_decay']
    cfg['coef_decay'] = args['coef_decay']
    cfg['coef_L2'] = args['coef_L2']
    cfg['coef_KD'] = args['coef_KD']
    cfg['coef_FD'] = args['coef_FD']

    cfg['LR_ratio'] = args['LR_ratio']
    cfg['decom_rule'] = args['decom_rule']
    cfg['init'] = args['init']



    return cfg

cfg = parse_args()
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, cfg['devices']))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets
import torchvision.transforms as transforms
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import time
import numpy as np
import ujson
import pathlib
from utils.model_utils import generate_model
from utils.data_utils import MyDataset, read_data
from torch.utils.data import DataLoader


def read_all_train_data(cfg, partition_mode, is_train=True):
    """
    collect all training data used to train the model
    """
    all_train_data = []
    dataset_name = cfg['dataset_name']
    for client_id in range(cfg['num_clients']):
        client_train_data = read_data(dataset_name, client_id, partition_mode, is_train)
        x_train = torch.Tensor(client_train_data['x']).type(torch.float32)
        y_train = torch.Tensor(client_train_data['y']).type(torch.int64)
        all_train_data.extend([(x, y) for x, y in zip(x_train, y_train)])
    return all_train_data


def check_data(cfg):
    """
    check if the model is matched with the training data
    """
    config_path = f"./datasets/data/{cfg['dataset_name']}/noniid{cfg['noniid_type']}/config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_users'] == cfg['num_clients'] and \
                config['noniid_type'] == cfg['noniid_type'] and \
                (config['alpha'] == cfg['dir_alpha'] or config['shard'] == cfg['shard']):
            print("The model is matched with the training data")
        else:
            ValueError()

if __name__ == '__main__':

    # save path
    filename = f"central_{cfg['dataset_name']}_{cfg['algo_name']}_{cfg['model_name']}_decom={cfg['decom_rule']}_LR={cfg['LR_ratio']}_decay={cfg['train_decay']}_coef={cfg['coef_decay']}_SI={cfg['init']}" + \
               f"_coeL2={cfg['coef_L2']}_coeKD={cfg['coef_KD']}_coeFD={cfg['coef_FD']}_epochs={cfg['epochs']}_B={cfg['batch_size']}_{cfg['optim_mode']}_lr={cfg['lr']}_{cfg['sche_mode']}_seed={cfg['seed']}"
    logs_path = "./results/centralized_file/log_file/"
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    log_path_name = os.path.join(logs_path, filename)
    mylogger = LoggerCreator.create_logger(log_path=log_path_name, logging_name="FedLMT",
                                           level=logging.INFO)
    mylogger.info(' '.join(f' \'{k}\': {v}, ' for k, v in cfg.items()))

    set_seed(cfg['seed'])

    model = generate_model(cfg['model_name'], cfg['dataset_name'], cfg=cfg, LR_ratio=cfg['LR_ratio'])

    # experimental setting
    dataset_name = cfg['dataset_name']
    lr = cfg['lr']
    min_lr = cfg['min_lr']
    train_epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    device = cfg['device']
    optim_mode = cfg['optim_mode']
    schedule_mode = cfg['sche_mode']
    criterion = torch.nn.CrossEntropyLoss().to(device)

    if optim_mode in ['sgd']:
        # optimizers_decoder[model_index] = torch.optim.SGD(decoders[model_index].parameters(), lr=lr, momentum=0.9)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0, eps=1e-8, betas=(0.9, 0.999))

    if schedule_mode in ['CosineAnnealingLR']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs, eta_min=cfg['min_lr'])
    elif schedule_mode in ['MultiStepLR']:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])

    # load train data and test data
    dir_path = f"./datasets/data/{cfg['dataset_name']}/"
    if cfg['dataset_name'] == 'cifar10':
        train_data = datasets.CIFAR10(root=dir_path + "rawdata", train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        test_data = datasets.CIFAR10(root=dir_path + "rawdata", train=False, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    elif cfg['dataset_name'] == 'cifar100':
        train_data = datasets.CIFAR100(root=dir_path + "rawdata", train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))]))
        test_data = datasets.CIFAR100(root=dir_path + "rawdata", train=False, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))]))

    train_loader = DataLoader(dataset=train_data, shuffle=cfg['shuffle'],
                             batch_size=batch_size, pin_memory=True, num_workers=cfg['num_workers'])
    test_loader = DataLoader(dataset=test_data, shuffle=False,
                             batch_size=batch_size, pin_memory=True, num_workers=cfg['num_workers'])

    train_loss_logs, train_acc_logs = [], []
    test_loss_logs, test_acc_logs = [], []
    svdvals_logs = []

    start_time = time.time()

    # this zero gradient update is needed to avoid a warning message, https://github.com/ildoonet/pytorch-gradual-warmup-lr/issues/8
    optimizer.zero_grad()
    optimizer.step()

    for epoch in range(train_epochs):
        model.train()

        t0 = time.time()

        train_loss_log, train_acc_log, train_total_log = [], 0., 0.

        svdvals_logs.append(model.cal_smallest_svdvals())
        mylogger.info(
            f'round = {epoch:d}, '
            f'svdvals = {svdvals_logs[-1]}, '
        )

        for idx, data in enumerate(train_loader):
            inputs, targets = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if cfg['train_decay'] == 'frobenius':
                loss += cfg['coef_decay'] * model.frobenius_decay()
            elif cfg['train_decay'] == 'kronecker':
                loss += cfg['coef_decay'] * model.kronecker_decay()
            elif cfg['train_decay'] == 'L2':
                loss += cfg['coef_decay'] * model.L2_decay()
            elif cfg['train_decay'] == 'L2-KD':
                loss += (cfg['coef_L2'] * model.L2_decay() + cfg['coef_KD'] * model.kronecker_decay())
            elif cfg['train_decay'] == 'L2-FD':
                loss += (cfg['coef_L2'] * model.L2_decay() + cfg['coef_FD'] * model.frobenius_decay())
            elif cfg['train_decay'] == 'FD-KD':
                loss += (cfg['coef_KD'] * model.kronecker_decay() + cfg['coef_FD'] * model.frobenius_decay())

            loss.backward()
            optimizer.step()

            # todo: save train loss and train accuracy
            train_loss_log.append(loss.item())
            _, predicts = torch.max(outputs, 1)
            train_acc_log += (predicts == targets).sum().item()
            train_total_log += len(targets)

        train_loss_log = np.mean(train_loss_log).item()
        train_acc_log = train_acc_log / train_total_log
        train_loss_logs.append(train_loss_log)
        train_acc_logs.append(train_acc_log)

        # --------------------test model----------------------------
        test_loss_log, test_acc_log, test_total_log = [], 0., 0.
        model.eval()
        for idx, data in enumerate(test_loader):
            inputs, targets = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            test_loss = criterion(outputs, targets)
            # todo: save train loss and train accuracy
            test_loss_log.append(test_loss.item())
            _, predicts = torch.max(outputs, 1)
            test_acc_log += (predicts == targets).sum().item()
            test_total_log += len(targets)

        test_loss_log = np.mean(test_loss_log).item()
        test_acc_log = test_acc_log / test_total_log

        test_loss_logs.append(test_loss_log)
        test_acc_logs.append(test_acc_log)

        scheduler.step()


        t1 = time.time()
        mylogger.info(
            f'round = {epoch:d}, '
            f'cost = {(t1-t0):.4f}s, '
            f'train_loss = {train_loss_log:.4f}, '
            f'test_loss = {test_loss_log:.4f}, '
            f'train_acc = {train_acc_log:.4f}, '
            f'test_acc = {test_acc_log:.4f}'
        )
    end_time = time.time()

    save_data = {
        'train_acc': train_acc_logs,
        'test_acc': test_acc_logs,
        'train_loss': train_loss_logs,
        'test_loss': test_loss_logs,
        'svd_values': svdvals_logs,
    }
    save_json_results(save_data, filename=filename + ".json")

    mylogger.info(f"\nTotal time cost: {round(end_time-start_time, 2)} s.")

