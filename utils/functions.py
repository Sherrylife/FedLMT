import numpy as np
import random
import os
import errno
import torch


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # set the same hash value
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # set log information
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # set how to print error information


def check_exists(path):
    return os.path.exists(path)


def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


def save(input, path, protocol=2, mode='torch'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path, pickle_protocol=protocol)
    elif mode == 'numpy':
        np.save(path, input, allow_pickle=True)
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'numpy':
        return np.load(path, allow_pickle=True)
    else:
        raise ValueError('Not valid save mode')
    return



def process_control(args, cfg):
    cfg['overlap'] = args['overlap']
    cfg['weighting'] = args['weighting']
    cfg['algo_name'] = args['algo_name']
    cfg['g_rounds'] = args['g_rounds']
    cfg['l_epochs'] = args['l_epochs']
    cfg['dataset_name'] = args['dataset_name']
    cfg['model_name'] = args['model_name']
    cfg['lr'] = args['lr']
    cfg['optimizer_name'] = args['optimizer_name']
    cfg['seed'] = int(args['seed'])
    cfg['num_workers'] = args['num_workers']
    cfg['PROCESS_NUM'] = args['PROCESS_NUM']
    cfg['batch_size']['train'] = args['B']

    # FedLMT and pFedLMT
    cfg['meta_round'] = args['meta_round']
    cfg['ratio_LR'] = args['ratio_LR']
    cfg['decom_rule'] = args['decom_rule']
    cfg['train_decay'] = args['train_decay']
    cfg['coef_decay'] = args['coef_decay']

    # optimization method
    cfg['scheduler_name'] = args['scheduler_name']
    cfg['warmupT'] = args['warmupT']
    cfg['warmup'] = args['warmup']

    for k in cfg:
        cfg[k] = args[k]

    if args['schedule'] is not None:
        cfg['milestones'] = args['schedule']

    if args['control_name']:
        cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
            if args['control_name'] != 'None' else {}
    cfg['control_name'] = '_'.join([cfg['control'][k] for k in cfg['control']])
    cfg['exp_name'] = cfg['control_name']
    if cfg['dataset_name'] in ['mnist', 'fashion_mnist', 'cifar10', 'svhn']:
        cfg['classes_size'] = 10
    elif cfg['dataset_name'] in ['emnist']:
        cfg['classes_size'] = 62
    elif cfg['dataset_name'] in ['cifar100']:
        cfg['classes_size'] = 100
    elif cfg['dataset_name'] in ['tinyImagenet']:
        cfg['classes_size'] = 200
    elif cfg['dataset_name'] in ['wikitext2', 'wikitext103', 'PennTreebank']:
        cfg['num_tokens'] = 33279 # wikitext2
        cfg['bptt'] = 64
        cfg['mask_rate'] = 0.15
    # elif cfg['dataset_name'] in ['Stackoverflow']:
    #     # cfg['vocab'] = dataset['vocab']
    #     cfg['num_tokens'] = len(dataset['vocab'])
    elif cfg['dataset_name'] in ['gld']:
        cfg['classes_size'] = 2028
    else:
        NotImplementedError('Not valid dataset name')


    if cfg['algo_name'] == 'pFedLMT':
        cfg['model_LR_rate'] = {'a': 1.4, 'b': 0.6, 'c': 0.15, 'd': 0.01, }


    cfg['fed'] = int(cfg['control']['fed'])
    cfg['num_users'] = int(cfg['control']['num_users'])
    cfg['active_user_rate'] = float(cfg['control']['active_user_rate'])
    cfg['data_split_mode'] = cfg['control']['data_split_mode']
    cfg['model_split_mode'] = cfg['control']['model_split_mode']
    cfg['model_mode'] = cfg['control']['model_mode']
    cfg['norm'] = cfg['control']['norm']
    cfg['scale'] = bool(int(cfg['control']['scale']))
    cfg['mask'] = bool(int(cfg['control']['mask']))
    cfg['global_model_mode'] = cfg['model_mode'][0]
    cfg['global_model_rate'] = 1.0


    cfg['conv'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['resnet'] = {'hidden_size': [64, 128, 256, 512],
                     'resnet18_block': [2, 2, 2, 2],
                     'resnet34_blcok': [3, 4, 6, 3],
                     }
    if cfg['model_name'] == 'resnet18':
        cfg['max_module_num'] = 9
    elif cfg['model_name'] == 'resnet34':
        cfg['max_module_num'] = 17
    else:
        cfg['max_module_num'] = 9
    # cfg['resnet'] = {'hidden_size': [16, 16, 32, 64]}
    cfg['transformer'] = {'embedding_size': 128,
                          'num_heads': 8,
                          'hidden_size': 2048,
                          'num_layers': 3,
                          'dropout': 0.1}
    if cfg['dataset_name'] in ['mnist']:
        cfg['data_shape'] = [1, 28, 28]
        cfg['weight_decay'] = 1e-4
        cfg['factor'] = 0.1

    elif cfg['dataset_name'] in ['cifar10', 'cifar100', 'svhn']:
        cfg['data_shape'] = [3, 32, 32]
        cfg['min_lr'] = 1e-4
        cfg['factor'] = 0.25

    elif cfg['dataset_name'] in ['tinyImagenet']:
        cfg['data_shape'] = [3, 64, 64]
        cfg['min_lr'] = 1e-5
        cfg['factor'] = 0.25

    elif cfg['dataset_name'] in ['gld']:
        cfg['data_shape'] = [3, 92, 92]
        cfg['num_users'] = 1262
        cfg['active_user'] = 80
        cfg['min_lr'] = 5e-4
        cfg['weight_decay'] = 1e-3
        cfg['factor'] = 0.1

    elif cfg['dataset_name'] in ['wikitext2', 'wikitext103', 'PennTreebank']:
        cfg['weight_decay'] = 5e-4
        cfg['factor'] = 0.1
        cfg['bptt'] = 64
        cfg['mask_rate'] = 0.15

    elif cfg['dataset_name'] in ['Stackoverflow']:
        cfg['num_users'] = 342477
        cfg['active_user'] = 50
        cfg['weight_decay'] = 5e-4
        cfg['factor'] = 0.1
        cfg['bptt'] = 64
        cfg['mask_rate'] = 0.15
        cfg['num_users'] = 342477
        cfg['seq_length'] = 21
    else:
        raise ValueError('Not valid dataset name')
    return cfg