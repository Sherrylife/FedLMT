
import yaml
import os
import time
import random
import numpy as np
import argparse
import ray
from servers.serverFedAvg import FedAvg
from servers.serverFedLMT import FedLMT
from servers.serverpFedLMT import pFedLMT
from utils.data_utils import read_experiment_setting
from utils.model_utils import generate_model
from utils.functions import *
from utils.logger import *

def parse_args(cfg):
    parser = argparse.ArgumentParser(description='cfg')
    for k in cfg:
        exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
    parser.add_argument('--control_name', default=None, type=str)
    parser.add_argument('--dataset_name', default="cifar10", choices=[
       "mnist", "cifar10", "cifar100", "tinyImagenet", "wikitext2", "svhn",
    ], type=str)

    parser.add_argument('--model_name', default="resnet18", choices=["resnet18",
                                     "transformer", "hyper_transformer", ], type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--devices', default=None, nargs='+', type=int)
    parser.add_argument('--num_workers', default=0, type=int, help='number of data loading workers (default: 0)')
    parser.add_argument('--algo_name', default='FedLMT', choices=['FedAvg', 'FedLMT', 'pFedLMT'],
                        type=str)

    parser.add_argument('--weighting', default='avg', type=str)
    parser.add_argument('--g_rounds', default=None, type=int)
    parser.add_argument('--l_epochs', default=None, type=int)
    parser.add_argument('--PROCESS_NUM', default=10, type=int)
    parser.add_argument('--overlap', default=None, type=float)
    parser.add_argument('--B', default=64, type=int, help='train batch size')

    # optimizer setting
    parser.add_argument('--optimizer_name', default='SGD', choices=['SGD', 'Adam', 'Adamax', 'RMSprop'], type=str)
    parser.add_argument('--scheduler_name', default='CosineAnnealingLR', choices=['None', 'StepLR', 'MultiStepLR',
                    'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau', 'CyclicLR'], type=str)
    parser.add_argument('--schedule', default=None, nargs='+', type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--warmupT', default=10, type=int)


    # FedLMT and pFedLMT
    parser.add_argument('--ratio_LR', default=0.2, type=float, help='Low rank ratio, hyper parameter for '
                                                                    'FedLMT and pFedLMT')
    parser.add_argument('--decom_rule', default=None, nargs='+', type=int,
                        help='a hyper parameter for the technique of "hybrid model architecture", '
                             'where the first K layers in the model are not decomposed with low-rank technique. '
                             'decom_rule is a 2-tuple like (block_index, layer_index). '
                             'For resnet18, block_index is selected from [0,1,2,3] and layer_index is selected from [0,1].'
                             'In resnet18 model which has 18 layers, the 1st layer and the 18th layer'
                             'will be not decomposed, and the value of decom_rule can be: '
                             '`0 0`: which means we start decomposing resnet18 at the first residual block '
                             'in the first blocks (actually the 2nd layer in the original model), '
                             '`0 1`: which means we start decomposing resnet18 at the second residual block '
                             'in the first blocks (actually the 4th layer in the original model), '
                             '`1 0`: which means we start decomposing resnet18 at the first residual block '
                             'in the second blocks (actually the 6th layer in the original model), '
                             '`1 1`: which means we start decomposing resnet18 at the second residual block '
                             'in the second blocks (actually the 8th layer in the original model), '
                             '...'
                             '`3 0`: which means we start decomposing resnet18 at the first residual block '
                             'in the fourth blocks (actually the 14th layer in the original model), '
                             '`3 1`: which means we start decomposing resnet18 at the second residual block '
                             'in the fourth blocks (actually the 16th layer in the original model), '
                        )
    parser.add_argument('--train_decay', type=str, default='none',
                        choices=['frobenius', 'kronecker', 'L2', 'L2-FD', 'L2-KD', 'FD-KD', 'none'], help="decay scheme")
    parser.add_argument('--coef_decay', default=0.001, type=float,
                        help='train decay coefficient')
    parser.add_argument('--meta_round', default=0, type=int, help="the hyper parameter is used in pFedLMT which"
                            "splits the whole training process into two stages. In stage 1, all clients use the same low-rank"
                            "ratio; In stage 2, different clients use different low-rank ratio according to their own personalized"
                            "need. `meta_round` means the duration of stage 1. Default value is 0, which means that we don't execute"
                                                                  "stage 1 in pFedLMT.")
    args = vars(parser.parse_args())
    args = read_experiment_setting(args['dataset_name'], args)
    cfg = process_control(args, cfg) # update cfg according to args
    return args, cfg

# print(os.path.dirname(__file__))
# with open('./config.yml', 'r') as f:
# print(os.getcwd())
with open(os.getcwd()+'/config.yml', 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
args, cfg = parse_args(cfg)
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args['devices']))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.backends.cudnn as cudnn

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ray.init(num_gpus=len(args['devices']),
         ignore_reinit_error=True,
         # local_mode=True, # run sequentially, used for debug
         )



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def select_algorithm(algo_name, global_model, mylogger):
    if algo_name == 'FedAvg':
        return FedAvg(cfg, global_model, mylogger)
    elif algo_name == 'FedLMT':
        return FedLMT(cfg, global_model, mylogger)
    elif algo_name == 'pFedLMT':
        return pFedLMT(cfg, global_model, mylogger)
    else:
        NotImplementedError("The algorithm name is not proper")

def run_experiment(seed, cfg, mylogger):
    set_seed(seed)
    global_model = generate_model(
        model_name=cfg['model_name'],
        model_rate=1.0,
        depth_rate=4,
        cfg=cfg,
        LR_ratio=cfg['ratio_LR']
    )
    server = select_algorithm(cfg['algo_name'], global_model, mylogger)
    server.train()


def main():

    log_path = './results/log_file'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if args['devices'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in args['devices']])

    seeds = list(range(cfg['seed'], cfg['seed'] + cfg['num_experiments']))
    for i, seed in enumerate(seeds):
        start_time = time.time()
        filename = f"{cfg['dataset_name']}_{cfg['algo_name']}_large={cfg['model_name']}_LRratio={cfg['ratio_LR']}_" + \
                   f"Tg={cfg['g_rounds']}_E={cfg['l_epochs']}_B={cfg['batch_size']['train']}_{cfg['optimizer_name']}_lr={cfg['lr']}_seed={seed}_" + \
                   f"Decom={cfg['decom_rule']}_Decay={cfg['train_decay']}_{cfg['coef_decay']}_{cfg['control_name']}"
        cfg['save_file_name'] = filename
        log_path_name = os.path.join(log_path, filename)
        mylogger = LoggerCreator.create_logger(log_path=log_path_name, logging_name="FedLMT",
                                               level=logging.INFO)
        mylogger.info(' '.join(f' \'{k}\': {v}, ' for k, v in cfg.items()))

        model_tag_list = [str(seeds[i]), cfg['dataset_name'], cfg['subset'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        print('Seed: {}'.format(seed))
        run_experiment(seed, cfg, mylogger)
        end_time = time.time()
        mylogger.info(f"\nTotal time cost: {round(end_time-start_time, 2)} s.")

if __name__ == '__main__':
    main()

