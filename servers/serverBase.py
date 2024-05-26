
import copy
from torch import nn, optim
import json
import ray
from pathlib import Path
from utils.data_utils import *
from utils.logger import LoggerCreator
from warmup_scheduler import GradualWarmupScheduler

class ServerBase(object):
    def __init__(self, cfg, global_model, logger):
        self.PROCESS_NUM = cfg['PROCESS_NUM']

        self.mylogger = logger
        self.user_idx = None
        self.cfg = cfg
        self.current_round = 0
        self.dataset_name = cfg['dataset_name']
        self.lr = cfg['lr']
        self.num_clients = cfg['num_users']
        self.client_join_ratio = cfg['active_user_rate']
        self.num_active_clients = int(self.num_clients * self.client_join_ratio)
        self.clients_model_type = cfg['model_name']
        self.communication_round = cfg['g_rounds']
        self.evaluate_gap = cfg['evaluate_gap'] # test frequency
        self.global_model = global_model.cpu()
        self.clients_obj = None
        self.optimizer = None
        self.scheduler = None
        self.label_split, _ = read_clients_label(self.dataset_name, cfg)
        self.clients_train_loss = []
        self.clients_test_loss = []
        self.clients_test_acc = []
        self.clients_train_acc = []
        self.clients_model_rate_dist = []
        self.clients_model_rate_dist1 = []
        self.client_svdvals = []

    def test_comm_comput(self, all_model_rate, user_idx):
        """
        at each round, record the model rate of the participated clients
        :return:
        """
        # print(f"round={i}, setting={self.train_mode}, rate={np.unique(self.model_depth_rate)}")
        model_rate = [all_model_rate[user_idx[m]] for m in range(len(user_idx))]
        values, counts = np.unique(model_rate, return_counts=True)
        values, counts = values.tolist(), counts.tolist()
        self.clients_model_rate_dist.append(values)
        self.clients_model_rate_dist1.append(counts)


    def test_model(self, model, seleceted_all=True):
        """
        test the global model
        :param seleceted_all: if false, only calculate the average accuracy of the clients participating in the current round.
        :return: evaluation: containing mean test accuracy and mean loss.
        """

        if seleceted_all:
            with torch.no_grad():
                model.train(False)
                model_id = ray.put(copy.deepcopy(model))
                results = []
                for m in range(0, self.num_clients, self.PROCESS_NUM):
                    processes = []
                    for k in range(m, min(m + self.PROCESS_NUM, self.num_clients)):
                        processes.append(self.clients_obj[k % self.PROCESS_NUM].test_model_client.remote(k, [model_id]))
                # results store test set loss and accuracy for each clients
                    results.extend(ray.get(processes))
            return results
        else:
            with torch.no_grad():
                model.train(False)
                model_id = ray.put(copy.deepcopy(model))
                results = []
                for m in range(0, self.num_active_clients, self.PROCESS_NUM):
                    processes = []
                    for k in range(m, min(m + self.PROCESS_NUM, self.num_active_clients)):
                        processes.append(self.clients_obj[k % self.PROCESS_NUM].test_model_client.remote(k, [model_id]))
                    # results store test set loss and accuracy for each clients
                    results.extend(ray.get(processes))
            return results


    def save_results(self, json_path = "./results/json_file/",):
        if not os.path.exists(json_path):
            os.makedirs(json_path)
        logs = Path(json_path)
        filename = self.cfg['save_file_name'] + ".json"
        # filename = "pFedLMT_comm_comput" + ".json"
        # store_data = {
        #     'model_rate': self.clients_model_rate_dist,
        #     'model_rate_dist': self.clients_model_rate_dist1,
        # }
        store_data = {
            'test_acc': self.clients_test_acc,
            'train_acc': self.clients_train_acc,
            'test_loss': self.clients_test_loss,
            'train_loss': self.clients_train_loss,
            'model_rate': self.clients_model_rate_dist,
            'model_rate_dist': self.clients_model_rate_dist1,
            'svd_values': self.client_svdvals,
        }
        with (logs / filename).open('w', encoding='utf8') as f:
            json.dump(store_data, f)

    def save_model(self, the_model, path="./results/model_file/", current_round=None):
        if not os.path.exists(path):
            os.makedirs(path)
        if current_round is None:
            filename = path + self.cfg['save_file_name'] + ".pth"
        else:
            filename = path + "curT=" + str(current_round) + "_" + self.cfg['save_file_name'] + ".pth"
        torch.save(the_model.state_dict(), filename)

    def log(self, round, cost_time, train_evaluation, test_evaluation, svd_vals=None):
        """
        :param train_evaluation: a list, containing dictionaries (train loss and train accuracy) of each clients
        :param test_evaluation:  a list, containing dictionaries (test loss and test accuracy) of each clients
        :return:
        """
        train_info = [list(info.values()) for info in train_evaluation]
        test_info = [list(info.values()) for info in test_evaluation]
        train_loss, train_acc,  = np.mean(train_info, axis=0)
        test_loss, test_acc = np.mean(test_info, axis=0)
        self.clients_train_loss.append(train_loss)
        self.clients_test_loss.append(test_loss)
        self.clients_train_acc.append(train_acc)
        self.clients_test_acc.append(test_acc)
        self.client_svdvals.append(svd_vals)

        if self.cfg['dataset_name'] in ['wikitext2', 'wikitext103', 'PennTreebank']:
            self.mylogger.info(
                f'round = {round:d}, '
                f'cost = {cost_time:.4f}s, '
                f'mean train loss = {train_loss:.4f}, '
                f'mean test loss = {test_loss:.4f}, '
                f'mean train acc = {train_acc:.4f}, '
                f'mean test acc = {test_acc:.4f}, '
            )
        else:
            self.mylogger.info(
                f'round = {round:d}, '
                f'cost = {cost_time:.4f}s, '
                f'mean train acc = {train_acc:.4f}, '
                f'mean test acc = {test_acc:.4f}, '
            )

    def select_client(self, method=0):
        """
        Two methods to select clients:
        1. random manner to select clients
        2. robin manner to select clients (usually used in different privacy)
        """
        if method == 0:  # random select
            selected_clients_idx = list(
                np.random.choice(range(self.num_clients), int(self.client_join_ratio * self.num_clients), replace=False))
        else:  # robin manner to select
            shard_size = self.num_clients * self.client_join_ratio
            shard_num = np.ceil(1 / self.client_join_ratio)
            shard_idx = self.current_round % shard_num

            start = shard_idx * shard_size
            end = min((shard_idx + 1) * shard_size, self.num_clients)
            end = max(end, start + 1)
            selected_clients_idx = range(int(start), int(end))
            self.current_round += 1
        return selected_clients_idx




    def make_model_rate(self, current_round, mode='setting2'):
        """
        generate model rate for each clients according to width, selected from {1/16, 1/8, 1/4, 1/2, 1}
        'dynamic' means that the pruning rate changes on each clients per round
        'fix' means that the pruning rate does not change on each clients per round
        :param mode: setting 2 just like ProgFed, setting 1 is like HeteroFL, FedRolex and so on.
        :return:
        """
        if mode =='setting2':
            model_mode = self.cfg['model_mode'].split('-')
            each_client_rate = []
            mode_rate, proportion = [], []
            if current_round < self.cfg['meta_round']:
                m = model_mode[-1]
                mode_rate.append(self.cfg['model_split_rate'][m[0]])
                proportion.append(int(m[1:]))
            else:
                m = model_mode[0]
                mode_rate.append(self.cfg['model_split_rate'][m[0]])
                proportion.append(int(m[1:]))
            num_users_proportion = self.cfg['num_users'] // sum(proportion)
            for i in range(len(mode_rate)):
                each_client_rate += np.repeat(mode_rate[i], num_users_proportion * proportion[i]).tolist()
            each_client_rate = each_client_rate + [each_client_rate[-1] for _ in
                                                   range(self.cfg['num_users'] - len(each_client_rate))]
            each_client_rate = np.array(each_client_rate)
            return each_client_rate

        elif mode == 'setting3':
            model_mode = self.cfg['model_mode'].split('-')
            each_client_rate = []
            mode_rate, proportion = [], []
            if current_round < int(self.cfg['g_rounds'] * 0.125):
                m = model_mode[-1]
                mode_rate.append(self.cfg['model_split_rate'][m[0]])
                proportion.append(int(m[1:]))
            elif current_round < int(2 * self.cfg['g_rounds'] * 0.125):
                m = model_mode[-2]
                mode_rate.append(self.cfg['model_split_rate'][m[0]])
                proportion.append(int(m[1:]))
            elif current_round < int(3 * self.cfg['g_rounds'] * 0.125):
                m = model_mode[-3]
                mode_rate.append(self.cfg['model_split_rate'][m[0]])
                proportion.append(int(m[1:]))
            else:
                m = model_mode[0]
                mode_rate.append(self.cfg['model_split_rate'][m[0]])
                proportion.append(int(m[1:]))
            num_users_proportion = self.cfg['num_users'] // sum(proportion)
            for i in range(len(mode_rate)):
                each_client_rate += np.repeat(mode_rate[i], num_users_proportion * proportion[i]).tolist()
            each_client_rate = each_client_rate + [each_client_rate[-1] for _ in
                                                   range(self.cfg['num_users'] - len(each_client_rate))]
            each_client_rate = np.array(each_client_rate)
            return each_client_rate
        else:
            model_mode = self.cfg['model_mode'].split('-')
            each_client_rate = []
            if self.cfg['model_split_mode'] == 'dynamic':
                mode_rate, proportion = [], []
                for m in model_mode:
                    mode_rate.append(self.cfg['model_split_rate'][m[0]])
                    proportion.append(int(m[1:]))
                proportion = (np.array(proportion) / sum(proportion)).tolist()  # Convert to percentage
                rate_idx = torch.multinomial(torch.tensor(proportion), num_samples=self.cfg['num_users'],
                                             replacement=True).tolist()
                each_client_rate = np.array(mode_rate)[rate_idx]
                return each_client_rate

            elif self.cfg['model_split_mode'] == 'fix':
                mode_rate, proportion = [], []
                for m in model_mode:
                    mode_rate.append(self.cfg['model_split_rate'][m[0]])
                    proportion.append(int(m[1:]))
                num_users_proportion = self.cfg['num_users'] // sum(proportion)
                for i in range(len(mode_rate)):
                    each_client_rate += np.repeat(mode_rate[i], num_users_proportion * proportion[i]).tolist()
                each_client_rate = each_client_rate + [each_client_rate[-1] for _ in
                                                       range(self.cfg['num_users'] - len(each_client_rate))]
                each_client_rate = np.array(each_client_rate)
                return each_client_rate
            else:
                raise ValueError('Not valid model split mode!')

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
        elif self.cfg['scheduler_name'] == 'ExposnentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        elif self.cfg['scheduler_name'] == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg['g_rounds'],
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

        if self.cfg['warmup']:
            scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=self.cfg['warmupT'],
                                                      after_scheduler=scheduler)
            return scheduler_warmup
        else:
            return scheduler



