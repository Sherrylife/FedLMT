import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder, ImageFolder, VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import extract_archive, check_integrity, download_url, verify_str_arg
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


class TinyImageNet(VisionDataset):
    """`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            split (string, optional): The dataset split, supports ``train``, or ``val``.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'tinyImagenet/rawdata/tiny-imagenet-200/'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(TinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)

        self.dataset_path = os.path.join(root, self.base_folder)
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val",))
        self.transform = transform
        self.target_transform = target_transform

        # if self._check_integrity():
        #     print('Files already downloaded and verified.')
        # elif download:
        #     self._download()
        # else:
        #     raise RuntimeError(
        #         'Dataset not found. You can use download=True to download it.')
        if not os.path.isdir(self.dataset_path):
            print('Extracting...')
            extract_archive(os.path.join(root, self.filename))

        _, class_to_idx = find_classes(os.path.join(self.dataset_path, 'wnids.txt'))

        self.data = make_dataset(self.root, self.base_folder, self.split, class_to_idx)

    def _download(self):
        print('Downloading...')
        download_url(self.url, root=self.root, filename=self.filename)
        print('Extracting...')
        extract_archive(os.path.join(self.root, self.filename))

    def _check_integrity(self):
        return check_integrity(os.path.join(self.root, self.filename), self.md5)

    def __getitem__(self, index):
        img_path, target = self.data[index]
        image = self.loader(img_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.data)


def find_classes(class_file):
    with open(class_file) as r:
        classes = list(map(lambda s: s.strip(), r.readlines()))

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


def make_dataset(root, base_folder, dirname, class_to_idx):
    images = []
    dir_path = os.path.join(root, base_folder, dirname)

    if dirname == 'train':
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, 'images')
                for imgname in sorted(os.listdir(cls_imgs_path)):
                    path = os.path.join(cls_imgs_path, imgname)
                    item = (path, class_to_idx[fname])
                    images.append(item)
    else:
        imgs_path = os.path.join(dir_path, 'images')
        imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

        with open(imgs_annotations) as r:
            data_info = map(lambda s: s.split('\t'), r.readlines())

        cls_map = {line_data[0]: line_data[1] for line_data in data_info}

        for imgname in sorted(os.listdir(imgs_path)):
            path = os.path.join(imgs_path, imgname)
            item = (path, class_to_idx[cls_map[imgname]])
            images.append(item)

    return images



def generate_tinyImagenet(dir_path, args):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    train_path = os.path.dirname(args.train_path)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    test_path = os.path.dirname(args.test_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    transform_tiny_imagenet_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_tiny_imagenet_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = {}
    dataset['train'] = TinyImageNet(root='./data/', split='train', transform=transform_tiny_imagenet_train, download=False)
    dataset['test'] = TinyImageNet(root='./data/', split='val', transform=transform_tiny_imagenet_test, download=False)

    num_classes = 200

    # The same data partition mode has been generated. No further action is required.
    # if check(args, num_classes):
    #     return

    trainloader = torch.utils.data.DataLoader(
        dataset['train'], batch_size=len(dataset['train']), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        dataset['test'], batch_size=len(dataset['test']), shuffle=False)

    # Extended data dimension
    for _, train_data in enumerate(trainloader, 0):
        dataset['train'].data, dataset['train'].targets = train_data
    for _, test_data in enumerate(testloader, 0):
        dataset['test'].data, dataset['test'].targets = test_data

    data_split = {}
    dataset_name = "tinyImagenet"
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
    parser.add_argument('-shard', type=int, default=10,
                        help="hyper-parameter for pathological distribution")
    parser.add_argument('-alpha', type=float, default=0.5,
                        help="hyper-parameter for dilliclet distribution")
    args = parser.parse_args()
    args.classes_size = 200
    dir_path = "./data/tinyImagenet/"
    args.config_path = dir_path + "noniid" + args.noniid + "/config.json"
    args.train_path = dir_path + "noniid" + args.noniid + "/proc_train/"
    args.test_path = dir_path + "noniid" + args.noniid + "/proc_test/"
    generate_tinyImagenet(dir_path, args)


