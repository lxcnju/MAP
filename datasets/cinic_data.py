import os
import copy
import numpy as np
from collections import Counter

import torch
from torch.utils import data
from torchvision import transforms

from paths import cinic_fdir

from utils import load_pickle


def load_cinic_data(dataset, combine=True):
    """ Load Cinic10 Data from pickle data
    params:
    @dataset: "cinic10"
    return:
    @xs: numpy.array, (n, c, w, h)
    @ys: numpy.array, (n, ), 0-9
    """
    train_fpath = os.path.join(cinic_fdir, "train.pkl")
    train_xs, train_ys = load_pickle(train_fpath)

    test_fpath = os.path.join(cinic_fdir, "test.pkl")
    test_xs, test_ys = load_pickle(test_fpath)

    print(Counter(train_ys))
    print(Counter(test_ys))

    if combine:
        xs = np.concatenate([train_xs, test_xs], axis=0)
        ys = np.concatenate([train_ys, test_ys], axis=0)
        return xs, ys
    else:
        return train_xs, train_ys, test_xs, test_ys


class CinicDataset(data.Dataset):
    def __init__(self, xs, ys, is_train=True):
        self.xs = copy.deepcopy(xs)
        self.ys = copy.deepcopy(ys)

        if is_train is True:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                )
            ])

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        img = self.xs[index]
        label = self.ys[index]

        img = img.transpose((1, 2, 0)).astype(np.uint8)
        img = self.transform(img)

        img = torch.FloatTensor(img)
        label = torch.LongTensor([label])[0]
        return img, label
