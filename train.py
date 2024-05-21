import os
import random
from collections import namedtuple
import numpy as np

import torch

from datasets.feddata import FedData

from algorithms.fedavg import FedAvg
from algorithms.fedreg import FedReg
from algorithms.scaffold import Scaffold
from algorithms.feddf import FedDF

from algorithms.fedopt import FedOpt
from algorithms.fednova import FedNova
from algorithms.fedaws import FedAws
from algorithms.moon import MOON
from algorithms.feddyn import FedDyn

from algorithms.flda import FLDA
from algorithms.feddml import FedDML
from algorithms.fedrep import FedRep
from algorithms.pfedme import pFedMe
from algorithms.perfedavg import PerFedAvg
from algorithms.fedproto import FedProto
from algorithms.ditto import Ditto
from algorithms.fedrod import FedROD

from algorithms.fedrs import FedRS
from algorithms.fedphp import FedPHP

from algorithms.scaffoldrs import ScaffoldRS
from algorithms.dittors import DittoRS

from algorithms.fedap import FedAP
from algorithms.scaffoldap import ScaffoldAP

from networks.basic_classify_nets import get_basic_net
from networks.basic_classify_nets import FLDANet
from networks.basic_classify_nets import FedDMLNet
from networks.basic_classify_nets import FedRODNet

from paths import save_dir
from config import default_param_dicts
from utils import set_gpu

torch.set_default_tensor_type(torch.FloatTensor)


def construct_model(args):
    try:
        input_size = args.input_size
    except Exception:
        input_size = None

    try:
        input_channel = args.input_channel
    except Exception:
        input_channel = None

    model = get_basic_net(
        net=args.net,
        n_classes=args.n_classes,
        input_size=input_size,
        input_channel=input_channel,
        bias=False
    )

    if args.algo == "flda":
        model = FLDANet(model)
    elif args.algo == "feddml":
        model = FedDMLNet(model)
    elif args.algo == "fedrod":
        model = FedRODNet(model)
    else:
        pass

    return model


def construct_algo(args):
    if args.algo == "fedavg":
        FedAlgo = FedAvg
    elif args.algo == "fedprox":
        FedAlgo = FedReg
    elif args.algo == "fedmmd":
        FedAlgo = FedReg
    elif args.algo == "scaffold":
        FedAlgo = Scaffold
    elif args.algo == "feddf":
        FedAlgo = FedDF
    elif args.algo == "fedopt":
        FedAlgo = FedOpt
    elif args.algo == "fednova":
        FedAlgo = FedNova
    elif args.algo == "fedaws":
        FedAlgo = FedAws
    elif args.algo == "moon":
        FedAlgo = MOON
    elif args.algo == "feddyn":
        FedAlgo = FedDyn
    elif args.algo == "flda":
        FedAlgo = FLDA
    elif args.algo == "feddml":
        FedAlgo = FedDML
    elif args.algo == "fedrep":
        FedAlgo = FedRep
    elif args.algo == "pfedme":
        FedAlgo = pFedMe
    elif args.algo == "perfedavg":
        FedAlgo = PerFedAvg
    elif args.algo == "fedproto":
        FedAlgo = FedProto
    elif args.algo == "ditto":
        FedAlgo = Ditto
    elif args.algo == "fedrod":
        FedAlgo = FedROD
    elif args.algo == "fedrs":
        FedAlgo = FedRS
    elif args.algo == "fedphp":
        FedAlgo = FedPHP
    elif args.algo == "scaffoldrs":
        FedAlgo = ScaffoldRS
    elif args.algo == "dittors":
        FedAlgo = DittoRS
    elif args.algo == "fedap":
        FedAlgo = FedAP
    elif args.algo == "scaffoldap":
        FedAlgo = ScaffoldAP
    else:
        raise ValueError("No such fed algo:{}".format(args.algo))
    return FedAlgo


def get_hypers(algo):
    if algo == "fedavg":
        hypers = {
            "cnt": 3,
            "none": ["none", "none", "none"]
        }
    elif algo == "fedprox":
        hypers = {
            "cnt": 3,
            "reg_way": ["fedprox", "fedprox", "fedprox"],
            "reg_lamb": [1e-4, 1e-3, 1e-2]
        }
    elif algo == "fedmmd":
        hypers = {
            "cnt": 3,
            "reg_way": ["fedmmd", "fedmmd", "fedmmd"],
            "reg_lamb": [1e-3, 1e-2, 1e-1]
        }
    elif algo == "scaffold":
        hypers = {
            "cnt": 3,
            "glo_lr": [0.5, 0.75, 0.9]
        }
    elif algo == "feddf":
        hypers = {
            "cnt": 3,
            "df_lr": [1e-2, 1e-3, 1e-4],
            "df_steps": [200, 300, 500],
        }
    elif algo == "fedopt":
        hypers = {
            "cnt": 3,
            "glo_optimizer": ["SGD", "SGD", "SGD"],
            "glo_lr": [1.0, 0.1, 0.05],
        }
    elif algo == "fednova":
        hypers = {
            "cnt": 3,
            "gmf": [0.9, 0.5, 0.3],
            "prox_mu": [1e-4, 1e-3, 1e-2],
        }
    elif algo == "fedaws":
        hypers = {
            "cnt": 3,
            "margin": [0.9, 0.8, 0.5],
            "aws_steps": [20, 30, 50],
            "aws_lr": [1.0, 0.1, 0.01],
        }
    elif algo == "moon":
        hypers = {
            "cnt": 3,
            "reg_lamb": [1e-3, 1e-2, 1e-1]
        }
    elif algo == "feddyn":
        hypers = {
            "cnt": 3,
            "reg_lamb": [1e-3, 1e-2, 1e-1]
        }
    elif algo == "flda":
        hypers = {
            "cnt": 3,
            "none": ["none", "none", "none"]
        }
    elif algo == "feddml":
        hypers = {
            "cnt": 3,
            "kt_tau": [1.0, 4.0, 8.0],
            "kt_alpha": [0.25, 0.5, 0.75],
            "kt_beta": [0.75, 0.5, 0.25],
        }
    elif algo == "fedrep":
        hypers = {
            "cnt": 3,
            "none": ["none", "none", "none"]
        }
    elif algo == "pfedme":
        hypers = {
            "cnt": 3,
            "reg_lamb": [0.1, 0.01, 0.001],
            "alpha": [0.9, 0.75, 0.5],
            "k_step": [5, 10, 15],
            "beta": [1.0, 1.0, 1.0],
        }
    elif algo == "perfedavg":
        hypers = {
            "cnt": 3,
            "meta_lr": [0.1, 0.05, 0.01],
        }
    elif algo == "fedproto":
        hypers = {
            "cnt": 3,
            "reg_lamb": [0.1, 0.01, 0.001],
        }
    elif algo == "ditto":
        hypers = {
            "cnt": 3,
            "reg_lamb": [0.1, 0.01, 0.001],
        }
    elif algo == "fedrod":
        hypers = {
            "cnt": 3,
            "bal_gamma": [1.0, 0.5, 0.25],
        }
    elif algo == "fedrs":
        hypers = {
            "cnt": 3,
            "alpha": [0.9, 0.5, 0.1],
        }
    elif algo == "fedphp":
        hypers = {
            "cnt": 3,
            "reg_way": ["KD", "MMD", "MMD"],
            "reg_lamb": [0.1, 0.1, 0.01],
        }
    elif algo == "scaffoldrs":
        hypers = {
            "cnt": 3,
            "alpha": [0.1, 0.9, 0.5],
            "glo_lr": [0.5, 0.75, 0.9]
        }
    elif algo == "dittors":
        hypers = {
            "cnt": 3,
            "alpha": [0.1, 0.9, 0.5],
            "reg_lamb": [0.1, 0.01, 0.001],
        }
    elif algo == "fedap":
        hypers = {
            "cnt": 3,
            "reg_way": ["KD", "KD", "MMD"],
            "reg_lamb": [0.1, 0.01, 0.01],
            "alpha": [0.5, 0.9, 0.5],
        }
    elif algo == "scaffoldap":
        hypers = {
            "cnt": 3,
            "reg_way": ["KD", "KD", "MMD"],
            "reg_lamb": [0.1, 0.01, 0.01],
            "alpha": [0.5, 0.9, 0.5],
            "glo_lr": [0.5, 0.9, 0.5]
        }
    else:
        raise ValueError("No such fed algo:{}".format(algo))
    return hypers


def main_federated(para_dict):
    print(para_dict)
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    # DataSets
    try:
        n_clients = args.n_clients
    except Exception:
        n_clients = None

    try:
        nc_per_client = args.nc_per_client
    except Exception:
        nc_per_client = None

    try:
        dir_alpha = args.dir_alpha
    except Exception:
        dir_alpha = None

    feddata = FedData(
        dataset=args.dataset,
        split=args.split,
        n_clients=n_clients,
        nc_per_client=nc_per_client,
        dir_alpha=dir_alpha,
        n_max_sam=args.n_max_sam,
        ln_sigma=args.ln_sigma
    )
    csets, gset = feddata.construct()

    try:
        nc = int(args.dset_ratio * len(csets))
        clients = list(csets.keys())
        sam_clients = np.random.choice(
            clients, nc, replace=False
        )
        csets = {
            c: info for c, info in csets.items() if c in sam_clients
        }

        n_test = int(args.dset_ratio * len(gset.xs))
        inds = np.random.permutation(len(gset.xs))
        gset.xs = gset.xs[inds[0:n_test]]
        gset.ys = gset.ys[inds[0:n_test]]

    except Exception:
        pass

    feddata.print_info(csets, gset)

    # Model
    model = construct_model(args)
    print(model)
    print([name for name, _ in model.named_parameters()])
    n_params = sum([
        param.numel() for param in model.parameters()
    ])
    print("Total number of parameters : {}".format(n_params))

    if args.cuda:
        torch.backends.cudnn.benchmark = True
        model = model.cuda()

    FedAlgo = construct_algo(args)
    algo = FedAlgo(
        csets=csets,
        gset=gset,
        model=model,
        args=args
    )
    algo.train()

    fpath = os.path.join(
        save_dir, args.fname
    )
    algo.save_logs(fpath)
    print(algo.logs)


def main(algo):
    datasets = [
        "famnist", "famnist",
        "cifar10", "cifar10",
        "cifar100", "cifar100",
        "cinic10", "cinic10"
    ]
    nets = [
        "MLPNet", "LeNet",
        "TFCNN", "VGG8",
        "VGG8-BN", "VGG11",
        "ResNet20", "ResNet56"
    ]
    all_local_epochs = [
        5, 5, 10, 10,
        20, 20, 40, 40
    ]
    lrs = [
        0.03, 0.03, 0.03, 0.03,
        0.03, 0.03, 0.05, 0.05
    ]

    hypers = get_hypers(algo)

    for d in [0, 1, 2, 3, 4, 5, 6, 7]:
        dataset = datasets[d]
        net = nets[d]
        local_epochs = all_local_epochs[d]
        n_clients = 100
        c_ratio = 0.2
        max_round = 150
        lr = lrs[d]

        for j in range(hypers["cnt"]):
            para_dict = {}
            for k, vs in default_param_dicts[dataset].items():
                para_dict[k] = random.choice(vs)

            para_dict["algo"] = algo
            para_dict["dataset"] = dataset
            para_dict["net"] = net
            para_dict["split"] = "map"
            para_dict["ln_sigma"] = 0.3
            para_dict["n_clients"] = n_clients
            para_dict["c_ratio"] = c_ratio
            para_dict["local_epochs"] = local_epochs
            para_dict["max_round"] = max_round
            para_dict["test_round"] = 2
            para_dict["lr"] = lr

            for key, values in hypers.items():
                if key == "cnt":
                    continue
                else:
                    para_dict[key] = values[j]

            if algo in [
                "fedap", "scaffoldap", "fedproto", "ditto", "fedrod",
                "scaffoldrs", "dittors"
            ]:
                para_dict["fname"] = "{}-{}-{}.log".format(
                    dataset, net, algo
                )
            else:
                para_dict["fname"] = "{}-{}.log".format(
                    dataset, net
                )

            main_federated(para_dict)


if __name__ == "__main__":
    algos = [
        "fedavg", "fedprox", "fedmmd", "scaffold", "feddf",
        "fedopt", "fednova", "fedaws", "moon", "feddyn",
        "fedrep", "flda", "pfedme", "perfedavg",
        "fedproto", "ditto", "fedrod", "fedrs", "fedphp",
        "scaffoldrs", "dittors", "fedap", "scaffoldap"
    ]

    set_gpu("1")
    for algo in algos:
        main(algo)
