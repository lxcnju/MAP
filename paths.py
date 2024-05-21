import os

data_dir = r"C:\Workspace\work\datasets"
cur_dir = "./"

if not os.path.exists(data_dir):
    data_dir = "/home/lixc/datasets"

digits_fdir = os.path.join(data_dir, "Digits-five")
cifar_fdir = os.path.join(data_dir, "Cifar")
cinic_fdir = os.path.join(data_dir, "cinic10")
famnist_fdir = os.path.join(data_dir, "Fashion-MNIST")

save_dir = os.path.join(cur_dir, "logs")

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

cifar_fpaths = {
    "cifar10": {
        "train_fpaths": [
            os.path.join(cifar_fdir, "cifar10-train-part1.pkl"),
            os.path.join(cifar_fdir, "cifar10-train-part2.pkl"),
        ],
        "test_fpath": os.path.join(cifar_fdir, "cifar10-test.pkl")
    },
    "cifar100": {
        "train_fpaths": [
            os.path.join(cifar_fdir, "cifar100-train-part1.pkl"),
            os.path.join(cifar_fdir, "cifar100-train-part2.pkl"),
        ],
        "test_fpath": os.path.join(cifar_fdir, "cifar100-test.pkl")
    },
}
