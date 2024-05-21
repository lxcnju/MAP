import numpy as np

from collections import Counter

from datasets.mnist_data import load_mnist_data, MnistDataset
from datasets.cifar_data import load_cifar_data, CifarDataset
from datasets.digits_data import load_digits_data, DigitsDataset
from datasets.famnist_data import load_famnist_data, FaMnistDataset
from datasets.cinic_data import load_cinic_data, CinicDataset

np.random.seed(0)


class FedData():
    """ Federated Datasets: support different scenes and split ways
    params:
    @dataset: "mnist", "cifar10", "cifar100", "cinic10",
              "digits-five", "femnist", "speechcommands",
              "sent140", "shakespeare", "ohsumed"
    @split: "label", "user", None
        if split by "user", split each user to a client;
        if split by "label", split to n_clients with samples from several class
    @n_clients: int, None
        if split by "user", is Num.users;
        if split by "label", it is pre-defined;
    @nc_per_client: int, None
        number of classes per client, only for split="label";
    @n_client_perc: int, None
        number of clients per class, only for split="label" and dataset="sa";
    @dir_alpha: float > 0.0, 1.0
        Dir(alpha), the larger the uniform, 0.1, 1.0, 10.0
    @n_max_sam: int, None
        max number of samples per client, for low-resource learning;
    @split_sent140_way: str
        the way to split sent140
    """

    def __init__(
        self,
        dataset="mnist",
        test_ratio=0.2,
        split=None,
        n_clients=None,
        nc_per_client=None,
        n_client_perc=None,
        dir_alpha=1.0,
        ln_sigma=0.0,
        n_max_sam=None,
    ):
        self.dataset = dataset
        self.test_ratio = test_ratio
        self.split = split
        self.n_clients = n_clients
        self.nc_per_client = nc_per_client
        self.n_client_perc = n_client_perc
        self.dir_alpha = dir_alpha
        self.ln_sigma = ln_sigma
        self.n_max_sam = n_max_sam

        self.label_dsets = [
            "mnist", "svhn", "mnistm", "usps", "syn", "famnist",
            "cifar10", "cifar100", "cinic10", "ohsumed",
            "sa", "gtsrb"
        ]
        self.user_dsets = [
            "digits-five", "femnist", "shakespeare",
            "speechcommands"
        ]

        if dataset in self.label_dsets:
            assert self.split in ["shard", "dirichlet", "real", "map"]

            assert (n_clients is not None), \
                "{} needs pre-defined n_clients".format(dataset)

            if self.split == "shard":
                if dataset == "sa":
                    assert (n_client_perc is not None), \
                        "{} needs pre-defined n_client_perc".format(dataset)
                else:
                    assert (nc_per_client is not None), \
                        "{} needs pre-defined nc_per_client".format(dataset)

            if self.split == "dirichlet":
                assert (dir_alpha is not None), \
                    "{} needs pre-defined dir_alpha".format(dir_alpha)

            if self.split == "real" or self.split == "map":
                assert (dir_alpha is None) and (nc_per_client is None), \
                    "split real does not need dir_alpha, nc_per_client"

            assert (ln_sigma is not None), "client imbalance needs ln_sigma"

        if dataset in self.user_dsets:
            self.split = "user"

    def split_by_real(self, xs, ys):
        """ split data into N clients with real-world distributions
        params:
        @xs: numpy.array, shape=(N, ...)
        @ys: numpy.array, shape=(N, ), only for classes
        return:
        @clients_data, a dict like {
            client: {
                "train_xs":,
                "train_ys":,
                "test_xs":,
                "test_ys":
            }
        }
        """
        # unique classes
        n_classes = len(np.unique(ys))
        class_cnts = np.array([
            np.sum(ys == c) for c in range(n_classes)
        ])
        class_indxes = {
            c: np.argwhere(ys == c).reshape(-1) for c in range(n_classes)
        }

        # (n_clients, n_classes)

        if self.split == "shard":
            dists = []
            nc = self.nc_per_client
            uni_cs = list(range(n_classes)) * self.n_clients
            for k in range(self.n_clients):
                cs = uni_cs[k * nc: (k + 1) * nc]
                dist = np.array([0.0] * n_classes)
                dist[cs] = 1.0 / nc
                dists.append(dist)
            dists = np.array(dists)
        elif self.split == "dirichlet":
            dists = np.random.dirichlet(
                alpha=[self.dir_alpha] * n_classes,
                size=self.n_clients
            )
        elif self.split == "real":
            # imbalance
            n_dir_clients = int(0.5 * self.n_clients)

            dir_dists = []
            for k in range(n_dir_clients):
                dir_alpha = np.random.choice(
                    [0.01, 0.1, 0.5, 1.0, 10.0]
                )
                dist = np.random.dirichlet(
                    alpha=[dir_alpha] * n_classes,
                    size=1
                ).reshape(-1)
                dir_dists.append(dist)
            dir_dists = np.array(dir_dists)

            # missing
            n_miss_clients = self.n_clients - n_dir_clients

            miss_dists = []
            for k in range(n_miss_clients):
                nc = np.random.choice(range(1, n_classes + 1))
                cs = np.random.choice(range(n_classes), nc, replace=False)
                als = np.array([1e-8] * n_classes)
                als[cs] = 1.0

                dist = np.random.dirichlet(
                    alpha=als, size=1
                ).reshape(-1)
                miss_dists.append(dist)
            miss_dists = np.array(miss_dists)

            dists = np.concatenate([dir_dists, miss_dists], axis=0)
        elif self.split == "map":
            # imbalance
            dists = []
            for k in range(self.n_clients):
                nc = np.random.choice(range(2, n_classes + 1))
                cs = np.random.choice(range(n_classes), nc, replace=False)
                dist = np.array([0.0] * n_classes)
                dist[cs] = 1.0 / nc
                dists.append(dist)
            dists = np.array(dists)
        else:
            raise ValueError("No such way: {}".format(self.split))

        # number of sample distributions
        n_dist = np.random.lognormal(
            mean=0.0, sigma=self.ln_sigma, size=(self.n_clients, )
        ).reshape(self.n_clients, 1)
        dists = dists * n_dist

        dists = dists / dists.sum(axis=0)

        # (n_clients, n_classes)
        cnts = (dists * class_cnts.reshape((1, -1)))
        cnts = np.round(cnts).astype(np.int32)

        cnts = np.cumsum(cnts, axis=0)
        cnts = np.concatenate([
            np.zeros((1, n_classes)).astype(np.int32),
            cnts
        ], axis=0)

        # split data by Dists
        clients_data = {}
        for n in range(self.n_clients):
            client_xs = []
            client_ys = []
            for c in range(n_classes):
                cinds = class_indxes[c]
                bi, ei = cnts[n][c], cnts[n + 1][c]
                c_xs = xs[cinds[bi:ei]]
                c_ys = ys[cinds[bi:ei]]

                client_xs.append(c_xs)
                client_ys.append(c_ys)
                if n == self.n_clients - 1:
                    print(c, len(cinds), bi, ei)

            client_xs = np.concatenate(client_xs, axis=0)
            client_ys = np.concatenate(client_ys, axis=0)

            inds = np.random.permutation(client_xs.shape[0])
            client_xs = client_xs[inds]
            client_ys = client_ys[inds]

            # filter small corpus
            if len(client_xs) < 5:
                continue

            # split train and test
            n_test = max(int(self.test_ratio * len(client_xs)), 1)

            # max train samples
            if self.n_max_sam is None:
                n_end = None
            else:
                n_end = self.n_max_sam + n_test

            clients_data[n] = {
                "train_xs": client_xs[n_test:n_end],
                "train_ys": client_ys[n_test:n_end],
                "test_xs": client_xs[:n_test],
                "test_ys": client_ys[:n_test],
            }

        return clients_data

    def construct_datasets(
            self, clients_data, Dataset, glo_test_xs=None, glo_test_ys=None):
        """
        params:
        @clients_data, {
            client: {
                "train_xs":,
                "train_ys":,
                "test_xs":,
                "test_ys":
            }
        }
        @Dataset: torch.utils.data.Dataset type
        @glo_test_xs: global test xs, ys
        @glo_test_ys: global test xs, ys
        return: client train and test Datasets and global test Dataset
        @csets: {
            client: (train_set, test_set)
        }
        @gset: data.Dataset
        """
        csets = {}

        if glo_test_xs is None or glo_test_ys is None:
            glo_test = False
        else:
            glo_test = True

        if glo_test is False:
            glo_test_xs = []
            glo_test_ys = []

        for client, cdata in clients_data.items():
            train_set = Dataset(
                cdata["train_xs"], cdata["train_ys"], is_train=True
            )
            test_set = Dataset(
                cdata["test_xs"], cdata["test_ys"], is_train=False
            )
            csets[client] = (train_set, test_set)

            if glo_test is False:
                glo_test_xs.append(cdata["test_xs"])
                glo_test_ys.append(cdata["test_ys"])

        if glo_test is False:
            glo_test_xs = np.concatenate(glo_test_xs, axis=0)
            glo_test_ys = np.concatenate(glo_test_ys, axis=0)

        gset = Dataset(glo_test_xs, glo_test_ys, is_train=False)
        return csets, gset

    def construct(self):
        """ load raw data
        """
        if self.dataset == "mnist":
            train_xs, train_ys, test_xs, test_ys = load_mnist_data(
                "mnist", combine=False
            )
            clients_data = self.split_by_real(train_xs, train_ys)
            csets, gset = self.construct_datasets(
                clients_data, MnistDataset, test_xs, test_ys
            )
        elif self.dataset in ["mnistm", "svhn", "usps", "syn"]:
            train_xs, train_ys, test_xs, test_ys = load_digits_data(
                self.dataset, combine=False
            )
            clients_data = self.split_by_real(train_xs, train_ys)
            csets, gset = self.construct_datasets(
                clients_data, DigitsDataset, test_xs, test_ys
            )
        elif self.dataset == "famnist":
            train_xs, train_ys, test_xs, test_ys = load_famnist_data(
                "famnist", combine=False
            )
            clients_data = self.split_by_real(train_xs, train_ys)
            csets, gset = self.construct_datasets(
                clients_data, FaMnistDataset, test_xs, test_ys
            )
        elif self.dataset in ["cifar10", "cifar100"]:
            train_xs, train_ys, test_xs, test_ys = load_cifar_data(
                self.dataset, combine=False
            )
            clients_data = self.split_by_real(train_xs, train_ys)
            csets, gset = self.construct_datasets(
                clients_data, CifarDataset, test_xs, test_ys
            )
        elif self.dataset in ["cinic10"]:
            train_xs, train_ys, test_xs, test_ys = load_cinic_data(
                self.dataset, combine=False
            )
            clients_data = self.split_by_real(train_xs, train_ys)
            csets, gset = self.construct_datasets(
                clients_data, CinicDataset, test_xs, test_ys
            )
        elif self.dataset == "digits-five":
            clients_data = {}
            domains = ["mnist", "mnistm", "usps", "svhn", "syn"]
            for domain in domains:
                train_xs, train_ys, test_xs, test_ys = load_digits_data(
                    domain, combine=False
                )

                inds = np.random.permutation(train_xs.shape[0])
                train_xs = train_xs[inds]
                train_ys = train_ys[inds]

                clients_data[domain] = {
                    "train_xs": train_xs[0:self.n_max_sam],
                    "train_ys": train_ys[0:self.n_max_sam],
                    "test_xs": test_xs,
                    "test_ys": test_ys,
                }
            csets, gset = self.construct_datasets(
                clients_data, DigitsDataset
            )
        else:
            raise ValueError("No such dataset: {}".format(self.dataset))

        return csets, gset

    def print_info(self, csets, gset, max_cnt=5):
        """ print information
        """
        print("#" * 50)
        cnt = 0
        print("Dataset:{}".format(self.dataset))
        print("N clients:{}".format(len(csets)))

        for client, (cset1, cset2) in csets.items():
            print("Information of Client {}:".format(client))
            print(
                "Local Train Set: ", cset1.xs.shape,
                cset1.xs.max(), cset1.xs.min(), Counter(cset1.ys)
            )
            print(
                "Local Test Set: ", cset2.xs.shape,
                cset2.xs.max(), cset2.xs.min(), Counter(cset2.ys)
            )

            cnts = [n for _, n in Counter(cset1.ys).most_common()]
            probs = np.array([n / sum(cnts) for n in cnts])
            ent = -1.0 * (probs * np.log(probs + 1e-8)).sum()
            print("Class Distribution, Min:{}, Max:{}, Ent:{}".format(
                np.min(probs), np.max(probs), ent
            ))

            if cnt >= max_cnt:
                break
            cnt += 1

        print(
            "Global Test Set: ", gset.xs.shape,
            gset.xs.max(), gset.xs.min(), Counter(gset.ys)
        )
        print("#" * 50)


if __name__ == "__main__":
    for dataset in ["mnist", "svhn", "cifar10", "cifar100"]:
        for n_max_sam in [500, None]:
            for n_clients in [100]:
                for nc_per_client in [5]:
                    print("#" * 50)
                    feddata = FedData(
                        dataset=dataset,
                        n_clients=n_clients,
                        nc_per_client=nc_per_client,
                        n_max_sam=n_max_sam
                    )
                    csets, gset = feddata.construct()
                    feddata.print_info(csets, gset, max_cnt=5)

    for dataset in ["sa"]:
        for n_max_sam in [500, None]:
            for n_clients in [100]:
                for n_client_perc in [50]:
                    print("#" * 50)
                    feddata = FedData(
                        dataset=dataset,
                        n_clients=n_clients,
                        n_client_perc=n_client_perc,
                        n_max_sam=n_max_sam
                    )
                    csets, gset = feddata.construct()
                    feddata.print_info(csets, gset, max_cnt=5)

    for n_max_sam in [50, 500, None]:
        feddata = FedData(
            dataset="digits-five",
            n_max_sam=n_max_sam
        )
        print("#" * 50)
        csets, gset = feddata.construct()
        feddata.print_info(csets, gset, max_cnt=5)

    for n_max_sam in [50, 500, None]:
        feddata = FedData(
            dataset="shakespeare",
            n_max_sam=n_max_sam
        )
        print("#" * 50)
        csets, gset = feddata.construct()
        feddata.print_info(csets, gset, max_cnt=5)
