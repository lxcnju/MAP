import copy
import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from utils import Averager
from utils import count_acc
from utils import append_to_logs
from utils import format_logs

from tools import construct_dataloaders
from tools import construct_optimizer


class FLDA():
    def __init__(
        self,
        csets,
        gset,
        model,
        args
    ):
        self.csets = csets
        self.gset = gset
        self.model = model
        self.args = args

        self.clients = list(csets.keys())
        self.n_client = len(self.clients)

        # copy private layers for each client
        self.private_models = {}
        self.private_weight_layers = {}
        for client in self.clients:
            self.private_models[client] = copy.deepcopy(
                model.private_model
            )
            self.private_weight_layers[client] = copy.deepcopy(
                model.private_weight_layer
            )

        # to cuda
        if self.args.cuda is True:
            self.model = self.model.cuda()

        # construct dataloaders
        self.train_loaders, self.test_loaders, self.glo_test_loader = \
            construct_dataloaders(
                self.clients, self.csets, self.gset, self.args
            )

        self.logs = {
            "ROUNDS": [],
            "LOSSES": [],
            "GLO_TACCS": [],
            "LOCAL_TACCS": [],
            "LOCAL_MF1S": [],
        }

    def train(self):
        # Training
        for r in range(1, self.args.max_round + 1):
            n_sam_clients = int(self.args.c_ratio * len(self.clients))
            sam_clients = np.random.choice(
                self.clients, n_sam_clients, replace=False
            )

            local_models = {}

            avg_loss = Averager()
            all_per_accs = []
            all_per_mf1s = []

            for client in sam_clients:
                local_model = copy.deepcopy(self.model)

                # to cuda
                if self.args.cuda is True:
                    self.private_models[client].cuda()
                    self.private_weight_layers[client].cuda()

                state_dict = copy.deepcopy(self.model.state_dict())
                state_dict.update({
                    "private_model.{}".format(k): v for k, v in
                    self.private_models[client].state_dict().items()
                })
                state_dict.update({
                    "private_weight_layer.{}".format(k): v for k, v in
                    self.private_weight_layers[client].state_dict().items()
                })

                local_model.load_state_dict(state_dict, strict=True)

                local_model, per_accs, per_mf1s, loss = self.update_local(
                    r=r,
                    model=local_model,
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                )

                local_models[client] = copy.deepcopy(local_model)

                # update local model
                state_dict = local_model.private_model.state_dict()
                self.private_models[client].load_state_dict(
                    state_dict, strict=True
                )

                state_dict = local_model.private_weight_layer.state_dict()
                self.private_weight_layers[client].load_state_dict(
                    state_dict, strict=True
                )

                # to cpu
                if self.args.cuda is True:
                    self.private_models[client].to("cpu")
                    self.private_weight_layers[client].to("cpu")

                avg_loss.add(loss)
                all_per_accs.append(per_accs)
                all_per_mf1s.append(per_mf1s)

            train_loss = avg_loss.item()
            per_accs = list(np.array(all_per_accs).mean(axis=0))
            per_mf1s = list(np.array(all_per_mf1s).mean(axis=0))

            self.update_global(
                r=r,
                global_model=self.model,
                local_models=local_models,
            )

            if r % self.args.test_round == 0:
                # global test loader
                glo_test_acc = self.global_test(
                    model=self.model,
                    loader=self.glo_test_loader,
                )

                # add to log
                self.logs["ROUNDS"].append(r)
                self.logs["LOSSES"].append(train_loss)
                self.logs["GLO_TACCS"].append(glo_test_acc)
                self.logs["LOCAL_TACCS"].extend(per_accs)
                self.logs["LOCAL_MF1S"].extend(per_mf1s)

                print("[R:{}] [Ls:{}] [TAc:{}] [PAc:{},{}] [PF1:{},{}]".format(
                    r, train_loss, glo_test_acc, per_accs[0], per_accs[-1],
                    per_mf1s[0], per_mf1s[-1]
                ))

    def update_local(self, r, model, train_loader, test_loader):
        lr = min(r / 5.0, 1.0) * self.args.lr

        optimizer = construct_optimizer(
            model, lr, self.args
        )

        if self.args.local_steps is not None:
            n_total_bs = self.args.local_steps
        elif self.args.local_epochs is not None:
            n_total_bs = max(
                int(self.args.local_epochs * len(train_loader)), 5
            )
        else:
            raise ValueError(
                "local_steps and local_epochs must not be None together"
            )

        model.train()

        loader_iter = iter(train_loader)

        avg_loss = Averager()
        per_accs = []
        per_mf1s = []

        for t in range(n_total_bs + 1):
            if t in [0, n_total_bs]:
                per_acc, per_mf1 = self.local_test(
                    model=model,
                    loader=test_loader,
                )
                per_accs.append(per_acc)
                per_mf1s.append(per_mf1)

            if t >= n_total_bs:
                break

            model.train()
            try:
                batch_x, batch_y = next(loader_iter)
            except Exception:
                loader_iter = iter(train_loader)
                batch_x, batch_y = next(loader_iter)

            if self.args.cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            logits, plogits = model(batch_x)

            criterion = nn.CrossEntropyLoss()
            loss1 = criterion(logits, batch_y)
            loss2 = criterion(plogits, batch_y)
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.args.max_grad_norm
            )
            optimizer.step()

            avg_loss.add(loss.item())

        loss = avg_loss.item()
        return model, per_accs, per_mf1s, loss

    def update_global(self, r, global_model, local_models):
        mean_state_dict = {}

        for name, param in global_model.state_dict().items():
            if "private" in name:
                continue

            vs = []
            for client in local_models.keys():
                vs.append(local_models[client].state_dict()[name])
            vs = torch.stack(vs, dim=0)

            try:
                mean_value = vs.mean(dim=0)
            except Exception:
                # for BN's cnt
                mean_value = (1.0 * vs).mean(dim=0).long()

            mean_state_dict[name] = mean_value

        global_model.load_state_dict(mean_state_dict, strict=False)

    def local_test(self, model, loader):
        model.eval()

        acc_avg = Averager()

        preds = []
        reals = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):
                if self.args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                logits, plogits = model(batch_x)
                probs1 = logits.softmax(dim=-1)
                probs2 = plogits.softmax(dim=-1)
                probs = 0.5 * (probs1 + probs2)

                acc = count_acc(probs, batch_y)
                acc_avg.add(acc)

                preds.append(np.argmax(probs.cpu().detach().numpy(), axis=1))
                reals.append(batch_y.cpu().detach().numpy())

        preds = np.concatenate(preds, axis=0)
        reals = np.concatenate(reals, axis=0)

        acc = acc_avg.item()

        # MACRO F1
        mf1 = f1_score(y_true=reals, y_pred=preds, average="macro")

        acc = acc_avg.item()
        return acc, mf1

    def global_test(self, model, loader):
        model.eval()

        acc_avg = Averager()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):
                if self.args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                res = model.global_forward(batch_x)
                acc = count_acc(res, batch_y)
                acc_avg.add(acc)

        acc = acc_avg.item()
        return acc

    def save_logs(self, fpath):
        all_logs_str = []
        all_logs_str.append(str(self.args))

        logs_str = format_logs(self.logs)
        all_logs_str.extend(logs_str)

        append_to_logs(fpath, all_logs_str)
