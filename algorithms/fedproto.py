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


class FedProto():
    def __init__(
        self, csets, gset, model, args
    ):
        self.csets = csets
        self.gset = gset
        self.model = model
        self.args = args

        self.clients = list(csets.keys())

        # construct dataloaders
        self.train_loaders, self.test_loaders, self.glo_test_loader = \
            construct_dataloaders(
                self.clients, self.csets, self.gset, self.args
            )

        self.local_models = {}
        for client in self.clients:
            self.local_models[client] = copy.deepcopy(self.model)

        self.global_protos = None

        self.logs = {
            "ROUNDS": [],
            "LOSSES": [],
            "GLO_TACCS": [],
            "LOCAL_TACCS": [],
            "LOCAL_MF1S": [],
        }

        print("Train FedProto!")

    def train(self):
        # Training
        for r in range(1, self.args.max_round + 1):
            n_sam_clients = int(self.args.c_ratio * len(self.clients))
            sam_clients = np.random.choice(
                self.clients, n_sam_clients, replace=False
            )

            all_local_protos = []
            all_local_cnts = []

            avg_loss = Averager()
            all_per_accs = []
            all_per_mf1s = []
            for client in sam_clients:
                local_protos, local_cnts, per_accs, per_mf1s, loss \
                    = self.update_local(
                        r=r,
                        model=self.local_models[client],
                        train_loader=self.train_loaders[client],
                        test_loader=self.test_loaders[client],
                        global_protos=self.global_protos
                    )

                all_local_protos.append(copy.deepcopy(local_protos))
                all_local_cnts.append(copy.deepcopy(local_cnts))
                avg_loss.add(loss)
                all_per_accs.append(per_accs)
                all_per_mf1s.append(per_mf1s)

            train_loss = avg_loss.item()
            per_accs = list(np.array(all_per_accs).mean(axis=0))
            per_mf1s = list(np.array(all_per_mf1s).mean(axis=0))

            self.update_global(
                r=r,
                all_local_protos=all_local_protos,
                all_local_cnts=all_local_cnts
            )

            if r % self.args.test_round == 0:
                # global test loader
                glo_test_acc = 0.0

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

    def update_local(self, r, model, train_loader, test_loader, global_protos):
        # lr = min(r / 10.0, 1.0) * self.args.lr
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

        all_protos = []
        all_cnts = []
        for t in range(n_total_bs + 1):
            if t in [0, n_total_bs]:
                per_acc, mf1 = self.test(
                    model=model,
                    loader=test_loader,
                )
                per_accs.append(per_acc)
                per_mf1s.append(mf1)

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

            hs, logits = model(batch_x)

            criterion = nn.CrossEntropyLoss()
            loss1 = criterion(logits, batch_y)

            loss2, protos, cnts = self.proto_reg(
                feats=hs,
                labels=batch_y.detach().cpu(),
                global_protos=global_protos
            )

            loss = loss1 + self.args.reg_lamb * loss2

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.args.max_grad_norm
            )
            optimizer.step()

            avg_loss.add(loss.item())
            all_protos.append(protos.detach())
            all_cnts.append(cnts.detach())

        # aggregate protos
        protos = None
        cnts = None
        for cs, ps in zip(all_cnts, all_protos):
            if cnts is None:
                cnts, protos = cs, ps * cs
            else:
                cnts += cs
                protos += ps * cs

        protos = protos / (cnts + 1e-8)

        loss = avg_loss.item()
        return protos, cnts, per_accs, per_mf1s, loss

    def proto_reg(self, feats, labels, global_protos):
        C = self.args.n_classes

        mat = torch.diag(torch.ones(C)).to(feats.device)
        ys = mat[labels].transpose(0, 1)

        cnts = ys.sum(axis=1, keepdims=True)

        norm_ys = ys / (cnts + 1e-8)
        protos = torch.mm(norm_ys, feats)

        if global_protos is None:
            loss = 0.0
        else:
            mask = (cnts > 0.0).float()
            losses = ((protos - global_protos) ** 2).sum(axis=1, keepdims=True)
            loss = (mask * losses).sum() / mask.sum()

        return loss, protos, cnts

    def update_global(
            self, r, all_local_protos, all_local_cnts):
        protos = None
        cnts = None
        for cs, ps in zip(all_local_cnts, all_local_protos):
            if cnts is None:
                cnts, protos = cs, ps * cs
            else:
                cnts += cs
                protos += ps * cs

        protos = protos / (cnts + 1e-8)

        if self.global_protos is None:
            self.global_protos = protos
        else:
            self.global_protos = 0.5 * self.global_protos + 0.5 * protos

    def test(self, model, loader):
        model.eval()

        acc_avg = Averager()

        preds = []
        reals = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):
                if self.args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                _, logits = model(batch_x)
                acc = count_acc(logits, batch_y)
                acc_avg.add(acc)

                preds.append(np.argmax(logits.cpu().detach().numpy(), axis=1))
                reals.append(batch_y.cpu().detach().numpy())

        preds = np.concatenate(preds, axis=0)
        reals = np.concatenate(reals, axis=0)

        acc = acc_avg.item()

        # MACRO F1
        mf1 = f1_score(y_true=reals, y_pred=preds, average="macro")
        return acc, mf1

    def save_logs(self, fpath):
        all_logs_str = []
        all_logs_str.append(str(self.args))

        logs_str = format_logs(self.logs)
        all_logs_str.extend(logs_str)

        append_to_logs(fpath, all_logs_str)
