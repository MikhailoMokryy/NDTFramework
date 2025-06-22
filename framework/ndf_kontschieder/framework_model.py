from argparse import Namespace
from typing import Union, cast
import torch
import torch.optim.adam as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from framework.ndf_kontschieder.model import NeuralDecisionForest, NeuralDecisionTree
from framework.utils.framework_model import FrameworkModel
from framework.utils.common import progress_bar
from framework.utils.incremental_learning.dataset import IncrementalNoiseCIFAR10
from framework.utils.model_actions import save_model, test_model


class NDModel(FrameworkModel):
    def __init__(
        self,
        net: Union[NeuralDecisionForest, NeuralDecisionTree],
        args: Namespace,
    ):
        super(NDModel, self).__init__(net, args)
        self.optimizer = optim.Adam(
            params=[p for p in self.net.parameters() if p.requires_grad],
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, T_max=self.args.epochs
        # )

    def train_(self, train_loader: DataLoader, epoch: int):
        self.net.train()

        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.args.device), target.to(self.args.device)
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = F.nll_loss(torch.log(output), target)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            pred = output.max(1, keepdim=True)[1]
            # TODO: Check this line
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total += target.size(0)

            progress_bar(
                batch_idx,
                len(train_loader),
                "Epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    epoch,
                    train_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )
        self.train_record(train_loss / len(train_loader), 100 * correct / total)

    def eval_(self, eval_loader, epoch: int):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(eval_loader):
                data, target = data.to(self.args.device), target.to(self.args.device)
                output = self.net(data)
                loss = F.nll_loss(torch.log(output), target).item()
                test_loss += loss

                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
                total += target.size(0)

                progress_bar(
                    batch_idx,
                    len(eval_loader),
                    "Epoch: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        epoch,
                        test_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )

        accuracy = 100.0 * correct / total

        if accuracy > self.best_accuracy:
            save_model(self.net, accuracy, epoch, self.args)
            self.best_accuracy = accuracy

        self.eval_record(test_loss / len(eval_loader), accuracy)


class FrameworkNDT(NDModel, FrameworkModel):
    def __init__(self, args: Namespace):
        net = NeuralDecisionTree(args)
        super(FrameworkNDT, self).__init__(net, args)

    def train(self, train_loader: DataLoader, eval_loader: DataLoader):
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            if self.args.train_type == "inc_learning":
                current_noise_percentage = cast(
                    IncrementalNoiseCIFAR10, train_loader.dataset
                ).add_noise(epoch)
                self.start_record({"noise": current_noise_percentage})
            else:
                self.start_record()

            if not self.args.jointly_training:
                print("Not jointly updating tree...")
                self.net.update_tree(train_loader)

            # print("LR", self.scheduler.get_last_lr())
            self.train_(train_loader, epoch)
            self.eval_(eval_loader, epoch)
            # self.scheduler.step()

    def test(self, test_loader: DataLoader):
        return test_model(self.net, test_loader, self.args.device)


class FrameworkNDF(NDModel, FrameworkModel):
    def __init__(self, args: Namespace):
        net = NeuralDecisionForest(args)
        super(FrameworkNDF, self).__init__(net, args)

    def train(self, train_loader: DataLoader, eval_loader: DataLoader):
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            if self.args.train_type == "inc_learning":
                current_noise_percentage = cast(
                    IncrementalNoiseCIFAR10, train_loader.dataset
                ).add_noise(epoch)
                self.start_record({"noise": current_noise_percentage})
            else:
                self.start_record()

            if not self.args.jointly_training:
                print("Not jointly updating trees...")
                self.net.update_trees(train_loader)

            self.train_(train_loader, epoch)
            self.eval_(eval_loader, epoch)
            # self.scheduler.step()

    def test(self, test_loader: DataLoader):
        return test_model(self.net, test_loader, self.args.device)
