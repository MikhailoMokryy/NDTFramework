from typing import cast
import torch
import torch.nn as nn
import torch.optim.sgd as optim
from torch.utils.data import DataLoader
from framework.resnet.model import ResNet18
from framework.utils.common import progress_bar
from framework.utils.framework_model import FrameworkModel
from framework.utils.incremental_learning.dataset import IncrementalNoiseCIFAR10
from framework.utils.model_actions import save_model, test_model


class FrameworkResNet18(FrameworkModel):
    def __init__(self, args):
        net = ResNet18()
        super(FrameworkResNet18, self).__init__(net, args)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            momentum=self.args.momentum,
        )
        # NOTE: Used in Kaggle for some reason
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.epochs
        )
        # NOTE: Used in original paper
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     self.optimizer,
        #     milestones=[int(3 / 7.0 * args.epochs), int(5 / 7.0 * args.epochs)],
        # )

    def train(self, train_loader: DataLoader, eval_loader: DataLoader):
        for epoch in range(self.start_epoch, self.args.epochs + 1):
            self._train(train_loader, epoch)
            self._eval(eval_loader, epoch)
            self.scheduler.step()

    def test(self, test_loader: DataLoader):
        return test_model(self.net, test_loader, self.args.device)

    def _train(self, train_loader: DataLoader, epoch: int):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0

        if self.args.train_type == "inc_learning":
            current_noise_percentage = cast(
                IncrementalNoiseCIFAR10, train_loader.dataset
            ).add_noise(epoch)
            self.start_record({"noise": current_noise_percentage})
        else:
            self.start_record()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = (
                inputs.to(self.args.device),
                targets.to(self.args.device),
            )

            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()

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

    def _eval(self, eval_loader: DataLoader, epoch: int):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(eval_loader):
                inputs, targets = (
                    inputs.to(self.args.device),
                    targets.to(self.args.device),
                )
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()

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
