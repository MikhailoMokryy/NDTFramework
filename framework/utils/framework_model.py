from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import torch.nn as nn
import argparse
from torch.utils.data.dataloader import DataLoader
from datetime import datetime
import time
from framework.utils.common import initRecords
from framework.utils.model_actions import load_model, save_logs


class FrameworkModel(ABC):
    def __init__(
        self,
        net: nn.Module,
        args: argparse.Namespace,
    ):
        super(FrameworkModel, self).__init__()
        self.args = args
        self.records = initRecords()

        if self.args.eval or self.args.resume:
            if not self.args.model_path:
                raise ValueError("Model path is required for evaluation")
            load_state = load_model(net, self.args.model_path, self.args.device)
            self.net = load_state["net"]
            self.start_epoch = load_state["epoch"]
            self.best_accuracy = load_state["acc"]
        else:
            self.net = net
            self.start_epoch = 1
            self.best_accuracy = 0.0

        self.net.to(self.args.device)

    @abstractmethod
    def train(self, train_loader: DataLoader, eval_loader: DataLoader):
        pass

    @abstractmethod
    def test(self, test_loader: DataLoader) -> float:
        pass

    def start_record(self, data: Optional[Dict[str, Union[str, float]]] = None):
        if data is not None:
            for k, v in data.items():
                if k not in self.records:
                    self.records[k] = []
                self.records[k].append(v)

        self.records["time"].append(time.time())

    def train_record(self, loss: float, accuracy: float):
        self.records["train_acc"].append(accuracy)
        self.records["train_loss"].append(loss)

    def eval_record(self, loss: float, accuracy: float):
        self.records["val_acc"].append(accuracy)
        self.records["val_loss"].append(loss)
        self.records["timestamp"].append(datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
        self.records["time"].append(time.time() - self.records["time"].pop())

        save_logs(self.args, self.records)


class PartialFrameworkModel(ABC):
    def __init__(self, args: argparse.Namespace):
        super(PartialFrameworkModel, self).__init__()
        self.args = args

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self, testloader: DataLoader) -> Any:
        pass
