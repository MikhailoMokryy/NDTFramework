import torch.nn as nn
from torch.utils.data import DataLoader
from nbdt.model import HardNBDT, SoftNBDT
from nbdt.hierarchy import generate_hierarchy
from nbdt.loss import HardTreeSupLoss, SoftTreeSupLoss
from framework.resnet.framework_model import FrameworkResNet18
from framework.utils.model_actions import test_model


class FrameworkNBDT(FrameworkResNet18):
    def __init__(self, args):
        super(FrameworkNBDT, self).__init__(args)

        # generate_hierarchy(dataset=args.dataset, arch=ResNet18, method="induced")
        if args.loss == "SoftTreeSupLoss":
            self.criterion = SoftTreeSupLoss(
                dataset=args.dataset,
                criterion=nn.CrossEntropyLoss(),
                hierarchy=args.hierarchy,
            )
        elif args.loss == "HardTreeSupLoss":
            self.criterion = HardTreeSupLoss(
                dataset=args.dataset,
                criterion=nn.CrossEntropyLoss(),
                hierarchy=args.hierarchy,
            )
        else:
            raise ValueError(f"Loss is not supported for FrameworkNBDT: {args.loss}")

    def test(self, test_loader: DataLoader):
        # hierarchy="induced-resnet18"

        if not self.args.hierarchy:
            raise ValueError("Hierarchy is not provided for NBDT")

        if self.args.loss == "SoftTreeSupLoss":
            self.net = SoftNBDT(
                dataset=self.args.dataset, model=self.net, hierarchy=self.args.hierarchy
            )
        elif self.args.loss == "HardTreeSupLoss":
            self.net = HardNBDT(
                dataset=self.args.dataset, model=self.net, hierarchy=self.args.hierarchy
            )
        else:
            raise ValueError(
                f"Loss is not supported for FrameworkNBDT: {self.args.loss}"
            )

        return test_model(self.net, test_loader, self.args.device)


class FrameworkHardNBDT(FrameworkResNet18):
    def __init__(self, args):
        super(FrameworkHardNBDT, self).__init__(args)
        self.args.loss = "HardTreeSupLoss"
        self.criterion = HardTreeSupLoss(
            dataset=args.dataset,
            criterion=nn.CrossEntropyLoss(),
            hierarchy=args.hierarchy,
        )

    def test(self, test_loader: DataLoader):
        if not self.args.hierarchy:
            raise ValueError("Hierarchy is not provided for NBDT")

        self.net = HardNBDT(
            dataset=self.args.dataset, model=self.net, hierarchy=self.args.hierarchy
        )

        return test_model(self.net, test_loader, self.args.device)
