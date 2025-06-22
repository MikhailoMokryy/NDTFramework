from argparse import Namespace
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader


class MNISTFeatureLayer(nn.Sequential):
    def __init__(self, dropout_rate, shallow=False):
        super(MNISTFeatureLayer, self).__init__()
        self.shallow = shallow
        if shallow:
            self.add_module(
                "conv1", nn.Conv2d(1, 64, kernel_size=15, padding=1, stride=5)
            )
        else:
            self.add_module("conv1", nn.Conv2d(1, 32, kernel_size=3, padding=1))
            self.add_module("relu1", nn.ReLU())
            self.add_module("pool1", nn.MaxPool2d(kernel_size=2))
            self.add_module("drop1", nn.Dropout(dropout_rate))
            self.add_module("conv2", nn.Conv2d(32, 64, kernel_size=3, padding=1))
            self.add_module("relu2", nn.ReLU())
            self.add_module("pool2", nn.MaxPool2d(kernel_size=2))
            self.add_module("drop2", nn.Dropout(dropout_rate))
            self.add_module("conv3", nn.Conv2d(64, 128, kernel_size=3, padding=1))
            self.add_module("relu3", nn.ReLU())
            self.add_module("pool3", nn.MaxPool2d(kernel_size=2))
            self.add_module("drop3", nn.Dropout(dropout_rate))

    def get_out_feature_size(self):
        if self.shallow:
            return 64 * 4 * 4
        else:
            return 128 * 3 * 3


class CIFARFeatureLayer(nn.Sequential):
    def __init__(self, dropout_rate):
        super(CIFARFeatureLayer, self).__init__()

        self.add_module("conv1", nn.Conv2d(3, 64, kernel_size=3, padding=1))
        self.add_module("bn1", nn.BatchNorm2d(64))
        self.add_module("relu1", nn.ReLU())

        self.add_module("conv2", nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.add_module("bn2", nn.BatchNorm2d(64))
        self.add_module("relu2", nn.ReLU())
        self.add_module("pool2", nn.MaxPool2d(kernel_size=2, stride=2))
        self.add_module("drop2", nn.Dropout(dropout_rate))

        self.add_module("conv2_1", nn.Conv2d(64, 128, kernel_size=3, padding=1))
        self.add_module("bn2_1", nn.BatchNorm2d(128))
        self.add_module("relu2_1", nn.ReLU())

        self.add_module("conv3", nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self.add_module("bn3", nn.BatchNorm2d(128))
        self.add_module("relu3", nn.ReLU())
        self.add_module("pool3", nn.MaxPool2d(kernel_size=2, stride=2))
        self.add_module("drop3", nn.Dropout(dropout_rate))

        self.add_module("conv4", nn.Conv2d(128, 256, kernel_size=3, padding=1))
        self.add_module("bn4", nn.BatchNorm2d(256))
        self.add_module("relu4", nn.ReLU())

        self.add_module("conv5", nn.Conv2d(256, 256, kernel_size=3, padding=1))
        self.add_module("bn5", nn.BatchNorm2d(256))
        self.add_module("relu5", nn.ReLU())
        self.add_module("pool5", nn.MaxPool2d(kernel_size=2, stride=2))
        self.add_module("drop5", nn.Dropout(dropout_rate))

    def get_out_feature_size(self):
        return 256 * 4 * 4


class Tree(nn.Module):
    def __init__(
        self,
        depth,
        n_in_feature,
        used_feature_rate,
        n_class,
        device,
        jointly_training=True,
    ):
        super(Tree, self).__init__()
        self.device = device
        self.depth = depth
        self.n_leaf = 2**depth
        self.n_class = n_class
        self.jointly_training = jointly_training
        # used features in this tree

        n_used_feature = int(n_in_feature * used_feature_rate)
        onehot = np.eye(n_in_feature)
        using_idx = np.random.choice(
            np.arange(n_in_feature), n_used_feature, replace=False
        )
        feature_mask = onehot[using_idx].T
        self.feature_mask = Parameter(
            torch.from_numpy(feature_mask).to(self.device, dtype=torch.float32),
            requires_grad=False,
        )
        # leaf label distribution
        if jointly_training:
            pi = np.random.rand(self.n_leaf, n_class)
            self.pi = Parameter(
                torch.from_numpy(pi).to(self.device, dtype=torch.float32),
                requires_grad=True,
            )
        else:
            pi = np.ones((self.n_leaf, n_class)) / n_class
            self.pi = Parameter(
                torch.from_numpy(pi).to(self.device, dtype=torch.float32),
                requires_grad=False,
            )

        # decision
        self.decision = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(n_used_feature, self.n_leaf)),
                    ("sigmoid", nn.Sigmoid()),
                ]
            )
        )

    def forward(self, x):
        """
        :param x(Variable): [batch_size,n_features]
        :return: route probability (Variable): [batch_size,n_leaf]
        """
        if self.feature_mask.device != x.device:
            self.feature_mask = self.feature_mask.to(x.device)

        feats = torch.mm(x, self.feature_mask)  # ->[batch_size,n_used_feature]
        decision = self.decision(feats)  # ->[batch_size,n_leaf]

        decision = torch.unsqueeze(decision, dim=2)
        decision_comp = 1 - decision
        decision = torch.cat(
            (decision, decision_comp), dim=2
        )  # -> [batch_size,n_leaf,2]

        # compute route probability
        # note: we do not use decision[:,0]
        batch_size = x.size()[0]
        _mu = x.data.new(batch_size, 1, 1).fill_(1.0).clone().detach().to(self.device)
        begin_idx = 1
        end_idx = 2
        for n_layer in range(0, self.depth):
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            _decision = decision[
                :, begin_idx:end_idx, :
            ]  # -> [batch_size,2**n_layer,2]
            _mu = _mu * _decision  # -> [batch_size,2**n_layer,2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer + 1)

        mu = _mu.view(batch_size, self.n_leaf)

        return mu

    def get_pi(self):
        if self.jointly_training:
            return F.softmax(self.pi, dim=-1)
        else:
            return self.pi

    def cal_prob(self, mu, pi):
        """

        :param mu [batch_size,n_leaf]
        :param pi [n_leaf,n_class]
        :return: label probability [batch_size,n_class]
        """
        p = torch.mm(mu, pi)
        return p

    def update_pi(self, new_pi):
        self.pi.data = new_pi


class Forest(nn.Module):
    def __init__(
        self,
        n_tree,
        tree_depth,
        n_in_feature,
        tree_feature_rate,
        n_class,
        device,
        jointly_training,
    ):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.n_tree = n_tree
        for _ in range(n_tree):
            tree = Tree(
                tree_depth,
                n_in_feature,
                tree_feature_rate,
                n_class,
                device,
                jointly_training,
            )
            self.trees.append(tree)

    def forward(self, x):
        probs = []
        for tree in self.trees:
            mu = tree(x)
            p = tree.cal_prob(mu, tree.get_pi())
            probs.append(p.unsqueeze(2))
        probs = torch.cat(probs, dim=2)
        prob = torch.sum(probs, dim=2) / self.n_tree

        return prob


class NeuralDecisionForest(nn.Module):
    def __init__(self, args: Namespace):
        super(NeuralDecisionForest, self).__init__()
        self.args = args
        self.feature_layer = CIFARFeatureLayer(self.args.feat_dropout)
        self.forest = Forest(
            n_tree=self.args.n_tree,
            tree_depth=self.args.tree_depth,
            n_in_feature=self.feature_layer.get_out_feature_size(),
            tree_feature_rate=self.args.tree_feature_rate,
            n_class=self.args.n_class,
            device=self.args.device,
            jointly_training=self.args.jointly_training,
        )

    def forward(self, x):
        out = self.feature_layer(x)
        out = out.view(x.size()[0], -1)
        out = self.forest(out)
        return out

    def update_trees(self, dataloader: DataLoader):
        # prepare feats
        cls_onehot = torch.eye(self.args.n_class)
        feat_batches = []
        target_batches = []

        with torch.no_grad():
            for _, (data, target) in enumerate(dataloader):
                data, target, cls_onehot = (
                    data.to(self.args.device),
                    target.to(self.args.device),
                    cls_onehot.to(self.args.device),
                )

                # Get feats
                feats = self.feature_layer(data)
                feats = feats.view(feats.size()[0], -1)
                feat_batches.append(feats)
                target_batches.append(cls_onehot[target])

            # Update \Pi for each tree
            for tree in self.forest.trees:
                mu_batches = []
                for feats in feat_batches:
                    mu = tree(feats)  # [batch_size,n_leaf]
                    mu_batches.append(mu)
                for _ in range(20):
                    new_pi = torch.zeros(
                        (tree.n_leaf, tree.n_class), device=self.args.device
                    )
                    # Tensor [n_leaf,n_class]

                    for mu, target in zip(mu_batches, target_batches):
                        pi = tree.get_pi()  # [n_leaf,n_class]
                        # [batch_size,n_class]
                        prob = tree.cal_prob(mu, pi)

                        # [batch_size,1,n_class]
                        _target = target.unsqueeze(1)
                        _pi = pi.unsqueeze(0)  # [1,n_leaf,n_class]
                        _mu = mu.unsqueeze(2)  # [batch_size,n_leaf,1]
                        _prob = torch.clamp(
                            prob.unsqueeze(1), min=1e-6, max=1.0
                        )  # [batch_size,1,n_class]

                        _new_pi = (
                            torch.mul(torch.mul(_target, _pi), _mu) / _prob
                        )  # [batch_size,n_leaf,n_class]
                        new_pi += torch.sum(_new_pi, dim=0)
                    # NOTE: Code was replaced!!!
                    # new_pi = F.softmax(new_pi, dim=1)
                    # new_pi = F.normalize(new_pi, p=1, dim=1)
                    new_pi = torch.clamp(
                        new_pi, min=-1e12, max=1e12
                    )  # Prevent extreme values
                    new_pi = F.softmax(new_pi, dim=1)
                    tree.update_pi(new_pi)


class NeuralDecisionTree(nn.Module):
    def __init__(self, args: Namespace):
        super(NeuralDecisionTree, self).__init__()
        self.args = args
        self.feature_layer = CIFARFeatureLayer(self.args.feat_dropout)

        self.tree = Tree(
            depth=self.args.tree_depth,
            n_in_feature=self.feature_layer.get_out_feature_size(),
            used_feature_rate=self.args.tree_feature_rate,
            n_class=self.args.n_class,
            device=self.args.device,
            jointly_training=self.args.jointly_training,
        )

    def forward(self, x):
        out = self.feature_layer(x)
        out = out.view(x.size(0), -1)
        mu = self.tree(out)
        out = self.tree.cal_prob(mu, self.tree.get_pi())
        return out

    def update_tree(self, dataloader: DataLoader):
        # prepare feats
        cls_onehot = torch.eye(self.args.n_class)
        feat_batches = []
        target_batches = []

        with torch.no_grad():
            for _, (data, target) in enumerate(dataloader):
                data, target, cls_onehot = (
                    data.to(self.args.device),
                    target.to(self.args.device),
                    cls_onehot.to(self.args.device),
                )

                # Get feats
                feats = self.feature_layer(data)
                feats = feats.view(feats.size()[0], -1)
                feat_batches.append(feats)
                target_batches.append(cls_onehot[target])

            # Update \Pi for the tree
            mu_batches = []
            for feats in feat_batches:
                mu = self.tree(feats)  # [batch_size,n_leaf]
                mu_batches.append(mu)

            # EM algorithm for updating leaf node distributions
            for _ in range(20):
                new_pi = torch.zeros(
                    (self.tree.n_leaf, self.tree.n_class), device=self.args.device
                )

                for mu, target in zip(mu_batches, target_batches):
                    pi = self.tree.get_pi()  # [n_leaf,n_class]
                    # [batch_size,n_class]
                    prob = self.tree.cal_prob(mu, pi)

                    # [batch_size,1,n_class]
                    _target = target.unsqueeze(1)
                    _pi = pi.unsqueeze(0)  # [1,n_leaf,n_class]
                    _mu = mu.unsqueeze(2)  # [batch_size,n_leaf,1]
                    _prob = torch.clamp(
                        prob.unsqueeze(1), min=1e-6, max=1.0
                    )  # [batch_size,1,n_class]

                    _new_pi = (
                        torch.mul(torch.mul(_target, _pi), _mu) / _prob
                    )  # [batch_size,n_leaf,n_class]
                    new_pi += torch.sum(_new_pi, dim=0)

                # Normalize to get probability distribution
                # NOTE: Code was replaced!!!
                # -- Old code:
                # new_pi = F.normalize(new_pi, p=1, dim=1)
                # -- Original code:
                # new_pi = F.softmax(new_pi, dim=1)
                # -- New code:
                new_pi = torch.clamp(
                    new_pi, min=-1e12, max=1e12
                )  # Prevent extreme values
                new_pi = F.softmax(new_pi, dim=1)
                self.tree.update_pi(new_pi)
