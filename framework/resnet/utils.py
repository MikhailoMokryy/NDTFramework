import argparse


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-1,
        help="learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="SGD weight decay (default: 5e-4)",
    )

    parser.add_argument(
        "--dataset",
        choices=["CIFAR10"],
        type=str,
        default="CIFAR10",
        help="Dataset (default: CIFAR10)",
    )

    parser.add_argument(
        "--loss",
        type=str,
        choices=["SoftTreeSupLoss", "HardTreeSupLoss", "CrossEntropyLoss"],
        default="CrossEntropyLoss",
        help="Loss function (default: CrossEntropyLoss)",
    )

    parser.add_argument(
        "--hierarchy",
        type=str,
        help="Hierarchy for NBDT",
    )
