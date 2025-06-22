import argparse


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--feat-dropout", type=float, default=0.5)
    parser.add_argument("--n-tree", type=int, default=5)
    parser.add_argument("--tree-depth", type=int, default=3)
    parser.add_argument("--n-class", type=int, default=10)
    parser.add_argument("--tree-feature-rate", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3, help="sgd: 10, adam: 0.001")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="weight decay (default: adam 1e-5)",
    )
    parser.add_argument("--jointly-training", action="store_true", default=False)
