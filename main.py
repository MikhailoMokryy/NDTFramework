from nbdt.loss import Colors
import argparse
from framework.models import models
from framework.framework import Framework
import framework.ndf_kontschieder.utils as ndf_kontschieder_utils
import framework.resnet.utils as resnet_utils
from framework.utils.common import set_random_seed
from framework.utils.device import get_device


def main():
    parser = argparse.ArgumentParser(description="PyTorch NDT Framework")

    parser.add_argument(
        "--arch",
        type=str,
        choices=[
            "ndf",
            "ndt",
            "resnet18",
            "nbdt",
            "hard_nbdt",
        ],
        help="Model arch",
        required=True,
    )

    parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment",
        default="default_experiment",
        help="Experiment name",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="input batch size for training (default: 128)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="number of epochs to train (default: 50)",
    )

    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")

    # parser.add_argument(
    #     "--pretrained",
    #     action="store_true",
    #     help="Download pretrained model. Not all models support this.",
    # )

    parser.add_argument(
        "--eval", help="Eval only. For inference purpose", action="store_true"
    )

    parser.add_argument("--resume", help="Resume training", action="store_true")

    parser.add_argument("--model-path", help="Overrides checkpoint path generation")

    # NOTE: Args for experiments
    parser.add_argument(
        "--augmentation",
        "-aug",
        dest="augmentation",
        type=str,
        choices=[
            "gaussian_blur",
        ],
        help="Data augmentation",
    )

    parser.add_argument(
        "--train-type",
        type=str,
        choices=["const_learning", "inc_learning"],
        help="Training type variants",
    )

    args, _other_args = parser.parse_known_args()

    if args.train_type == "const_learning":
        parser.add_argument(
            "--apply_prob",
            default=0.4,
            help="Apply probability for partial transform",
        )
    elif args.train_type == "inc_learning":
        parser.add_argument(
            "--initial-noise",
            default=0.05,
            help="Initial noise percentage for incremental learning",
        )
        parser.add_argument(
            "--max-noise",
            default=0.95,
            help="Max noise percentage for incremental learning",
        )

    # NOTE: Optional args for different models
    if args.arch == "ndf" or args.arch == "ndt":
        ndf_kontschieder_utils.add_arguments(parser)
        args = parser.parse_args()
    elif args.arch == "resnet18" or args.arch == "nbdt" or args.arch == "hard_nbdt":
        resnet_utils.add_arguments(parser)
        args = parser.parse_args()

    args.device = get_device().type

    print(f"Input args: {vars(args)}")

    set_random_seed(int(args.seed), args.device == "cuda")

    Colors.cyan("==> Running a main script..")

    model = models[args.arch]
    framework = Framework(model=model, args=args)
    framework.run()


if __name__ == "__main__":
    main()
