import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import os
import argparse
from torch.utils.data import DataLoader
from framework.utils.common import Colors, progress_bar
from framework.utils.device import get_device
from datetime import datetime
import json


def load_model(net: nn.Module, path: str, device: str):
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if os.path.exists(path):
        map_location = get_device()

        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        state_dict = checkpoint["net"]

        if device != "cuda":
            new_state_dict = {}
            for key in state_dict:
                new_key = (
                    key.replace("module.", "") if key.startswith("module.") else key
                )
                new_state_dict[new_key] = state_dict[key]
            net.load_state_dict(new_state_dict)
        else:
            net.load_state_dict(state_dict)
    else:
        Colors.red(f"Checkpoint file {path} does not exist.")
        raise FileNotFoundError(f"Checkpoint file {path} does not exist.")

    net.to(device)

    accuracy: float = checkpoint["acc"]
    epoch: int = checkpoint["epoch"]
    args = checkpoint["args"]

    Colors.green(
        f"Model loaded from {path} with accuracy {accuracy:.6f} on epoch {epoch} with following args: {args}."
    )

    return {"net": net, "acc": accuracy, "epoch": epoch}


def save_model(net: nn.Module, accuracy: float, epoch: int, args: argparse.Namespace):
    Colors.green(f"Saving model on epoch {epoch} with test acc {accuracy:.6f}...")
    state = {
        "net": net.state_dict(),
        "acc": accuracy,
        "epoch": epoch,
        "args": vars(args),
    }

    os.makedirs(f"checkpoints/{args.experiment}/{args.arch}", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d")

    torch.save(
        state,
        "./checkpoints/{}/{}/model-{}.pth".format(
            args.experiment, args.arch, timestamp
        ),
    )


def test_model(net: nn.Module, dataloader: DataLoader, device: str) -> float:
    net.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = net(data)

            _, pred = output.data.max(1)

            correct += pred.eq(target).cpu().sum().item()
            total += target.size(0)

            progress_bar(
                batch_idx,
                len(dataloader),
                "Acc: %.3f%% (%d/%d)"
                % (
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    return 100.0 * correct / total


def save_logs(args: argparse.Namespace, records: dict):
    timestamp = datetime.now().strftime("%Y-%m-%d")

    with open(
        "./checkpoints/{}/{}/logs-{}.json".format(
            args.experiment, args.arch, timestamp
        ),
        "w",
    ) as f:
        json.dump(records, f, indent=4)


def load_logs(records_path: str) -> dict:
    with open(records_path, "r") as f:
        records = json.load(f)

    return records
