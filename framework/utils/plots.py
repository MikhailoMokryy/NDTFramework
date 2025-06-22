from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import os
import json
import numpy as np

from framework.utils.common import Colors


# TODO: Add grapic
def plot_accuracy(jsonfiles, model_names=[], figsize=(5, 5), ymax=100.0, title=""):
    plt.figure(figsize=figsize)
    color = ["b", "g", "r", "c", "m", "y", "k", "w"]

    if not (model_names):
        model_names = [str(i) for i in range(len(jsonfiles))]

    for i, f in enumerate(jsonfiles):
        # load the information:
        records = json.load(open(f, "r"))

        # Plot train/test loss
        plt.plot(
            np.arange(len(records["test_epoch_accuracy"]), dtype=float),
            np.array(records["test_epoch_accuracy"]),
            color=color[i],
            linestyle="-",
            label=model_names[i],
        )
        # print(records['train_epoch_accuracy'])
        plt.ylabel("test accuracy (%)")
        plt.xlabel("epoch number")
        plt.ylim(ymax=ymax)
        print(
            model_names[i]
            + ": accuracy = {}".format(max(records["test_epoch_accuracy"]))
        )
    plt.legend(loc="lower right")
    plt.title(title)


def plot_models_accuracy(
    x_values: List[int],
    models: Dict[str, Tuple[str, List[float]]],
    title: str = "CNN Model Accuracy vs Metaparam Value",
    save_path: Optional[str] = None,
) -> None:
    plt.figure(figsize=(12, 6))

    color = ["b", "g", "r", "c", "m", "y", "k", "w"]
    colors = 5
    model_lines = []

    for i, (model_name, value) in enumerate(models.items()):
        linestyle, acc = value
        if i < colors:
            model_lines.append(
                Patch(
                    color=color[i % colors],
                    linestyle=linestyle,
                    label=model_name,
                )
            )
        plt.plot(
            x_values,
            acc,
            marker="o",
            color=color[i % colors],
            linestyle=linestyle,
            label=model_name,
        )

    plot_lines = [
        Line2D([0], [0], color="black", linestyle="-", label="Clean"),
        Line2D([0], [0], color="black", linestyle="--", label="Noise"),
    ]

    plt.title(
        title,
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Noise probability, p (%)")
    plt.ylabel("Accuracy (%)")
    plt.xticks(x_values)
    plt.grid(True, alpha=0.3)

    # Create model legend (e.g. different models, colors)
    model_legend = plt.legend(
        handles=model_lines,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        title="Models",
    )

    # Create plot legend (e.g. line style explanation)
    plot_legend = plt.legend(
        handles=plot_lines,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.5),
        title="Test set",
    )

    plt.gca().add_artist(model_legend)
    plt.gca().add_artist(plot_legend)

    plt.subplots_adjust(right=0.85)

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")  # Removed Colors.cyan dependency
    else:
        plt.show()


def plot_training_comparison(file1_path, file2_path, save_path=None):
    """
    Compare training loss between two different training approaches.

    Args:
        file1_path (str): Path to first training log JSON file
        file2_path (str): Path to second training log JSON file
        save_path (str, optional): Path to save the plot
    """

    # Load the data
    with open(file1_path, "r") as f:
        data1 = json.load(f)

    with open(file2_path, "r") as f:
        data2 = json.load(f)

    # Extract loss data
    train_loss_1 = data1["train_loss"]
    val_loss_1 = data1["val_loss"]
    train_loss_2 = data2["train_loss"]
    val_loss_2 = data2["val_loss"]

    # Create epoch arrays
    epochs_1 = list(range(1, len(train_loss_1) + 1))
    epochs_2 = list(range(1, len(train_loss_2) + 1))

    # Determine approach names based on file names and data characteristics
    approach_1_name = "Constant noise learning (p=0.6)"
    approach_2_name = "Incremental noise learning"

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "NBDT: Constant noise learning vs Incremental noise learning",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Training Loss Comparison
    ax1.plot(
        epochs_1, train_loss_1, "b-", linewidth=2, label=approach_1_name, alpha=0.8
    )
    ax1.plot(
        epochs_2, train_loss_2, "r-", linewidth=2, label=approach_2_name, alpha=0.8
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation Loss Comparison
    ax2.plot(epochs_1, val_loss_1, "b-", linewidth=2, label=approach_1_name, alpha=0.8)
    ax2.plot(epochs_2, val_loss_2, "r-", linewidth=2, label=approach_2_name, alpha=0.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # # Plot 3: Training vs Validation Loss for Approach 1
    # ax3.plot(epochs_1, train_loss_1, "b-", linewidth=2, label="Training Loss")
    # ax3.plot(epochs_1, val_loss_1, "b--", linewidth=2, label="Validation Loss")
    # ax3.set_xlabel("Epoch")
    # ax3.set_ylabel("Loss")
    # ax3.set_title(f"{approach_1_name}\nTraining vs Validation Loss")
    # ax3.legend()
    # ax3.grid(True, alpha=0.3)
    #
    # # Plot 4: Training vs Validation Loss for Approach 2
    # ax4.plot(epochs_2, train_loss_2, "r-", linewidth=2, label="Training Loss")
    # ax4.plot(epochs_2, val_loss_2, "r--", linewidth=2, label="Validation Loss")
    # ax4.set_xlabel("Epoch")
    # ax4.set_ylabel("Loss")
    # ax4.set_title(f"{approach_2_name}\nTraining vs Validation Loss")
    # ax4.legend()
    # ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        Colors.cyan(f"Plot saved to: {save_path}")
    else:
        plt.show()
