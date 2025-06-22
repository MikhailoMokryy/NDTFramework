import os
from typing import Optional
import matplotlib
import matplotlib.pyplot as plt

from framework.utils.common import Colors

matplotlib.use("agg")


def imshow(
    img,
    ax=None,
    mean=(0.4914, 0.4822, 0.4465),
    std=(0.2023, 0.1994, 0.2010),
    save_path: Optional[str] = None,
):
    """Unnormalize and display an image."""
    img = img.clone().detach().numpy()
    for i in range(3):
        img[i] = img[i] * std[i] + mean[i]
    img = img.transpose((1, 2, 0))  # Convert from (C, H, W) to (H, W, C)

    if ax is None:
        plt.imshow(img)
        plt.axis("off")
    else:
        ax.imshow(img)
        ax.axis("off")

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        plt.imsave(save_path, img)
        Colors.cyan(f"Image saved to {save_path}")
    else:
        plt.show()
