from argparse import Namespace
from typing import Callable, Optional
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from framework.utils.common import Colors
import random


class IncrementalNoiseCIFAR10(Dataset):
    def __init__(
        self,
        dataset: CIFAR10,
        args: Namespace,
        aug_transform: Callable,
        transform: Optional[Callable] = None,
    ):
        self.base_dataset = dataset
        self.transform = transform
        self.aug_transform = aug_transform
        self.noisy_indices = set()  # Keep track of indices of noisy samples
        # self.gaussian_blur = augmenters[args.augmentation]
        self.initial_noise_percentage = float(args.initial_noise)
        self.max_noise_percentage = float(args.max_noise)
        self.total_epochs = int(args.epochs) - 1
        self.current_epoch = 0
        self.noise_increment_per_epoch = (
            self.max_noise_percentage - self.initial_noise_percentage
        ) / self.total_epochs

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, target = self.base_dataset[idx]

        # If this is a noisy sample, apply Gaussian blur
        if idx in self.noisy_indices:
            # # Convert PIL Image to numpy array
            # img_np = np.array(img)
            # # Apply Gaussian blur using imgaug
            # img_np = self.gaussian_blur.augment_image(img_np)
            # # Convert back to PIL Image
            # img = Image.fromarray(img_np)
            img = self.aug_transform(img)
        elif self.transform is not None:
            # Apply the transform to the image
            # Note: The transform should be compatible with PIL Images
            img = self.transform(img)

        return img, target

    def calculate_noise_percentage(self, epoch: int):
        """
        Calculate the target noise percentage for the given epoch
        """
        # Start with initial_noise_percentage and increment by noise_increment_per_epoch each epoch
        target_percentage = (
            self.initial_noise_percentage + epoch * self.noise_increment_per_epoch
        )
        # Cap at max_noise_percentage to avoid having all noisy samples
        target_percentage = min(target_percentage, self.max_noise_percentage)

        return round(target_percentage, 5)

    def add_noise(self, epoch: int):
        """
        Add noise to the dataset based on the current epoch
        Importantly, we preserve existing noisy samples and only add new ones
        """
        self.current_epoch = epoch - 1

        # Calculate target noise percentage for the current epoch
        target_percentage = self.calculate_noise_percentage(self.current_epoch)
        total_samples = len(self.base_dataset)

        # Calculate number of samples that should be noisy after this call
        target_noise_samples = int(target_percentage * total_samples)

        # Calculate how many new noisy samples we need to add
        num_new_samples = target_noise_samples - len(self.noisy_indices)

        if num_new_samples <= 0:
            Colors.green(
                f"Epoch {self.current_epoch + 1}: No new noisy samples needed. Current noise: {len(self.noisy_indices) / total_samples:.2%}"
            )
            return target_percentage

        # Get indices of samples that are not yet noisy
        non_noisy_indices = list(set(range(total_samples)) - self.noisy_indices)

        # Select new indices to make noisy
        new_noisy_indices = random.sample(
            non_noisy_indices, min(num_new_samples, len(non_noisy_indices))
        )

        # Add these to our set of noisy indices
        self.noisy_indices.update(new_noisy_indices)

        Colors.green(
            f"Epoch {self.current_epoch + 1}: Added {len(new_noisy_indices)} new noisy samples. "
            + f"Total noisy samples: {len(self.noisy_indices)} ({len(self.noisy_indices) / total_samples:.2%})"
        )

        return target_percentage

    def get_current_noise_percentage(self):
        """
        Return the current percentage of noisy samples
        """
        return len(self.noisy_indices) / len(self.base_dataset)

    def get_noisy_indices(self):
        return self.noisy_indices
