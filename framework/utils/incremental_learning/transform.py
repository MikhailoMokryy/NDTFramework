import random
import torch

from framework.utils.common import Colors


class IncrementalNoiseTransform:
    """
    Transform that incrementally increases the number of samples that have noise applied
    based on the current epoch number.

    This allows for a curriculum learning approach where the model gradually learns
    to handle more noisy samples as training progresses.

    Compatible with PyTorch DataLoader batching.
    """

    def __init__(
        self, transform, initial_noise_prob=0.05, max_noise_prob=0.95, total_epochs=40
    ):
        """
        Initialize the incremental noise transform.

        Args:
            transform: The transform to apply (e.g., iaa.GaussianBlur)
            initial_noise_prob: Initial probability of applying noise (default: 0.05)
            max_noise_prob: Maximum probability of applying noise (default: 0.95)
            total_epochs: Total number of epochs for training (default: 40)
        """
        self.transform = transform
        self.initial_noise_prob = float(initial_noise_prob)
        self.max_noise_prob = float(max_noise_prob)
        self.total_epochs = int(total_epochs)

        # Current noise probability
        self.current_noise_prob = self.initial_noise_prob
        self.current_epoch = 0
        self.total_samples = 0

        # Keep track of which samples have noise applied using dataset indices
        self.noisy_indices = set()

    def __call__(self, x, idx=None):
        """
        Apply the transform to input x based on whether its index is in noisy_indices.

        Args:
            x: Input sample (can be a single sample or a batch)
            idx: Index or indices of the sample(s) in the dataset

        Returns:
            Transformed sample(s) if in noisy_indices, otherwise original sample(s)
        """
        # Handle batched data
        if (
            isinstance(x, torch.Tensor) and len(x.shape) > 3
        ):  # Assuming image data with batch dimension
            return self._apply_to_batch(x, idx)
        else:
            # Single sample processing
            return self._apply_to_single(x, idx)

    def _apply_to_single(self, x, idx):
        """Apply transform to a single sample if its index is in noisy_indices"""
        if idx is None:
            # Fall back to random probability if no index provided
            return self.transform(x) if random.random() < self.current_noise_prob else x

        # Apply noise only if this sample is in our noisy_indices set
        if idx in self.noisy_indices:
            return self.transform(x)

        # No noise applied
        return x

    def _apply_to_batch(self, batch, indices):
        """Apply transform to samples in a batch whose indices are in noisy_indices"""
        # If indices is None, we can't ensure consistency
        if indices is None:
            # Apply randomly based on current probability
            mask = torch.rand(len(batch)) < self.current_noise_prob
            result = batch.clone()
            for i in range(len(batch)):
                if mask[i]:
                    result[i] = self.transform(batch[i])
            return result

        # Process each sample in the batch based on its index
        result = []
        for i, (sample, idx) in enumerate(zip(batch, indices)):
            # If in noisy_indices, apply transform
            if idx in self.noisy_indices:
                result.append(self.transform(sample))
            else:
                result.append(sample)

        return torch.stack(result) if isinstance(batch, torch.Tensor) else result

    def calculate_noise_percentage(self, epoch):
        """
        Calculate the target noise percentage for the given epoch.

        Args:
            epoch: Current epoch number (0-indexed)

        Returns:
            Target noise percentage for this epoch
        """
        # Linear interpolation from initial to max noise probability
        progress = (
            min(epoch / (self.total_epochs - 1), 1.0) if self.total_epochs > 1 else 1.0
        )
        target_percentage = self.initial_noise_prob + progress * (
            self.max_noise_prob - self.initial_noise_prob
        )
        return target_percentage

    def add_noise(self, epoch):
        """
        Add noise to the dataset based on the current epoch.
        Importantly, we preserve existing noisy samples and only add new ones.

        Args:
            epoch: Current epoch number (1-indexed)

        Returns:
            Target noise percentage for this epoch
        """
        self.current_epoch = epoch - 1  # Convert to 0-indexed

        # Calculate target noise percentage for the current epoch
        target_percentage = self.calculate_noise_percentage(self.current_epoch)

        # Calculate number of samples that should be noisy after this call
        target_noise_samples = int(target_percentage * self.total_samples)

        # Calculate how many new noisy samples we need to add
        num_new_samples = target_noise_samples - len(self.noisy_indices)

        if num_new_samples <= 0:
            Colors.green(
                f"Epoch {epoch}: No new noisy samples needed. Current noise: {len(self.noisy_indices) / self.total_samples:.2%}"
            )
            return target_percentage

        # Get indices of samples that are not yet noisy
        non_noisy_indices = list(set(range(self.total_samples)) - self.noisy_indices)

        # Select new indices to make noisy
        new_noisy_indices = random.sample(
            non_noisy_indices, min(num_new_samples, len(non_noisy_indices))
        )

        # Add these to our set of noisy indices
        self.noisy_indices.update(new_noisy_indices)

        Colors.green(
            f"Epoch {epoch}: Added {len(new_noisy_indices)} new noisy samples. "
            + f"Total noisy samples: {len(self.noisy_indices)} ({len(self.noisy_indices) / self.total_samples:.2%})"
        )

        # Update current noise probability
        self.current_noise_prob = target_percentage

        return target_percentage

    def reset_noise_indices(self):
        """
        Reset the set of samples that have noise applied.
        This can be called when starting a new training run.
        """
        self.noisy_indices = set()
        self.current_noise_prob = self.initial_noise_prob
        self.current_epoch = 0
