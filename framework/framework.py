from argparse import Namespace
from framework.datasets import CIFAR10
from framework.models import ModelsType
from framework.utils.augmentations import (
    ImageNoise,
    PartialTransform,
    aug_transform_test,
    aug_transform_train,
    augmenters,
)
from framework.utils.common import Colors
from framework.utils.framework_model import FrameworkModel, PartialFrameworkModel
from framework.utils.incremental_learning.dataset import (
    IncrementalNoiseCIFAR10,
)


class Framework:
    def __init__(
        self,
        model: ModelsType,
        args: Namespace,
    ):
        self.args = args
        self.model = model(args)

    def run(self):
        if self.args.eval:
            acc = self.test()
            Colors.green(f"Model accuracy: {acc:.3f}")
        else:
            self.train()

    def train(self):
        if isinstance(self.model, FrameworkModel):
            if self.args.augmentation is None:
                train_loader = CIFAR10.train_loader(batch_size=self.args.batch_size)
                eval_loader = CIFAR10.eval_loader()

                self.model.train(train_loader, eval_loader)
            elif self.args.train_type == "const_learning":
                noise = ImageNoise(augmenter=augmenters[self.args.augmentation])
                part_transform = PartialTransform(noise, self.args.apply_prob)
                aug_transform = aug_transform_train(part_transform)

                train_loader = CIFAR10.train_loader(
                    batch_size=self.args.batch_size, transform=aug_transform
                )
                eval_loader = CIFAR10.eval_loader()

                self.model.train(train_loader, eval_loader)
            elif self.args.train_type == "inc_learning":
                noise = ImageNoise(augmenter=augmenters[self.args.augmentation])
                trainset = IncrementalNoiseCIFAR10(
                    dataset=CIFAR10.get_dataset(),
                    transform=CIFAR10.transform_train(),
                    aug_transform=aug_transform_train(noise),
                    args=self.args,
                )

                train_loader = CIFAR10.get_dataloader(
                    trainset, batch_size=self.args.batch_size
                )
                eval_loader = CIFAR10.eval_loader()
                self.model.train(train_loader, eval_loader)

        elif isinstance(self.model, PartialFrameworkModel):
            self.model.train()

    def test(self):
        if self.args.augmentation is None:
            test_loader = CIFAR10.test_loader()
        else:
            noise = ImageNoise(augmenter=augmenters[self.args.augmentation])
            aug_transform = aug_transform_test(noise)
            test_loader = CIFAR10.test_loader(transform=aug_transform)

        return self.model.test(test_loader)
