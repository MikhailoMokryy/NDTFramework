from typing import Optional
from nbdt.data.custom import Dataset
from torch.utils import data
from torchvision import transforms, datasets


class CIFAR10:
    @staticmethod
    def transform_train():
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    @staticmethod
    def transform_test():
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    @staticmethod
    def train_loader(
        root="./data/cifar",
        transform: Optional[transforms.Compose] = None,
        batch_size=128,
    ):
        transform_train = (
            transform if transform is not None else CIFAR10.transform_train()
        )

        trainset = datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform_train
        )
        train_loader = data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )

        return train_loader

    @staticmethod
    def test_loader(
        root="./data/cifar",
        transform: Optional[transforms.Compose] = None,
        batch_size=100,
    ):
        transform_test = (
            transform if transform is not None else CIFAR10.transform_test()
        )

        testset = datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform_test
        )
        test_loader = data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        return test_loader

    @staticmethod
    def eval_loader(
        root="./data/cifar",
        transform: Optional[transforms.Compose] = None,
        batch_size=100,
    ):
        transform_eval = (
            transform if transform is not None else CIFAR10.transform_test()
        )

        testset = datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform_eval
        )
        test_loader = data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        return test_loader

    @staticmethod
    def get_dataset(
        root="./data/cifar",
        train=True,
        transform: Optional[transforms.Compose] = None,
    ):
        trainset = datasets.CIFAR10(
            root=root, train=train, download=True, transform=transform
        )

        return trainset

    @staticmethod
    def get_dataloader(trainset: Dataset, batch_size=128):
        train_loader = data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        return train_loader
