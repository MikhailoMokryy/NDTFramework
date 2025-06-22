import numpy as np
import imgaug.augmenters as iaa
from PIL import Image
from torchvision import transforms
import random


class PartialTransform:
    def __init__(self, transform, apply_prob: float):
        self.transform = transform
        self.apply_prob = float(apply_prob)

    def __call__(self, x):
        if random.random() < self.apply_prob:
            return self.transform(x)
        return x


class OldImageNoise:
    def __init__(self, augmenter: iaa.GaussianBlur):
        self.aug = iaa.Sequential([augmenter])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug(image=img)
        return Image.fromarray(np.asarray(img))


class ImageNoise:
    def __init__(self, augmenter: iaa.GaussianBlur):
        self.aug = augmenter

    def __call__(self, img):
        img = np.array(img)
        np_img = self.aug.augment_image(img)
        return Image.fromarray(np_img)


def aug_transform_train(transform):
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform,
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


def aug_transform_test(transform):
    return transforms.Compose(
        [
            transform,
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


augmenters = dict(
    {
        "gaussian_blur": iaa.GaussianBlur(sigma=0.8),
    }
)
