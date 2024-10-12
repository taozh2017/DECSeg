import os
import time

import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler


class BaseDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            txt_path=None,
            num=None,
            transform=None,
            ops_weak=None,
            ops_strong=None
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.txt_path = txt_path
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "test":
            with open(self._base_dir + "/test.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/train/{}.h5".format(case), "r")
        elif self.split == "val":
            h5f = h5py.File(self._base_dir + "/val/{}.h5".format(case), "r")
        elif self.split == "test":
            h5f = h5py.File(self._base_dir + "/test/test/{}.h5".format(case), "r")

        image = h5f["image"][:] / 255.0
        label = h5f["label"][:]
        label = np.where((label / 255.0) > 0.5, np.ones_like(label), np.zeros_like(label))
        sample = {"image": image, "label": label}

        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)

        sample["idx"] = idx
        return sample


def random_rot_flip(image, label=None):  # image:HW3,label:HW,feature:CHW
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(0, 1))
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k, axes=(0, 1))
        label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(brightness=0.2 * s, hue=0.05 * s)  # saturation=0.2 * s, contrast=0.2 * s,
    jitter_image = jitter(image)
    return jitter_image


class DECAugment(object):
    """returns ori, downsample and perturbation augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size
        self.totensor = transforms.ToTensor()

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image_ori, label_ori = self.resize(image, label)
        # weak augmentation is rotation / flip
        # image_ori, label_ori = random_rot_flip(image, label)
        image_down, label_down = self.down(image_ori, label_ori)
        # strong augmentation is color jitter
        image_per = color_jitter(image_ori).type("torch.FloatTensor")
        image_down_per = color_jitter(image_down).type("torch.FloatTensor")
        # fix dimensions
        image_ori = self.totensor(image_ori)
        image_down = self.totensor(image_down)
        label_ori = torch.from_numpy(label_ori.astype(np.uint8))
        label_down = torch.from_numpy(label_down.astype(np.uint8))

        sample_new = {
            "image": image_ori,
            "image_per": image_per,
            "image_down": image_down,
            "image_down_per": image_down_per,
            "label": label_ori,
            "label_down": label_down,
        }
        return sample_new

    def resize(self, image, label=None):
        x, y, z = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
        if label is not None:
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        return image, label

    def down(self, image, label=None):
        # image = zoom(image, (1 / 2, 1 / 2, 1), order=0)
        # if label is not None:
        #     label = zoom(label, (1 / 2, 1 / 2), order=0)
        outsize = round(self.output_size[0] * 0.5 / 32) * 32
        image = zoom(image, (outsize / self.output_size[0], outsize / self.output_size[1], 1), order=0)
        if label is not None:
            label = zoom(label, (outsize / self.output_size[0], outsize / self.output_size[1]), order=0)
        return image, label


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
            grouper(primary_iter, self.primary_batch_size),
            grouper(secondary_iter, self.secondary_batch_size),
        )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
