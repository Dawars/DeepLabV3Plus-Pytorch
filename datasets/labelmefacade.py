import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class LabelMeFacade(data.Dataset):
    """LabelMeFacade <http://www.cityscapes-dataset.com/> Dataset.

    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    LabelMeFacadeClass = namedtuple('LabelMeFacadeClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        LabelMeFacadeClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        LabelMeFacadeClass('building',             1, 2, 'construction', 2, False, False, (128, 0, 0)),
        LabelMeFacadeClass('sky',                  2, 3, 'sky', 5, False, False, (0, 128, 128)),
        LabelMeFacadeClass('car',                  3, 4, 'vehicle', 7, True, False, (128, 0, 128)),
        LabelMeFacadeClass('door',                 4, 5, 'construction', 2, True, False, (128, 128, 0)),
        LabelMeFacadeClass('sidewalk',             5, 6, 'flat', 1, False, False, (128, 128, 128)),
        LabelMeFacadeClass('road',                 6, 7, 'flat', 1, False, False, (128, 64, 0)),
        LabelMeFacadeClass('vegetation',           7, 8, 'nature', 4, False, False, (107, 142, 35)),
        LabelMeFacadeClass('window',               8, 9, 'construction', 2, True, False, (0, 0, 128)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    #train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),

    #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
    #train_id_to_color = np.array(train_id_to_color)
    #id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

    def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None):
        self.root = os.path.expanduser(root)
        self.mode = 'gtFine'
        self.target_type = target_type
        self.images_dir = os.path.join(self.root, 'images')
        self.targets_dir = os.path.join(self.root, 'labels')
        self.transform = transform

        self.split = split
        self.images = []
        self.targets = []

        self.colormap = []
        for c in self.classes:
            self.colormap.append(c.color)

        self.colormap2label = np.zeros(256 ** 3)
        for c in self.classes:
            self.colormap2label[(c.color[0] * 256 + c.color[1]) * 256 + c.color[2]] = c.id

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        split_file = os.path.join(self.root, f"{split}.txt")

        with open(split_file) as f:
            for filename in f.readlines():
                filename = filename.strip()
                self.images.append(os.path.join(self.images_dir, filename+".jpg"))
                self.targets.append(os.path.join(self.targets_dir, filename+".png"))

    def encode_target(self, target):
        data = np.array(target, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.colormap2label[idx], dtype='int64')

    def decode_target(self, target):
        i = target.shape[0]
        j = target.shape[1]
        rgblbl = np.zeros((i, j, 3))
        for index_i in range(i):
            for index_j in range(j):
                rgblbl[index_i, index_j, :] = self.colormap[target[index_i, index_j]]
        return rgblbl

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])
        if self.transform:
            image, target = self.transform(image, target)
        target = self.encode_target(target)
        return image, target

    def __len__(self):
        return len(self.images)
