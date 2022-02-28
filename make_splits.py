import os
import random

import torch

path = "/mnt/hdd/datasets/facade/labelmefacade/"

filenames = sorted(os.listdir(os.path.join(path, 'images')))
random.shuffle(filenames)

train_length = int(0.7 * len(filenames))
val_length = int(0.2 * len(filenames))
test_length = len(filenames) - train_length - val_length

with open(os.path.join(path, 'train.txt'), 'w') as f:
    for name in filenames[:train_length]:
        f.write(name[:-len('.jpg')])
        f.write('\n')

with open(os.path.join(path, 'val.txt'), 'w') as f:
    for name in filenames[train_length:train_length + val_length]:
        f.write(name[:-len('.jpg')])
        f.write('\n')

with open(os.path.join(path, 'test.txt'), 'w') as f:
    for name in filenames[train_length + val_length:]:
        f.write(name[:-len('.jpg')])
        f.write('\n')
