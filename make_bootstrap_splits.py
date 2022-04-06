import os
import random

import torch

path = "/mnt/hdd/datasets/facade/ZuBuD/ZuBuD/png-ZuBuD"

filenames = sorted(os.listdir(path))
random.shuffle(filenames)
length = len(filenames)
num_batch = 3

out_path = '/mnt/hdd/datasets/facade/bootstrapping'

for split in range(num_batch):
    with open(os.path.join(out_path, f"split_{split}.txt"), 'w') as f:
        start = split * (length // num_batch)
        end = (split + 1) * (length // num_batch)

        for i in range(start, end):
            f.write(f"{filenames[i][:-len('.jpg')]}\n")
