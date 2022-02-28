import argparse
import os

import torch
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from utils import ext_transforms as et

# data
from datasets import Cityscapes
from datasets.labelmefacade import LabelMeFacade
from model import DeepLab


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=32,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=8,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Set random seed everywhere')
    parser.add_argument('--save_path', type=str, default='/mnt/hdd/datasets/facade/experiments/deeplab/',
                        help='paths to save checkpoints and logs to')
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    return parser.parse_args()


def main(opts):
    pl.seed_everything(opts.random_seed)

    train_transform = et.ExtCompose([
        et.ExtResize([512, 512]),
        # et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    is_debug = False

    val_transform = et.ExtCompose([
        et.ExtResize([512, 512]),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    dataset_train = LabelMeFacade('/mnt/hdd/datasets/facade/labelmefacade', 'train', transform=train_transform)
    dataset_val = LabelMeFacade('/mnt/hdd/datasets/facade/labelmefacade', 'val', transform=val_transform)
    train_loader = DataLoader(dataset_train, batch_size=opts.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=opts.val_batch_size, drop_last=True)
    # model
    model = DeepLab(opts)
    # training

    checkpoint_callback = \
        ModelCheckpoint(dirpath=os.path.join(opts.save_path, 'ckpts', opts.exp_name),
                        filename='{epoch:d}',
                        monitor='train/ce',
                        mode='max',
                        save_top_k=-1)

    logger = TestTubeLogger(save_dir=os.path.join(opts.save_path, 'logs'),
                            name=opts.exp_name,
                            # debug=opts.debug,
                            create_git_tag=False,
                            log_graph=False)

    trainer = pl.Trainer(gpus=None if is_debug else 1,
                         max_epochs=70,
                         checkpoint_callback=True,
                         callbacks=[checkpoint_callback],
                         # resume_from_checkpoint=hparams.ckpt_path,
                         logger=logger,
                         weights_summary=None,
                         progress_bar_refresh_rate=1,
                         # accelerator='ddp' if hparams.num_gpus > 1 else None,
                         num_sanity_val_steps=1,
                         benchmark=True,
                         profiler="simple",  # if hparams.num_gpus == 1 else None,
)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    hparams = get_argparser()
    main(hparams)
