import argparse
from types import SimpleNamespace

import imageio
import torchvision
import tqdm
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import argparse
import os

import optuna

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger

import pytorch_lightning as pl

from datasets.listdataset import ListDataset
from utils import ext_transforms as et

# data
from datasets import Cityscapes
from datasets.labelmefacade import LabelMeFacade
from model import DeepLab
from datasets.labelmefacade import LabelMeFacade
from model import DeepLab

val_transform = et.ExtCompose([
    et.ExtCenterCrop(512),
    et.ExtToTensor(),
    et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])

inference_transform = Compose([
    torchvision.transforms.CenterCrop(512),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
])

dataset_val = LabelMeFacade('/mnt/hdd/datasets/facade/labelmefacade', 'val', transform=val_transform)


def main():
    ckpt_path = '/mnt/hdd/datasets/facade/experiments/deeplab/ckpts/deeplabv3plus_resnet101_labelmefacade_best_hyper/trial_0/epoch=79.ckpt'

    exp_name = 'test'

    for i in range(3):
        ckpt_path = bootstrapping_iteration(ckpt_path, exp_name, i)


def bootstrapping_iteration(ckpt_path, exp_name, iteration):
    print(f"Starting iteration {iteration}")

    mode = 'last'  # or 'best'

    save_path = f"/mnt/hdd/datasets/facade/bootstrapping/{exp_name}/split_{iteration}"
    label_path = os.path.join(save_path, 'label')
    split_file = f"/mnt/hdd/datasets/facade/bootstrapping/split_{iteration}.txt"

    hparams = argparse.Namespace(**{'backbone': 'resnet101',
                                    'val_batch_size': 1,
                                    'batch_size': 8,
                                    'lr': 1e-05})

    train_transform = et.ExtCompose([
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtRandomRotation(15),
        et.ExtRandomScale([1, 1.5]),
        et.ExtRandomCrop(512),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    # create now, load images later when they exist
    dataset_train = LabelMeFacade('/mnt/hdd/datasets/facade/ZuBuD/ZuBuD/png-ZuBuD', 'train', transform=train_transform,
                                  label_root=label_path, split_file=split_file, img_dir='')

    model = DeepLab.load_from_checkpoint(ckpt_path, hparams=hparams, train_dataset=dataset_train,
                                         val_dataset=dataset_val)
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(save_path, 'ckpts'),
                                          filename='{epoch:d}',
                                          monitor='train/ce',
                                          mode='max',
                                          save_top_k=-1 if mode == 'last' else 1)

    logger = TestTubeLogger(save_dir=os.path.join(save_path, 'logs'),
                            name=exp_name,
                            # debug=opts.debug,
                            version=iteration,
                            create_git_tag=True,
                            log_graph=False)

    trainer = Trainer(gpus=1,
                      max_epochs=80,
                      checkpoint_callback=True,
                      callbacks=[checkpoint_callback],
                      # resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      # accelerator='ddp' if hparams.num_gpus > 1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple",  # if hparams.num_gpus == 1 else None,)
                      )
    # inference
    print(f"Inference iteration {iteration}")
    dataset_inference = ListDataset('/mnt/hdd/datasets/facade/ZuBuD/ZuBuD/png-ZuBuD',
                                    file_list_path=f"/mnt/hdd/datasets/facade/bootstrapping/split_{iteration}.txt",
                                    transform=inference_transform)

    dataloader = DataLoader(dataset_inference, batch_size=64, shuffle=False, num_workers=4,
                            pin_memory=True,
                            drop_last=False)

    os.makedirs(label_path, exist_ok=True)

    predictions = trainer.predict(model, dataloaders=dataloader)
    for img_batch, filename_batch in tqdm.tqdm(predictions):
        for i in range(len(img_batch)):
            pred = img_batch[i]
            filename = filename_batch[i]
            mask = dataset_val.decode_target(pred).astype("uint8")
            imageio.imsave(os.path.join(label_path, f"{filename}.png"), mask)

    print(f"Training iteration {iteration}")
    # train
    random_seed = 42
    pl.seed_everything(random_seed)

    # train
    trainer.fit(model)

    print(f"Finished iteration {iteration}")

    return sorted(os.listdir(os.path.join(save_path, 'ckpts')))[-1]


if __name__ == '__main__':
    main()
