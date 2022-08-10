import argparse
import glob
import json

import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Compose

from datasets.labelmefacade import LabelMeFacade
from model import DeepLab
import os
from utils import ext_transforms as et

OUT_PATH = '/mnt/hdd/datasets/facade/output/etrims/'
os.makedirs(OUT_PATH, exist_ok=True)


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet101',
                        choices=['resnet101', 'mobilenet'], help='backbone name')
    parser.add_argument('--ckpt_path', type=str,
                        default='/mnt/hdd/datasets/facade/experiments/deeplab/ckpts/deeplabv3plus_resnet101_labelmefacade_best_hyper/trial_0/epoch=79.ckpt',
                        help='path of ckpt to load')

    return parser.parse_args()


def run_validation(ckpt_path, exp_name):
    print(f"----------- {exp_name} --------------")
    logger = TestTubeLogger(save_dir=os.path.join(OUT_PATH),
                            name=exp_name,
                            log_graph=False)
    opts = get_argparser()
    opts.ckpt_path = ckpt_path
    model = DeepLab.load_from_checkpoint(ckpt_path, hparams=opts)
    trainer = Trainer(gpus=1, logger=logger)

    val_transform = et.ExtCompose([
        et.ExtCenterCrop(512),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    dataset_inference = LabelMeFacade('/mnt/hdd/datasets/facade/etrim/etrims-db_v1', 'val', transform=val_transform,
                                      img_dir='images/08_etrims-ds', label_dir='annotations/08_etrims-ds')

    dataloader = DataLoader(dataset_inference, batch_size=1, shuffle=False, num_workers=4,
                            pin_memory=True,
                            drop_last=False)
    model.val_dataset = dataset_inference
    predictions = trainer.validate(model, dataloaders=dataloader)
    with open(os.path.join(OUT_PATH, exp_name, 'val_error.json'), 'w') as f:
        f.write(json.dumps(predictions))

    # todo save
    # for batch in predictions:
    #     for i in range(len(batch[1])):
    #         pred = batch[0][i]
    #         name = batch[1][i]
    #         mask = dataset_inference.decode_target(pred).astype("uint8")

    #         Image.fromarray(mask).save(os.path.join(OUT_PATH, f"{name}.png"))


def main():
    for md in ['original', 'last', 'best']:
        for data in ['all']:
            for ckpt in ['hand', 'ce']:
                ckpt_name = f'labelmefacade_{ckpt}_{data}_{md}'
                exp_name = f'labelmefacade_{ckpt}_{data}_{md}'

                ckpt_path = sorted(glob.glob(f'/mnt/hdd/datasets/facade/bootstrapping/{ckpt_name}/split_2/ckpts/*'))[-1]
                print(ckpt_path)
                run_validation(ckpt_path, exp_name)

    run_validation(
        '/mnt/hdd/datasets/facade/experiments/deeplab/ckpts/deeplabv3plus_resnet101_labelmefacade_best_hyper/trial_0/epoch=79.ckpt',
        'fine_tune_hand')
    run_validation(
        '/mnt/hdd/datasets/facade/experiments/deeplab/ckpts/deeplabv3plus_resnet101_labelmefacade_batchsize2/trial_24/epoch=64.ckpt',
        'fine_tune_ce')


if __name__ == '__main__':
    main()
