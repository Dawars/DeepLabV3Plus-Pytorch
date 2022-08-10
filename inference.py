import argparse

import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Compose

from datasets.labelmefacade import LabelMeFacade
from model import DeepLab
import os

OUT_PATH = '/mnt/hdd/datasets/facade/output/etrims/test'
os.makedirs(OUT_PATH, exist_ok=True)


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet101',
                        choices=['resnet101', 'mobilenet'], help='backbone name')
    parser.add_argument('--ckpt_path', type=str,
                        default='/mnt/hdd/datasets/facade/experiments/deeplab/ckpts/deeplabv3plus_resnet101_labelmefacade_best_hyper/trial_0/epoch=79.ckpt',
                        help='path of ckpt to load')

#    '/mnt/hdd/datasets/facade/experiments/deeplab/ckpts/deeplabv3plus_resnet101_labelmefacade_best_hyper/trial_0/epoch=79.ckpt' if ckpt_mode == 'hand' else \
#    '/mnt/hdd/datasets/facade/experiments/deeplab/ckpts/deeplabv3plus_resnet101_labelmefacade_batchsize2/trial_24/epoch=64.ckpt'  # best ce

    return parser.parse_args()


def main():
    opts = get_argparser()
    model = DeepLab.load_from_checkpoint(opts.ckpt_path, hparams=opts)
    trainer = Trainer(gpus=1)

    val_transform = Compose([
        torchvision.transforms.CenterCrop(512),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])

    datasets = []
    for iteration in range(3):
        split_file = f"/mnt/hdd/datasets/facade/bootstrapping/split_{iteration}.txt"
        dataset = LabelMeFacade('/mnt/hdd/datasets/facade/ZuBuD/ZuBuD/png-ZuBuD', 'test', transform=val_transform,
                            split_file=split_file, img_dir='', ext='.png')
        datasets.append(dataset)
    dataset_inference = ConcatDataset(datasets)

    dataloader = DataLoader(dataset_inference, batch_size=len(dataset_inference), shuffle=False, num_workers=4,
                            pin_memory=True,
                            drop_last=False)

    predictions = trainer.predict(model, dataloaders=dataloader)
    for batch in predictions:
        for i in range(len(batch[1])):
            pred = batch[0][i]
            name = batch[1][i]
            mask = datasets[0].decode_target(pred).astype("uint8")

            Image.fromarray(mask).save(os.path.join(OUT_PATH, f"{name}.png"))

    # decode
    print(len(predictions))


if __name__ == '__main__':
    main()
