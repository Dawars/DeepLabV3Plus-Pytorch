import argparse

import torchvision
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from datasets.labelmefacade import LabelMeFacade
from model import DeepLab
from utils import ext_transforms as et

CKPT_PATH = '/mnt/hdd/datasets/facade/experiments/deeplab/ckpts/deeplabv3plus_resnet101_labelmefacade/trial_14/epoch=79.ckpt'


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='resnet101',
                        choices=['resnet101', 'mobilenet'], help='backbone name')
    parser.add_argument('--load_ckpt_path', type=str,
                        default='/mnt/hdd/datasets/facade/experiments/deeplab/ckpts/deeplabv3plus_resnet101_labelmefacade/trial_14/epoch=79.ckpt',
                        help='path of ckpt to load')
    return parser.parse_args()


def main():
    opts = get_argparser()
    model = DeepLab.load_from_checkpoint(CKPT_PATH, hparams=opts)
    trainer = Trainer(gpus=1)

    val_transform = Compose([
        torchvision.transforms.CenterCrop(512),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])

    dataset = LabelMeFacade('/mnt/hdd/datasets/facade/labelmefacade', split='test', transform=val_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    predictions = trainer.predict(model, dataloaders=dataloader)
    # decode
    print(len(predictions))


if __name__ == '__main__':
    main()
