"""
Loading pre-trained model on cityscapes dataset, freezeing backbone and replacing classifier
"""
import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F

import utils
from network import deeplabv3plus_resnet101, DeepLabHeadV3Plus

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 9  # for new task
output_stride = 16
ckpt = "pretrained/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar"


class DeepLab(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)  # todo add deeplab variant, stride

        model = deeplabv3plus_resnet101(num_classes=19, output_stride=output_stride)
        utils.set_bn_momentum(model.backbone, momentum=0.01)

        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        print("Resume model from %s" % ckpt)
        del checkpoint

        # replace classifier
        inplanes = 2048
        low_level_planes = 256
        if output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
            aspp_dilate = [12, 24, 36]
        else:
            replace_stride_with_dilation = [False, False, True]
            aspp_dilate = [6, 12, 18]
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        model.classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)

        # freeze backbone
        for param in model.backbone.parameters():
            param.requires_grad = False

        self.model = model

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, mask = train_batch
        x_hat = self.model(x)
        loss_val = F.cross_entropy(x_hat, mask, ignore_index=0)
        self.log('train/ce', loss_val)
        return {'loss': loss_val}

    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            x, mask = val_batch
            x_hat = self.model(x)
            loss_val = F.cross_entropy(x_hat, mask, ignore_index=0)
            return {'val_loss': loss_val}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val/ce', val_loss)
        return {'val_loss': val_loss}
