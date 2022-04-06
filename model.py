"""
Loading pre-trained model on cityscapes dataset, freezeing backbone and replacing classifier
"""
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

import utils
from network import deeplabv3plus_resnet101, DeepLabHeadV3Plus, deeplabv3plus_mobilenet

num_classes = 9  # for new task
output_stride = 16


class DeepLab(pl.LightningModule):
    def __init__(self, hparams, train_dataset=None, val_dataset=None):
        super().__init__()
        self.opts = hparams
        self.save_hyperparameters(hparams)

        self.log_val_idx = list(range(4))

        backbone = hparams.backbone

        if backbone == "resnet101":
            ckpt = "pretrained/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar"
            model = deeplabv3plus_resnet101(num_classes=19, output_stride=16)
        elif backbone == "mobilenet":
            ckpt = "pretrained/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
            model = deeplabv3plus_mobilenet(num_classes=19, output_stride=8)
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

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.opts.lr)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.opts.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.opts.val_batch_size, drop_last=True)

    def training_step(self, train_batch, batch_idx):
        x, mask = train_batch

        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            mask = mask.unsqueeze(0)

        x_hat = self.model(x)
        loss_val = F.cross_entropy(x_hat, mask, ignore_index=0)
        self.log('train/ce', loss_val)
        return {'loss': loss_val}

    def predict_step(self, batch, batch_id, **kwargs):
        img, filename = batch
        x_hat = self.model(img)
        mask_pred = torch.argmax(x_hat, dim=1)
        return mask_pred, filename

    def validation_step(self, val_batch, batch_idx):

        with torch.no_grad():
            x, mask = val_batch
            assert(x.size(0) == 1)
            x_hat = self.model(x)
            loss_val = F.cross_entropy(x_hat, mask, ignore_index=0)

            if batch_idx in self.log_val_idx:
                mask_pred = torch.argmax(x_hat, dim=1)

                mask_pred = self.val_dataset.decode_target(mask_pred[0].cpu().detach().numpy()).transpose(2, 0, 1)
                mask_label = self.val_dataset.decode_target(mask[0].cpu().detach().numpy()).transpose(2, 0, 1)

                # transform image back for display
                img = utils.denormalize(x[0], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\
                    .detach().cpu().numpy()

                image = np.stack([img, mask_pred/256, mask_label/256])  # (3, 3, H, W)

                self.logger.experiment.add_images(f'val/inference_{batch_idx}', image, self.global_step)

            return {'val_loss': loss_val}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val/ce', val_loss)
        return {'val_loss': val_loss}
