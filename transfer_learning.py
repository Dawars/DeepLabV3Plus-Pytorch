"""
Loading pre-trained model on cityscapes dataset, freezeing backbone and replacing classifier
"""
import torch
from torch import nn

import network
import utils
from network import deeplabv3plus_resnet101, DeepLabHeadV3Plus

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 8  # for new task
output_stride = 16
ckpt = "pretrained/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar"

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

print(model)

model = nn.DataParallel(model)
model.to(device)

print("end")
