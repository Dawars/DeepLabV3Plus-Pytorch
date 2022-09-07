# Bootstrapping Dataset Labeling for Architectural Facade Segmentation using Deep Learning

This project is built on code from https://github.com/VainF/DeepLabV3Plus-Pytorch

### Report
The final report for this IDP can be found at: [Facade_IDP_Report_David_Komorowicz.pdf](Facade_IDP_Report_David_Komorowicz.pdf)


## Transfer Learning
This part runs transfer learning from Cityscapes to LabelMeFacade dataset along with hyperparameter tuning, refer to the report for details

- Download LabelMeFacade dataset: https://github.com/cvjena/labelmefacade
- Download split files `training_splits_labelmefacade.zip` and put to labelmefacade directory from https://github.com/Dawars/DeepLabV3Plus-Pytorch/releases/tag/splits
- Download checkpoint for DeepLab (https://drive.google.com/file/d/1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb/view?usp=sharing), put it into `./pretrained/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar` without extracting it

```bash
python transfer_learning.py --data_path /path/to/labelmefacade/ --save_path ./saves/ --exp_name EXP_NAME
```

## Bootstrapping
Bootstrapping is fine-tuning the model from the previous step using data from the LabelMeFacade and ZuDuB datasets in different configurations.

- Download the checkpoints [here](https://github.com/Dawars/DeepLabV3Plus-Pytorch/releases/tag/bootstrapping) or run the previous step and use the best models.
- Download split files `bootstrapping_split_zubud.zip` from https://github.com/Dawars/DeepLabV3Plus-Pytorch/releases/tag/splits
- Modify the code so that the dataset paths point to the correct places.

```bash
python bootstrapping.py
```


## Evaluation
The evaluation script runs inference on unseen data and calculates performance based on labels.

- Comment out selected dataset to run evaluation on (eTrims or LabelMeFacade)
- Modify the code so that the dataset paths point to the correct places.
- Read results in tensorboard logs.

```bash
python evaluate.py
```


