import argparse
import os

import optuna

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger

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
                        help='batch size (default: 32)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 1)')
    parser.add_argument("--lr", type=float, default=1e-3,
                        help='learning_rate')
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Set random seed everywhere')
    parser.add_argument('--save_path', type=str, default='/mnt/hdd/datasets/facade/experiments/deeplab/',
                        help='paths to save checkpoints and logs to')
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--backbone', type=str, default='resnet101',
                        choices=['resnet101', 'mobilenet'], help='backbone name')
    parser.add_argument('--pruning', default=False, action='store_true',
                        help='prune trials during hyperparam search')

    return parser.parse_args()


def main(opts):
    pl.seed_everything(opts.random_seed)

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

    val_transform = et.ExtCompose([
        et.ExtCenterCrop(512),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    dataset_train = LabelMeFacade('/mnt/hdd/datasets/facade/labelmefacade', 'train', transform=train_transform)
    dataset_val = LabelMeFacade('/mnt/hdd/datasets/facade/labelmefacade', 'val', transform=val_transform)

    is_debug = False

    def objective(trial: optuna.trial.Trial) -> float:
        opts.lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
        opts.batch_size = trial.suggest_int('batch_size', 1, 32)

        # model
        model = DeepLab(opts, dataset_train, dataset_val)
        # training

        checkpoint_callback = \
            ModelCheckpoint(dirpath=os.path.join(opts.save_path, 'ckpts', opts.exp_name, f'trial_{trial.number}'),
                            filename='{epoch:d}',
                            monitor='train/ce',
                            mode='max',
                            every_n_epochs=5,
                            save_top_k=5)

        logger = TestTubeLogger(save_dir=os.path.join(opts.save_path, 'logs'),
                                name=opts.exp_name,
                                # debug=opts.debug,
                                version=trial.number,
                                create_git_tag=True,
                                log_graph=False)

        # early_stopping = pl.callbacks.EarlyStopping(monitor="val/ce")
        early_stopping = PyTorchLightningPruningCallback(trial, monitor="val/ce")
        trainer = pl.Trainer(gpus=None if is_debug else 1,
                             max_epochs=80,
                             checkpoint_callback=True,
                             callbacks=[checkpoint_callback, early_stopping],
                             # resume_from_checkpoint=hparams.ckpt_path,
                             logger=logger,
                             weights_summary=None,
                             progress_bar_refresh_rate=1,
                             # accelerator='ddp' if hparams.num_gpus > 1 else None,
                             num_sanity_val_steps=1,
                             benchmark=True,
                             profiler="simple",  # if hparams.num_gpus == 1 else None,
                             )
        trainer.fit(model)

        return trainer.callback_metrics["val/ce"].item()

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if opts.pruning else optuna.pruners.NopPruner()
    )

    search_space = {'lr': [1e-6, 1e-5, 1.7e-5, 1e-4, 1e-3], 'batch_size': [1, 2, 4, 8, 16, 32] }
    study = optuna.create_study(direction="minimize", pruner=pruner, sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(objective, n_trials=40)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == '__main__':
    hparams = get_argparser()
    main(hparams)
