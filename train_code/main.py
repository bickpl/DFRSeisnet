import os

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint

from data_modul import DataModule
from net import Net


def train():
    torch.set_float32_matmul_precision("high")
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    torch.backends.cudnn.benchmark = True

    data = DataModule(
        train_dir=r"E:\p1\JStar2\script\dataset\L585\train",
        val_dir=r"E:\p1\JStar2\script\dataset\L585\val",
        batch_size=8,
        num_workers=0,
        random_flip=True,
        full_shape=None,
        downsample_size=(256, 256),
        patch_size=(256, 256),
        stride=(256, 256),
        cache_in_memory=False,
    )

    net = Net(lr=1e-4, save_val_debug=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./DRUNet",
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        filename="epoch-{epoch:04d}-val_loss-{val_loss:.6f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        callbacks=[checkpoint_callback, lr_monitor],
        accumulate_grad_batches=2,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        max_epochs=50,
        default_root_dir="./DRUNet",
        check_val_every_n_epoch=1,
        benchmark=True,
    )

    trainer.fit(model=net, datamodule=data)


if __name__ == "__main__":
    train()
