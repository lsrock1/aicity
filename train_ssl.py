import torch

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning
import torch.utils.data
from pytorch_lightning.strategies import DDPStrategy
import pytorchvideo.models as models

from loss import UnsupervisedLoss, PairLoss
from transform import DataUnlabel
import argparse
import logging
from train import VideoClassificationLightningModule


class SSL(pytorch_lightning.LightningModule):
    def __init__(self, pretrained_model=None):
        super().__init__()
        if pretrained_model == None:
            self.model = VideoClassificationLightningModule()
        else:
            self.model = pretrained_model
        self.unsupervised_loss = UnsupervisedLoss()
        self.pair_loss = PairLoss()
        self.unsuper_weight = 1.
        self.pair_weight = 1
        self.step = 0.01

    def forward(self, x):
        return self.model(x)

    def label_guessing(self, video_weak):
        with torch.no_grad():
            probs = F.softmax(self(video_weak), dim=1)
        return probs

    def sharpen(self, x, temperature=0.5):
        sharpened_x = x ** (1 / temperature)
        return sharpened_x / sharpened_x.sum(dim=1, keepdim=True)
    
    def training_step(self, batch, batch_idx):
        # supervised
        self.model.train()
        y_hat = self(batch[0]["video"])
        loss = F.cross_entropy(y_hat, batch[0]['label'])

        # unsupervised
        u_target = self.sharpen(self.label_guessing(batch[1]["video"]))
        u_pred = self(batch[1]["video_strong"])
        u_loss = self.unsupervised_loss(u_pred, u_target)
        # pair_loss = self.pair_loss(u_pred, F.softmax(u_pred, dim=1), u_target)
        # if pair_loss == 0:
        #     pair_loss = pair_loss  * 0
        # print(pair_loss)
        loss = loss + self.unsuper_weight * u_loss
        accu = torch.sum(torch.argmax(y_hat, dim=1) == batch[0]['label']).item() / len(batch[0]['label'])
        # Log the train loss to Tensorboard
        self.log("train_loss", loss.item(), prog_bar=True)
        self.log("train_accu", accu, prog_bar=True)
        # self.step += 0.01
        return {'loss': loss, 'accu': accu, 'log': {'train_loss': loss, 'train_accu': accu}}

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        y_hat = self(batch["video"])
        # print(torch.argmax(y_hat, dim=1) == batch["label"])
        # print(torch.argmax(y_hat, dim=1), batch['label'])
        # print(F.softmax(y_hat, dim=1))
        # print(torch.argmax(y_hat, dim=1))
        y_hat = y_hat.reshape(-1, 18)
        y = batch["label"].reshape(-1)
        accu = torch.sum(torch.argmax(y_hat, dim=1) == y).item() / len(y)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accu", accu,prog_bar=True)
        return {'val_loss': loss, 'val_accu': accu}

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """
        optim = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 1000)#
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[40, 50], gamma=0.1)
        #CosineAnnealingLR(optim, 200)#, gamma=0.3),

        return [optim], [scheduler]


def train():
    parser = argparse.ArgumentParser(description='training')
    # view 0: dashboard, 1: rear, 2: right
    parser.add_argument("--view", type=int, required=True)
    args = parser.parse_args()

    if args.view == 0:
        model = VideoClassificationLightningModule()
        model = model.load_from_checkpoint("lightning_logs/dash_swa/checkpoints/epoch=60-step=732.ckpt", map_location='cpu')

    elif args.view == 1:
        model = VideoClassificationLightningModule()
        model = model.load_from_checkpoint("lightning_logs/rear_swa/checkpoints/epoch=60-step=732.ckpt", map_location='cpu')

    else:
        model = VideoClassificationLightningModule()
        model = model.load_from_checkpoint("lightning_logs/side_swa/checkpoints/epoch=60-step=732.ckpt", map_location='cpu')

    classification_module = SSL(model)
    data_module = DataUnlabel(args.view)
    trainer = pytorch_lightning.Trainer(max_epochs=60,
        accelerator="gpu", devices=2, strategy=DDPStrategy(find_unused_parameters=False), num_sanity_val_steps=0, precision=16,
        enable_checkpointing=True, accumulate_grad_batches=2, sync_batchnorm=True, gradient_clip_val=5, gradient_clip_algorithm="value",)
        # resume_from_checkpoint="lightning_logs/version_36/checkpoints/epoch=17-step=216.ckpt")
    trainer.fit(classification_module, data_module)
    # trainer.validate(classification_module, data_module.val_dataloader(), ckpt_path='lightning_logs/rear/checkpoints/epoch=59-step=600.ckpt')


if __name__ == "__main__":
    

    #Creating and Configuring Logger

    Log_Format = "%(levelname)s %(asctime)s - %(message)s"

    logging.basicConfig(filename = "logfile.log",
                        #stream = sys.stdout, 
                        filemode = "w",
                        format = Log_Format, 
                        level = logging.DEBUG)
    train()
