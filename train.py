import torch

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning
import torch.utils.data
from pytorch_lightning.strategies import DDPStrategy
import pytorchvideo.models as models

from transform import DataModule


class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        # slowfast = models.create_slowfast()
        # model_name = "slowfast_r50"
        self.model = torch.hub.load("facebookresearch/pytorchvideo", model="slowfast_r50", pretrained=True)
        # self.model.blocks[6].dropout.p = 0.1
        self.model.blocks[6].proj = nn.Linear(2304, 18)
        # self.model = make_kinetics_resnet()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # print(len(batch['video']))
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        y_hat = self.model(batch["video"])

        # Compute cross entropy loss, loss.backwards will be called behind the scenes
        # by PyTorchLightning after being returned from this method.
        # print(F.softmax(y_hat[0], dim=0))
        # print(batch["label"][0])
        loss = F.cross_entropy(y_hat, batch["label"])
        accu = torch.sum(torch.argmax(y_hat, dim=1) == batch["label"]).item() / len(batch["label"])
        # Log the train loss to Tensorboard
        self.log("train_loss", loss.item(), prog_bar=True)
        self.log("train_accu", accu, prog_bar=True)

        return {'loss': loss, 'accu': accu, 'log': {'train_loss': loss, 'train_accu': accu}}

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        accu = torch.sum(torch.argmax(y_hat, dim=1) == batch["label"]).item() / len(batch["label"])
        loss = F.cross_entropy(y_hat, batch["label"])
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
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[300, 400, 450], gamma=0.1)
        #CosineAnnealingLR(optim, 200)#, gamma=0.3),

        return [optim], [scheduler]


def train():
    classification_module = VideoClassificationLightningModule()
    data_module = DataModule()
    trainer = pytorch_lightning.Trainer(max_epochs=500,
        accelerator="gpu", devices=4, strategy=DDPStrategy(find_unused_parameters=False), num_sanity_val_steps=0, precision=16,
        enable_checkpointing=True, accumulate_grad_batches=4, sync_batchnorm=True, gradient_clip_val=0.5, gradient_clip_algorithm="value")
    trainer.fit(classification_module, data_module)

if __name__ == "__main__":
    import logging
    import sys

    #Creating and Configuring Logger

    Log_Format = "%(levelname)s %(asctime)s - %(message)s"

    logging.basicConfig(filename = "logfile.log",
                        #stream = sys.stdout, 
                        filemode = "w",
                        format = Log_Format, 
                        level = logging.DEBUG)
    train()
