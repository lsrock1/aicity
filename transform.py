import torch
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    CenterCrop,
    RandomHorizontalFlip
)
from pytorchvideo.data import LabeledVideoDataset
import pytorch_lightning
import pytorchvideo

from glob import glob
import os

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
alpha = 4


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

transform_val =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            Normalize(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCrop(244),
            PackPathway()
        ]
    ),
)

transform_train = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(244),
                    RandomHorizontalFlip(p=0.5),
                    PackPathway()
                  ]
                ),
              ),
            ]
        )


class DataModule(pytorch_lightning.LightningDataModule):

    # Dataset configuration
    _DATA_PATH = '/home/vitallab/ssd/vitallab/frames'
    _SKIP_CLASS = 18
    _CLIP_DURATION = (num_frames * sampling_rate)/frames_per_second  # Duration of sampled clip for each video
    _BATCH_SIZE = 8
    _NUM_WORKERS = 8  # Number of parallel processes fetching data
    _VIEW_MAPPING = {
        'Dashboard': 0,
        'Rear': 1,
        'Rightside': 2
    }

    def load_dataset_(self, split='train'):
        # user_id_number/action_number/view
        paths = glob(os.path.join(self._DATA_PATH, "*/*/*"))
        print(paths)
        info = []
        for path in paths:
            dirs = path.split('/')
            user_id, action_number, view = dirs[-3:]
            if len(user_id.split('_')) != 4: continue
            view, label = view.split('_')
            label = int(label)
            if label == self._SKIP_CLASS: continue
            info.append(
                (
                    path, 
                    {
                        'label': label, 'action_number': int(action_number),
                        'view': self._VIEW_MAPPING[view]
                    }
                )
            )
        info = sorted(info, key= lambda x: x[0])
        if split == 'train':
            info = info[:int(len(info) * 0.8)]
        else:
            info = info[int(len(info) * 0.8):]
        return LabeledVideoDataset(info, pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION), decode_audio=False)

    def train_dataloader(self):
        """
        Create the Kinetics train partition from the list of video labels
        in {self._DATA_PATH}/train
        """
        return torch.utils.data.DataLoader(
            self.load_dataset_(),
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            transform=transform_train
        )

    def val_dataloader(self):
        """
        Create the Kinetics validation partition from the list of video labels
        in {self._DATA_PATH}/val
        """
        val_dataset = self.load_dataset_('val')
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            transform=transform_val
        )


if __name__ == '__main__':
    d = DataModule()
    d.load_dataset_()