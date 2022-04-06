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
from data import City
from glob import glob
from collections import defaultdict
import os

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 24
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
                size=256
            ),
            CenterCrop((244, 434)),
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
                    ShortSideScale(256),
                    RandomCrop((244, 434)),
                    RandomHorizontalFlip(p=0.5),
                    PackPathway()
                  ]
                ),
              ),
            ]
        )


class DataModule(pytorch_lightning.LightningDataModule):

    # Dataset configuration
    _DATA_PATH = 'dataset/frames24'
    _SKIP_CLASS = 18
    _TARGET_VIEW = [1, ]
    _CLIP_DURATION = (num_frames * sampling_rate)/frames_per_second  # Duration of sampled clip for each video
    _BATCH_SIZE = 4
    _NUM_WORKERS = 8  # Number of parallel processes fetching data
    _VIEW_MAPPING = {
        'Dashboard': 0,
        'Rear': 1,
        'Rightside': 2
    }

    def load_dataset_(self, split='train'):
        # user_id_number/action_number/view
        paths = glob(os.path.join(self._DATA_PATH, "A1/*/*/*"))

        info = []
        for path in paths:
            if not os.path.isdir(path): continue
            # if len(os.listdir(path)) == 0:
            assert len(os.listdir(path)) > 0, "No frames in {}".format(path)
            dirs = path.split('/')
            # print(path)
            user_id, action_number, view = dirs[-3:]
            # print(user_id)
            # print(len(user_id.split('_')))
            if len(user_id.split('_')) != 4: continue
            view, label = view.split('_')
            label = int(label)
            if label == self._SKIP_CLASS: continue
            if self._VIEW_MAPPING[view] not in self._TARGET_VIEW: continue
            # if int(action_number) > 9:
            #     print(action_number)
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
        
        if split != 'train':
        #     info = info[:int(len(info) * 0.8)]
        # else:
            info = info[int(len(info) * 0.8):]
        print(len(info))
        viz = defaultdict(int)
        viz_2 = defaultdict(int)
        for v in info:
            viz[v[1]['label']] += 1
            viz_2[v[1]['view']] += 1
        print(viz)
        print(viz_2)
        # print(info)
        # count = {}
        # for i in range(len(info)):
        #     count[info[i][1]['label']] = count.get(info[i][1]['label'], 0) + 1
        # for v in info:
        #     print(v[1]['label'], v[1]['action_number'], v[1]['view'])
        tf = transform_train if split == 'train' else transform_val
        return City(
            info,
            pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
            transform=tf
        )
        # return LabeledVideoDataset(info, pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION), decode_audio=False,  transform=)

    def train_dataloader(self):
        """
        Create the Kinetics train partition from the list of video labels
        in {self._DATA_PATH}/train
        """
        return torch.utils.data.DataLoader(
            self.load_dataset_(),
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
           
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
        )


if __name__ == '__main__':
    d = DataModule()
    d.load_dataset_()
    print('done')
    d.load_dataset_('val')
