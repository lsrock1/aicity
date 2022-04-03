import csv
import functools
import itertools
import os
from collections import defaultdict
from typing import Any, Callable, List, Optional, Tuple, Type

import torch
import torch.utils.data
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.frame_video import FrameVideo

from pytorchvideo.data.utils import MultiProcessSampler


def iter_from_path(path, sampler, transform):
    # ~/situation_number, view/imgs
    dash = os.path.join(path, 'DashBoard_*')[0]
    label = int(dash.split('_')[-1])
    dash = FrameVideo.from_directory(dash)
    rear = FrameVideo.from_directory(os.path.join(path, 'Rear_*')[0])
    right = FrameVideo.from_directory(os.path.join(path, 'Rightside_*')[0])
    duration = min(dash.duration, rear.duration, right.duration)
    next_clip_start_time = 0    
    while True:
        clip_start, clip_end, clip_index, aug_index, is_last_clip = sampler(
            next_clip_start_time, duration, {}
        )
        next_clip_start_time = clip_end
        yield {
            'dash': dash.get_clip(clip_start, clip_end),
            'rear': rear.get_clip(clip_start, clip_end),
            'right': right.get_clip(clip_start, clip_end),
            'start': clip_start,
            'end': clip_end,
            'label': label
        }
        if is_last_clip:
            break


class City(torch.utils.data.IterableDataset):
    """
    Action recognition video dataset for
    `Charades <https://prior.allenai.org/projects/charades>`_ stored as image frames.

    This dataset handles the parsing of frames, loading and clip sampling for the
    videos. All io is done through :code:`iopath.common.file_io.PathManager`, enabling
    non-local storage uri's to be used.
    """

    # Number of classes represented by this dataset's annotated labels.
    NUM_CLASSES = 18

    def __init__(
        self,
        info: List,
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        video_path_prefix: str = "",
        frames_per_clip: Optional[int] = None,
    ) -> None:
        """
        Args:
            data_path (str): Path to the data file. This file must be a space
                separated csv with the format: (original_vido_id video_id frame_id
                path_labels)

            clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

            video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

            transform (Optional[Callable]): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations on the clips. The clip output format is described in __next__().

            video_path_prefix (str): prefix path to add to all paths from data_path.

            frames_per_clip (Optional[int]): The number of frames per clip to sample.
        """

        # torch._C._log_api_usage_once("PYTORCHVIDEO.dataset.Charades.__init__")

        self._transform = transform
        self._clip_sampler = clip_sampler
        (
            self._path_to_videos,
            self._labels
        ) = zip(*[(i[0], i[1]['label']) for i in info])
        self._video_sampler = video_sampler(self._path_to_videos)
        self._video_sampler_iter = None  # Initialized on first call to self.__next__()
        self._frame_filter = (
            functools.partial(
                City._sample_clip_frames,
                frames_per_clip=frames_per_clip,
            )
            if frames_per_clip is not None
            else None
        )

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video = None
        self._loaded_clip = None
        self._next_clip_start_time = 0.0


    @staticmethod
    def _sample_clip_frames(
        frame_indices: List[int], frames_per_clip: int
    ) -> List[int]:
        """
        Args:
            frame_indices (list): list of frame indices.
            frames_per+clip (int): The number of frames per clip to sample.

        Returns:
            (list): Outputs a subsampled list with num_samples frames.
        """
        num_frames = len(frame_indices)
        indices = torch.linspace(0, num_frames - 1, frames_per_clip)
        indices = torch.clamp(indices, 0, num_frames - 1).long()

        return [frame_indices[idx] for idx in indices]

    @property
    def video_sampler(self) -> torch.utils.data.Sampler:
        return self._video_sampler

    def __next__(self) -> dict:
        """
        Retrieves the next clip based on the clip sampling strategy and video sampler.

        Returns:
            A dictionary with the following format.

            .. code-block:: text

                {
                    'video': <video_tensor>,
                    'label': <index_label>,
                    'video_label': <index_label>
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                }
        """
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        if self._loaded_video:
            video, video_index = self._loaded_video
        else:
            video_index = next(self._video_sampler_iter)
            path_to_video_frames = self._path_to_videos[video_index]
            video = FrameVideo.from_directory(path_to_video_frames, multithreaded_io=False, fps=24)
            self._loaded_video = (video, video_index)

        clip_start, clip_end, clip_index, aug_index, is_last_clip = self._clip_sampler(
            self._next_clip_start_time, video.duration, {}
        )
        # print(clip_start)
        # print(clip_end)
        # Only load the clip once and reuse previously stored clip if there are multiple
        # views for augmentations to perform on the same clip.
        if aug_index == 0:
            self._loaded_clip = video.get_clip(clip_start, clip_end, self._frame_filter)
        # print(self._loaded_clip['video'].shape)
        frames, frame_indices = (
            self._loaded_clip["video"],
            self._loaded_clip["frame_indices"],
        )
        self._next_clip_start_time = clip_end

        if is_last_clip:
            self._loaded_video = None
            self._next_clip_start_time = 0.0

        # Merge unique labels from each frame into clip label.
        # labels_by_frame = [
        #     self._labels[video_index][i]
        #     for i in range(min(frame_indices), max(frame_indices) + 1)
        # ]
        sample_dict = {
            "video": frames,
            "label": self._labels[video_index],
            # "video_label": self._video_labels[video_index],
            "video_name": str(video_index),
            "video_index": video_index,
            "clip_index": clip_index,
            "aug_index": aug_index,
        }
        if self._transform is not None:
            sample_dict = self._transform(sample_dict)

        return sample_dict

    def __iter__(self):
        return self

