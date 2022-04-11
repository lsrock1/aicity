import csv
import functools
import itertools
import os
import random
from collections import defaultdict
from typing import Any, Callable, List, Optional, Tuple, Type
from glob import glob
import numpy as np
import math
import cv2

import torch
import torch.utils.data
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.frame_video import FrameVideo

from pytorchvideo.data.utils import MultiProcessSampler

from pytorchvideo.transforms.functional import (
    random_crop_with_boxes,
    uniform_crop_with_boxes,
    crop_boxes, clip_boxes_to_image
)


def _uniform_crop_helper(images: torch.Tensor, size: int, spatial_idx: int):
    """
    A helper function grouping the common components in uniform crop
    """
    assert spatial_idx in [0, 1, 2]
    height = images.shape[2]
    width = images.shape[3]

    y_offset = int(math.ceil((height - size[0]) / 2))
    x_offset = int(math.ceil((width - size[1]) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size[0]
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size[1]
    cropped = images[:, :, y_offset : y_offset + size[0], x_offset : x_offset + size[1]]

    return cropped, x_offset, y_offset


def uniform_crop_with_boxes(
    images: torch.Tensor,
    size: int,
    spatial_idx: int,
    boxes: torch.Tensor,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Perform uniform spatial sampling on the images and corresponding boxes.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `channel` x `num frames` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
        boxes (tensor): Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        cropped (tensor): images with dimension of
            `channel` x `num frames` x `height` x `width`.
        cropped_boxes (tensor): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    cropped, x_offset, y_offset = _uniform_crop_helper(images, size, spatial_idx)
    cropped_boxes = crop_boxes(boxes, x_offset, y_offset)
    return cropped, clip_boxes_to_image(
        cropped_boxes, cropped.shape[-2], cropped.shape[-1]
    )


def random_crop_with_boxes(
    images: torch.Tensor, size: int, boxes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform random spatial crop on the given images and corresponding boxes.
    Args:
        images (tensor): images to perform random crop. The dimension is
            `channel` x `num frames` x `height` x `width`.
        size (int): the size of height and width to crop on the image.
        boxes (tensor): Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        cropped (tensor): cropped images with dimension of
            `channel` x `num frames` x `height` x `width`.
        cropped_boxes (tensor): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    if images.shape[2] == size and images.shape[3] == size:
        return images
    height = images.shape[2]
    width = images.shape[3]
    y_offset = 0
    if height > size[0]:
        y_offset = int(np.random.randint(0, height - size[0]))
    x_offset = 0
    if width > size[1]:
        x_offset = int(np.random.randint(0, width - size[1]))
    cropped = images[:, :, y_offset : y_offset + size[0], x_offset : x_offset + size[1]]

    cropped_boxes = crop_boxes(boxes, x_offset, y_offset)
    return cropped, clip_boxes_to_image(
        cropped_boxes, cropped.shape[-2], cropped.shape[-1]
    )


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
                0, frames.shape[1] - 1, frames.shape[1] // 4
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def eval_from_path(path, clip_duration, transform):
    # ~/user_id
    user_id = os.path.basename(path)
    dirs = glob(os.path.join(path, '*'))

    numbers = set([os.path.basename(d)[-1] for d in dirs])
    numbers = sorted(list(numbers))
    dirs_by_number = [[], []]
    print(numbers)
    if len(numbers) == 1:
        dirs_by_number.pop(-1)
    for d in dirs:
        if os.path.basename(d)[-1] == numbers[0]:
            dirs_by_number[0].append(d)
        else:
            dirs_by_number[1].append(d)

    # dirs_by_number = [dirs_by_number[1]]

    for idx, dirs in enumerate(dirs_by_number):
        count = numbers[idx]
        for d in dirs:
            print(d)
            if 'Dash' in d:
                dash = sorted(glob(os.path.join(d, '*.png')), key= lambda x: int(x.split('/')[-1][:-4]))#FrameVideo.from_directory(d, fps=24)
            elif 'Rear' in d:
                rear = sorted(glob(os.path.join(d, '*.png')), key= lambda x: int(x.split('/')[-1][:-4]))
            elif 'Right' in d:
                right = sorted(glob(os.path.join(d, '*.png')), key= lambda x: int(x.split('/')[-1][:-4]))
                
        duration = min(len(dash), len(rear), len(right))
        # max_possible_clip_start = max(duration - clip_duration, 0)
        # max_possible_clip_start = int(max_possible_clip_start)

        for i in range(duration - 64):
            dash_data = np.stack(
                [cv2.cvtColor(cv2.imread(d), cv2.COLOR_BGR2RGB).transpose(2, 0, 1) for d in dash[i:i+64]], axis=1)#dash.get_clip(i, i+clip_duration)
            rear_data = np.stack(
                [cv2.cvtColor(cv2.imread(d), cv2.COLOR_BGR2RGB).transpose(2, 0, 1) for d in rear[i:i+64]], axis=1)
            right_data = np.stack(
                [cv2.cvtColor(cv2.imread(d), cv2.COLOR_BGR2RGB).transpose(2, 0, 1) for d in right[i:i+64]], axis=1)
            dash_data = torch.from_numpy(dash_data).float()
            rear_data = torch.from_numpy(rear_data).float()
            right_data = torch.from_numpy(right_data).float()
            dash_box = retrieve_box(dash[i:i+64])
            rear_box = retrieve_box(rear[i:i+64])
            right_box = retrieve_box(right[i:i+64])
            
            yield {
                'dash': transform._transform(dash_data),
                'rear': transform._transform(rear_data),
                'right': transform._transform(right_data),
                'start': i,
                'end': i + 64,
                'user_id': user_id.lower(),
                'count': count,
                'dash_detection': dash_box,
                'rear_detection': rear_box,
                'right_detection': right_box,
            }


def retrieve_box(video):
    index = len(video)//2
    while True:
        if not os.path.exists(video[index].replace('frames24', 'bboxes').replace('png', 'npy')):
            index -= 1
            if index < 0:
                return None
                assert False, video._video_frame_to_path(frame_indices[0])
        else:
            box_path = video[index].replace('frames24', 'bboxes').replace('png', 'npy')
            box = np.load(box_path)
            if len(box) == 0:
                index -= 1
                if index < 0:
                    return None
                    assert False, video._video_frame_to_path(frame_indices[0])
            else:
                break
    if len(box) > 1:
        # number, 4
        sizes = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        max_index = np.argmax(sizes)
        box = box[max_index][None, :]
    return torch.from_numpy(box).float()


def iter_from_path(path, clip_duration, transform):
    # ~/situation_number, view/imgs
    dash = os.path.join(path, 'DashBoard_*')[0]
    label = int(dash.split('_')[-1])
    dash = FrameVideo.from_directory(dash)
    rear = FrameVideo.from_directory(os.path.join(path, 'Rear_*')[0])
    right = FrameVideo.from_directory(os.path.join(path, 'Rightside_*')[0])
    duration = min(dash.duration, rear.duration, right.duration)
    next_clip_start_time = 0
    max_possible_clip_start = max(duration - clip_duration, 0)
    clip_start_sec = random.uniform(0, max_possible_clip_start)
    for i in range(clip_start_sec):
        yield {
            'dash': transform._transform(dash.get_clip(i, i+clip_duration)["video"]),
            'rear': transform._transform(rear.get_clip(i, i+clip_duration)["video"]),
            'right': transform._transform(right.get_clip(i, i+clip_duration)["video"]),
            'start': i,
            'end': i + clip_duration,
            'label': label
        }
    # while True:
    #     clip_start, clip_end, clip_index, aug_index, is_last_clip = sampler(
    #         next_clip_start_time, duration, {}
    #     )
    #     next_clip_start_time = clip_start
    #     yield {
    #         'dash': dash.get_clip(clip_start, clip_end),
    #         'rear': rear.get_clip(clip_start, clip_end),
    #         'right': right.get_clip(clip_start, clip_end),
    #         'start': clip_start,
    #         'end': clip_end,
    #         'label': label
    #     }
    #     if is_last_clip:
    #         break


class City(torch.utils.data.IterableDataset):
    def __init__(
        self,
        info: List,
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        video_path_prefix: str = "",
        frames_per_clip: Optional[int] = None,
    ) -> None:
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
        # print(clip_index)
        # Only load the clip once and reuse previously stored clip if there are multiple
        # views for augmentations to perform on the same clip.
        # print(clip_start, clip_end)
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


class CityDetection(City):
    def __init__(
        self,
        info: List,
        phase: str,
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        video_path_prefix: str = "",
        frames_per_clip: Optional[int] = None,
    ) -> None:
        super().__init__(
            info,
            clip_sampler,
            video_sampler,
            transform,
            video_path_prefix,
            frames_per_clip,
        )
        self.phase = phase

    def __next__(self) -> dict:
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
        # print(clip_start, clip_end)
        if aug_index == 0:
            self._loaded_clip = video.get_clip(clip_start, clip_end, self._frame_filter)
        # print(self._loaded_clip['video'].shape)
        frames, frame_indices = (
            self._loaded_clip["video"],
            self._loaded_clip["frame_indices"],
        )
        print(len(frame_indices))
        self._next_clip_start_time = clip_end

        if is_last_clip:
            self._loaded_video = None
            self._next_clip_start_time = 0.0
        index = len(frame_indices)//2
        while True:
            if not os.path.exists(video._video_frame_to_path(frame_indices[index]).replace('frames24', 'bboxes').replace('png', 'npy')):
                index -= 1
                if index < 0:
                    assert False, video._video_frame_to_path(frame_indices[0])
            else:
                box_path = video._video_frame_to_path(frame_indices[index]).replace('frames24', 'bboxes').replace('png', 'npy')
                box = np.load(box_path)
                if len(box) == 0:
                    index -= 1
                    if index < 0:
                        assert False, video._video_frame_to_path(frame_indices[0])
                else:
                    break
        
        if len(box) > 1:
            # number, 4
            sizes = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
            max_index = np.argmax(sizes)
            box = box[max_index][None, :]
        # Merge unique labels from each frame into clip label.
        # labels_by_frame = [
        #     self._labels[video_index][i]
        #     for i in range(min(frame_indices), max(frame_indices) + 1)
        # ]
        sample_dict = {
            "video": frames,
            "label": self._labels[video_index],
            "box": box,
            "video_name": str(video_index),
            "video_index": video_index,
            "clip_index": clip_index,
            "aug_index": aug_index,
        }
        # if self._transform is not None:
        sample_dict = self._transform(sample_dict)
        if self.phase == 'train':
            sample_dict['label'] = self._labels[video_index]

        if self.phase == 'train':
            video, box = random_crop_with_boxes(sample_dict['video'], (244, 434), sample_dict['box'])

            video = PackPathway()(video)
        else:
            video, box = uniform_crop_with_boxes(sample_dict['video'], (244, 434), 1, sample_dict['box'])
            video = PackPathway()(video)
        # box = torch.cat([torch.zeros(box.shape[0],1), box], dim=1)
        # [print(v.shape) for v in video]
        # print(box.shape)
        sample_dict['video'] = video
        sample_dict['box'] = box[0]

        return sample_dict
