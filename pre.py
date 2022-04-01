from pytorchvideo.data.encoded_video import EncodedVideo
from glob import glob
import cv2
import os
from tqdm import tqdm
import multiprocessing
import pandas as pd
from collections import defaultdict
from datetime import datetime
import shutil


def read_and_save(path):
    vidcap = cv2.VideoCapture(path)
    video_name = os.path.basename(path)
    video_dir = video_name.split('_')[0] + '_' + video_name[-5]
    basedir = os.path.dirname(path)
    basedir = basedir.replace('2022', 'frames')
    basedir = os.path.join(basedir, video_dir)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    success,image = vidcap.read()
    count = 0
    while success:
        image = cv2.resize(image, (455, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(basedir, f'{count}.png'), image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1


def main():
    # videos = glob('/home/vitallab/ssd/vitallab/2022/*/*/*.MP4')

    # with multiprocessing.Pool(processes=8) as pool:
    #     for _ in tqdm(pool.imap_unordered(read_and_save, videos), total=len(videos)):
    #         pass

    annotations = glob('/home/vitallab/ssd/vitallab/frames/*/*/*.csv')
    for a in annotations:
        annotation(a)


def annotation(path):
    fps = 30
    df = pd.read_csv(path)
    df[['Filename', 'Camera View', 'Start Time', 'End Time', 'Label/Class ID']]
    extracted_frames = defaultdict(list)

    current = ''
    prev_end = 0
    situation_count = 0
    for idx, row in df.iterrows():
        if not pd.isnull(row['Filename']):
            fn = row['Filename'].split('_')
            current = fn[0] + '_' + fn[-1]
            prev_end = 0
            situation_count = 0

        start_time = datetime.strptime(row['Start Time'], '%H:%M:%S')
        start_time = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
        end_time = datetime.strptime(row['End Time'], '%H:%M:%S')
        end_time = end_time.hour * 3600 + end_time.minute * 60 + end_time.second

        # if prev_end != 0:
        extracted_frames[current].append((prev_end, start_time, 0))
        frame_copy(situation_count, path, prev_end, start_time, current, 0, fps)

        extracted_frames[current].append(
            (start_time, end_time, int(row['Label/Class ID'])))
        frame_copy(situation_count, path, prev_end, start_time, current, int(row['Label/Class ID']), fps)

        prev_end = end_time
        situation_count += 1


def frame_copy(situation_count, path, start_time, end_time, current, label, fps):
    situation_count = str(situation_count)
    user_dir = os.path.dirname(path)
    if not os.path.exists(os.path.join(user_dir, situation_count, current+f'_{label}')):
        os.makedirs(os.path.join(user_dir, situation_count, current+f'_{label}'))
        
    for i in range(start_time *fps, end_time * fps):
        shutil.copy(os.path.join(user_dir, current, f'{i}.png'), os.path.join(user_dir, situation_count, current+f'_{label}', f'{i}.png'))


if __name__ == '__main__':
    main()
