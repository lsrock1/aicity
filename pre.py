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


def make_dir(path):
    video_name = os.path.basename(path)
    view_name = video_name.split('_')[0]
    if view_name == 'Right':
        view_name = 'Rightside'    
    if view_name == 'Rearview':
        view_name = 'Rear'
    video_dir = video_name.split('_')[0] + '_' + video_name[-5]
    basedir = os.path.dirname(path)
    basedir = basedir.replace('2022', 'frames24')
    basedir = os.path.join(basedir, video_dir)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    return basedir


def main():
    # videos = glob('/home/vitallab/ssd/vitallab/2022/*/*/*.MP4')
    # for video in tqdm(videos):

    #     save_dir = make_dir(video)
    #     p = os.path.join(save_dir, '%d.png')
    #     cmd = f'ffmpeg -i {video} -vf "scale=455:256,fps=24" {p}'
    #     os.system(cmd)

    # csvs = glob('/home/vitallab/ssd/vitallab/2022/*/*/*.csv')
    # for c in csvs:
    #     shutil.copy(c, c.replace('2022', 'frames24'))
    # with multiprocessing.Pool(processes=8) as pool:
    #     for _ in tqdm(pool.imap_unordered(read_and_save, videos), total=len(videos)):
    #         pass

    annotations = glob('/home/vitallab/ssd/vitallab/frames24/*/*/*.csv')
    
    with multiprocessing.Pool(processes=8) as pool:
        for _ in tqdm(pool.imap_unordered(annotation, annotations), total=len(annotations)):
            pass


def annotation(path):
    fps = 24
    df = pd.read_csv(path)
    if 'Filename' in df.columns:
        file_name_c = 'Filename'
    elif 'File name' in df.columns:
        file_name_c = 'File name'
    else:
        file_name_c = 'File Name'
    df[[file_name_c, 'Camera View', 'Start Time', 'End Time', 'Label/Class ID']]
    df["Label/Class ID"] = df["Label/Class ID"].astype(str)
    extracted_frames = defaultdict(list)

    current = ''
    postfix = ''
    prev_end = 0
    situation_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if not pd.isnull(row[file_name_c]) and len(row[file_name_c].strip()) > 0:
            fn = [r.strip() for r in row[file_name_c].split('_')]
            if fn[0] == 'Right': fn[0] = 'Rightside'
            if fn[0] == 'Rearview': fn[0] = 'Rear'
            print(fn[0])
            current = fn[0]# + '_' + fn[-1]
            postfix = fn[-1]
            prev_end = 0
            situation_count = 0
        
        # if 'Rear' not in current: continue
        try:
            start_time = datetime.strptime(row['Start Time'], '%H:%M:%S')
            start_time = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
            end_time = datetime.strptime(row['End Time'], '%H:%M:%S')
            end_time = end_time.hour * 3600 + end_time.minute * 60 + end_time.second
        except:
            start_time = datetime.strptime(row['Start Time'], '%M:%S')
            start_time = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
            end_time = datetime.strptime(row['End Time'], '%M:%S')
            end_time = end_time.hour * 3600 + end_time.minute * 60 + end_time.second
        print(start_time, end_time)
        if pd.isnull(row['Label/Class ID']) or row['Label/Class ID'].strip() == 'nan' or row['Label/Class ID'].strip() == 'NA':
            # extracted_frames[current].append((start_time, end_time, situation_count))
            prev_end = end_time
            continue

        # extracted_frames[current].append((prev_end, start_time, 18))
        # frame_copy(situation_count, path, prev_end, start_time, current, 18, fps, postfix)
        # situation_count += 1

        extracted_frames[current].append(
            (start_time, end_time, int(float(row['Label/Class ID']))))
        frame_copy(situation_count, path, start_time, end_time, current, int(float(row['Label/Class ID'])), fps, postfix)

        prev_end = end_time
        situation_count += 1


def frame_copy(situation_count, path, start_time, end_time, current, label, fps, postfix):
    situation_count = str(situation_count)
    user_dir = os.path.dirname(path)
    to_user_dir = user_dir + f'_{postfix}'
    if not os.path.exists(os.path.join(to_user_dir, situation_count, current+f'_{label}')):
        os.makedirs(os.path.join(to_user_dir, situation_count, current+f'_{label}'))
        
    for i in range(start_time *fps, end_time * fps):
        if not os.path.exists(os.path.join(user_dir, current + f'_{postfix}', f'{i}.png')): 
            print(os.path.join(user_dir, current + f'_{postfix}', f'{i}.png'))
            return
        shutil.copy(os.path.join(user_dir, current + f'_{postfix}', f'{i}.png'), os.path.join(to_user_dir, situation_count, current+f'_{label}', f'{i}.png'))


if __name__ == '__main__':
    main()
