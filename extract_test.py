from train import VideoClassificationLightningModule
from glob import glob
from data import eval_from_path
import os
from transform import transform_val
import torch
from collections import defaultdict
import torch.nn.functional as F
from torch import nn
import numpy as np


def read_video_ids():
    video_id_by_user_id = {}
    with open('dataset/2022/A2/video_ids.csv') as f:
        for idx, line in enumerate(f):
            if idx == 0: continue
            line = line.strip().split(',')
            count = line[1][:-4][-1]
            user_id = '_'.join(line[1][:-4].split('_')[1:4])
            video_id_by_user_id[(user_id.lower(), count)] = line[0]
            # video_id_by_user_id[user_id] = video_id
    return video_id_by_user_id


@torch.no_grad()
def main():
    if not os.path.exists('extracted'):
        os.mkdir('extracted')

    video_id_by_user_id = read_video_ids()
    dash = VideoClassificationLightningModule()
    dash = dash.load_from_checkpoint("lightning_logs/dash_new/checkpoints/epoch=119-step=1440.ckpt", map_location='cpu')
    dash.eval()
    dash.freeze()
    dash.model.blocks[6].proj = nn.Identity()

    rear = VideoClassificationLightningModule()
    rear = rear.load_from_checkpoint("lightning_logs/rear_new/checkpoints/epoch=119-step=1440.ckpt", map_location='cpu')
    rear.eval()
    rear.freeze()
    rear.model.blocks[6].proj = nn.Identity()

    right = VideoClassificationLightningModule()
    right = right.load_from_checkpoint("lightning_logs/side_new/checkpoints/epoch=119-step=1440.ckpt", map_location='cpu')
    right.eval()
    right.freeze()
    right.model.blocks[6].proj = nn.Identity()

    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dirs = glob(os.path.join('dataset/frames24/A2/*'))
    results = defaultdict(list)

    dash = dash.to(device0)
    rear = rear.to(device1)
    right = right.to(device2)
    file = None
    torch.set_grad_enabled(False)
    for dir in dirs:
        # if '42271' in dir: continue
        print(dir)
        end = -1
        idx = 0
        for data in eval_from_path(dir, (32 * 2)/24, transform_val):
            dash.eval(), rear.eval(), right.eval()
            if data['start'] == 0:
                idx = 0
                # if file != None:
                #     file.close()
                video_id = video_id_by_user_id[(data['user_id'], data['count'])]
                # file = open(f'extracted/{video_id}.txt', 'w')
                to_ = dir.replace('frames24', 'multiview')
                if not os.path.exists(os.path.join(to_, f'{video_id}')):
                    os.makedirs(os.path.join(to_, f'{video_id}'))
                results = []
            # print('dash index', data['dash_index'])
            # print('rear index', data['rear_index'])
            dash_re = dash([d.unsqueeze(0).to(device0) for d in data['dash']]).cpu()
            rear_re = rear([d.unsqueeze(0).to(device1) for d in data['rear']]).cpu()
            right_re = right([d.unsqueeze(0).to(device2) for d in data['right']]).cpu()
            dash_re = torch.cat([dash_re, rear_re, right_re], dim=0)
            
            
            np.save(os.path.join(to_, f'{video_id}', f'{idx}.npy'), dash_re.cpu().numpy())
            idx += 1
            # output = F.softmax(dash_re, dim=1).cpu() + F.softmax(rear_re, dim=1).cpu() + F.softmax(right_re,dim=1).cpu()
            # index = torch.argmax(output, dim=1).item()
            # video_id = video_id_by_user_id[(data['user_id'], data['count'])]
            # results[data['user_id'] + '_' + data['count']].append(
            #     (video_id, index, data['start'], data['end']))
            # start = data['start']
            # end = data['end']
            
            # if len(results) == 0:
            #     file.write(f'{video_id} {index} {start} {start+1}\n')
            #     results.append(output)
            #     results.append(output)
            # else:
            #     index = torch.argmax((results[-1] + output) / 2, dim=1).item()
            #     file.write(f'{video_id} {index} {start} {start+1}\n')
            #     results[-1] = (results[-1] + output) / 2
            #     results.append(output)
            # results[data['user_id'] + '_' + data['count']].append(
            #     (torch.argmax(dash_re, dim=1).item(), torch.argmax(rear_re, dim=1).item(), torch.argmax(right_re, dim=1).item(), data['start'], data['end']))
            # print(f'{video_id} {index} {start} {start+1}')


if __name__ == '__main__':
    main()
