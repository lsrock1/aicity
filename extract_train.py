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


@torch.no_grad()
def main():
    dash = VideoClassificationLightningModule()
    # print(dash.model.blocks[6].proj.weight[0])
    dash = dash.load_from_checkpoint("lightning_logs/dash_new/checkpoints/epoch=119-step=1440.ckpt", map_location='cpu')
    dash.eval()
    dash.freeze()
    # dash.model.blocks[6].proj = nn.Identity()
    # print(dash.model)
    # print(dash.model.blocks[6].proj.weight[0])

    rear = VideoClassificationLightningModule()
    rear = rear.load_from_checkpoint("lightning_logs/rear_new/checkpoints/epoch=119-step=1440.ckpt", map_location='cpu')
    rear.eval()
    rear.freeze()
    # rear.model.blocks[6].proj = nn.Identity()
    right = VideoClassificationLightningModule()
    right = right.load_from_checkpoint("lightning_logs/side_new/checkpoints/epoch=119-step=1440.ckpt", map_location='cpu')
    right.eval()
    right.freeze()
    # right.model.blocks[6].proj = nn.Identity()

    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dirs = glob(os.path.join('dataset/frames24/A1/*/*'))
    dirs = [d for d in dirs if len(d.split('/')[-2].split('_')) == 4]
    results = defaultdict(list)

    dash = dash.to(device0)
    rear = rear.to(device1)
    right = right.to(device2)
    
    if not os.path.exists('dataset/multiview'):
        os.makedirs('dataset/multiview')
    for dir in dirs:
        # if '42271' in dir: continue
        print(dir)
        label = glob(os.path.join(dir, '*'))[0].split('/')[-1].split('_')[-1]
        new_dir = dir.replace('frames24', 'multiview') + '_' + label
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        for idx, data in enumerate(eval_from_path(dir, (32 * 2)/24, transform_val)):
            dash.eval(), rear.eval(), right.eval()
            
            dash_re = dash([d.unsqueeze(0).to(device0) for d in data['dash']]).cpu()
            rear_re = rear([d.unsqueeze(0).to(device1) for d in data['rear']]).cpu()
            right_re = right([d.unsqueeze(0).to(device2) for d in data['right']]).cpu()
            # print(dash_re.shape)
            # return
            dash_re = torch.cat([dash_re, rear_re, right_re], dim=0)
            np.save(os.path.join(new_dir, f'{idx}.npy'), dash_re.cpu().numpy())



if __name__ == '__main__':
    main()
