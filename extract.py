from train import VideoClassificationLightningModule
from glob import glob
from data import eval_from_path
import os
from transform import transform_val
import torch
from collections import defaultdict


@torch.no_grad()
def main():
    dash = VideoClassificationLightningModule()
    dash.load_from_checkpoint("lightning_logs/dash/checkpoints/epoch=59-step=600.ckpt")
    dash.eval()

    rear = VideoClassificationLightningModule()
    rear.load_from_checkpoint("lightning_logs/rear/checkpoints/epoch=59-step=600.ckpt")
    rear.eval()

    right = VideoClassificationLightningModule()
    right.load_from_checkpoint("lightning_logs/side/checkpoints/epoch=59-step=600.ckpt")
    right.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dirs = glob(os.path.join('/home/vitallab/ssd/vitallab/frames24/A2/*'))
    results = defaultdict(list)

    dash = dash.to(device)
    rear = rear.to(device)
    right = right.to(device)
    for dir in dirs:
        print(dir)
        for data in eval_from_path(dir, (32 * 2)/24, transform_val):
            dash_re = dash([d.unsqueeze(0).to(device) for d in data['dash']])
            rear_re = rear([d.unsqueeze(0).to(device) for d in data['rear']])
            right_re = right([d.unsqueeze(0).to(device) for d in data['right']])
            index = torch.argmax(dash_re + rear_re + right_re, dim=1).item()
            results[data['user_id'] + '_' + data['count']].append(
                (index, data['start'], data['end']))
            # results[data['user_id'] + '_' + data['count']].append(
            #     (torch.argmax(dash_re, dim=1).item(), torch.argmax(rear_re, dim=1).item(), torch.argmax(right_re, dim=1).item(), data['start'], data['end']))
            print(results[data['user_id'] + '_' + data['count']][-1])
        p = data['user_id'] + '_' + data['count']
        with open(f'/home/vitallab/ssd/vitallab/ar/{p}.txt', 'w') as f:
            for k in results[p]:
                f.write(f'{k[0]} {k[1]} {k[2]}\n')
                # for i in v:
                #     f.write('\t'.join(map(str, i)) + '\n')


if __name__ == '__main__':
    main()
