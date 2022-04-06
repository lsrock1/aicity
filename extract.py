from train import VideoClassificationLightningModule
from glob import glob
from data import eval_from_path
import os
from transform import transform_val
import torch
from collections import defaultdict
import torch.nn.functional as F


def read_video_ids():
    video_id_by_user_id = {}
    with open('../2022/A2/video_ids.csv') as f:
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
    video_id_by_user_id = read_video_ids()
    dash = VideoClassificationLightningModule()
    print(dash.model.blocks[6].proj.weight[0])
    dash = dash.load_from_checkpoint("lightning_logs/dash/checkpoints/epoch=59-step=600.ckpt", map_location='cpu')
    dash.eval()
    dash.freeze()
    print(dash.model.blocks[6].proj.weight[0])

    rear = VideoClassificationLightningModule()
    rear = rear.load_from_checkpoint("lightning_logs/rear/checkpoints/epoch=59-step=600.ckpt", map_location='cpu')
    rear.eval()
    rear.freeze()

    right = VideoClassificationLightningModule()
    right = right.load_from_checkpoint("lightning_logs/side/checkpoints/epoch=59-step=600.ckpt", map_location='cpu')
    right.eval()
    right.freeze()

    device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    dirs = glob(os.path.join('/home/vitallab/ssd/vitallab/frames24/A2/*'))
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
        for data in eval_from_path(dir, (32 * 2)/24, transform_val):
            dash.eval(), rear.eval(), right.eval()
            if data['start'] == 0:
                    
                if file != None:
                    file.close()
                video_id = video_id_by_user_id[(data['user_id'], data['count'])]
                file = open(f'/home/vitallab/ssd/vitallab/ar/{video_id}.txt', 'w')
                results = []
            # print('dash index', data['dash_index'])
            # print('rear index', data['rear_index'])
            dash_re = dash([d.unsqueeze(0).to(device0) for d in data['dash']])
            rear_re = rear([d.unsqueeze(0).to(device1) for d in data['rear']])
            right_re = right([d.unsqueeze(0).to(device2) for d in data['right']])
            output = F.softmax(dash_re, dim=1).cpu() + F.softmax(rear_re, dim=1).cpu() + F.softmax(right_re,dim=1).cpu()
            index = torch.argmax(output, dim=1).item()
            video_id = video_id_by_user_id[(data['user_id'], data['count'])]
            # results[data['user_id'] + '_' + data['count']].append(
            #     (video_id, index, data['start'], data['end']))
            start = data['start']
            end = data['end']
            
            if len(results) == 0:
                file.write(f'{video_id} {index} {start} {start+1}\n')
                results.append(output)
                results.append(output)
            else:
                index = torch.argmax((results[-1] + output) / 2, dim=1).item()
                file.write(f'{video_id} {index} {start} {start+1}\n')
                results[-1] = (results[-1] + output) / 2
                results.append(output)
            # results[data['user_id'] + '_' + data['count']].append(
            #     (torch.argmax(dash_re, dim=1).item(), torch.argmax(rear_re, dim=1).item(), torch.argmax(right_re, dim=1).item(), data['start'], data['end']))
            print(f'{video_id} {index} {start} {start+1}')
        # p = data['user_id'] + '_' + data['count']
        # with open(f'/home/vitallab/ssd/vitallab/ar/{p}.txt', 'w') as f:
        #     for k in results[p]:
        #         f.write(f'{k[0]} {k[1]} {k[2]} {k[3]}\n')
                # for i in v:
                #     f.write('\t'.join(map(str, i)) + '\n')


if __name__ == '__main__':
    main()
