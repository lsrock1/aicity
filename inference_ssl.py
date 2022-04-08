from train_ssl import SSL
from train import VideoClassificationLightningModule
from glob import glob
from data import eval_from_path
import os
from transform import transform_infer
import torch
from collections import defaultdict
import torch.nn.functional as F


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
    model = VideoClassificationLightningModule()
    video_id_by_user_id = read_video_ids()
    dash = SSL(model)
    # print(dash.model.blocks[6].proj.weight[0])
    dash = dash.load_from_checkpoint("lightning_logs/dash_ssl/checkpoints/epoch=59-step=1440.ckpt", map_location='cpu')
    dash.eval()
    dash.freeze()
    # print(dash.model.blocks[6].proj.weight[0])

    rear = SSL(model)
    rear = rear.load_from_checkpoint("lightning_logs/rear_ssl/checkpoints/epoch=59-step=1440.ckpt", map_location='cpu')
    rear.eval()
    rear.freeze()

    right = SSL(model)
    right = right.load_from_checkpoint("lightning_logs/side_ssl/checkpoints/epoch=59-step=1440.ckpt", map_location='cpu')
    right.eval()
    right.freeze()

    device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = sorted(glob(os.path.join('dataset/frames24/A2/*')))
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
        for data in eval_from_path(dir, (32 * 2)/24, transform_infer):
            dash.eval(), rear.eval(), right.eval()
            if data['start'] == 0:
                
                if file != None:
                    file.close()
                video_id = video_id_by_user_id[(data['user_id'], data['count'])]
                file = open(f'extracted/{video_id}.txt', 'w')
                # to_ = dir.replace('frames24/A2', 'multiview/A2')
                # if not os.path.exists(os.path.join(to_, f'{video_id}')):
                #     os.makedirs(os.path.join(to_, f'{video_id}'))
                results = []
            # print('dash index', data['dash_index'])
            # print('rear index', data['rear_index'])
            dash_re = dash([d.permute(4, 0, 1, 2, 3).to(device0) for d in data['dash']]).mean(0, keepdim=True)
            rear_re = rear([d.permute(4, 0, 1, 2, 3).to(device1) for d in data['rear']]).mean(0, keepdim=True)
            right_re = right([d.permute(4, 0, 1, 2, 3).to(device2) for d in data['right']]).mean(0, keepdim=True)
            # output = dash_re + rear_re + right_re
            # output = F.softmax(output, dim=1)
            logits = F.softmax(dash_re, dim=1) + F.softmax(rear_re, dim=1) + F.softmax(right_re,dim=1)
            logits = logits.cpu() / 3
            # index = torch.argmax(output, dim=1).item()
            
            video_id = video_id_by_user_id[(data['user_id'], data['count'])]
            # results[data['user_id'] + '_' + data['count']].append(
            #     (video_id, index, data['start'], data['end']))
            start = data['start']
            end = data['end']
            
            if len(results) == 0:
                pred = torch.argmax(logits, dim=1).item()
                prob = logits[0][pred]
                # if prob < 0.6:
                #     pred = 0
                file.write(f'{video_id} {pred} {start} {start+1} {prob}\n')
                results.append(logits)
                results.append(logits)
            else:
                mixed_logits = (results[-1] + logits)/2
                pred = torch.argmax(mixed_logits, dim=1).item()
                prob = mixed_logits[0][pred]
                # mix = (results[-1] + output) / 2
                # index = torch.argmax(mix, dim=1).item()
                # prob = mix[0][index]
                # if prob < 0.6:
                #     pred = 0
                file.write(f'{video_id} {pred} {start} {start+1} {prob}\n')
                results[-1] = mixed_logits
                results.append(logits)
            # results[data['user_id'] + '_' + data['count']].append(
            #     (torch.argmax(dash_re, dim=1).item(), torch.argmax(rear_re, dim=1).item(), torch.argmax(right_re, dim=1).item(), data['start'], data['end']))
            print(f'{video_id} {pred} {start} {start+1} {prob}')


if __name__ == '__main__':
    main()
