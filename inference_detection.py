from train_detection import Detection
from glob import glob
from data import eval_from_path
import os
from transform import transform_val_detection
import torch
from collections import defaultdict
import torch.nn.functional as F
from data import PackPathway, uniform_crop_with_boxes


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


def process(video, box):
    video, box = uniform_crop_with_boxes(video, (244, 434), 1, box)
    video = PackPathway()(video)
    return video, box


@torch.no_grad()
def main():
    if not os.path.exists('extracted'):
        os.mkdir('extracted')

    video_id_by_user_id = read_video_ids()
    dash = Detection()
    dash = dash.load_from_checkpoint("lightning_logs/version_1/checkpoints/epoch=59-step=720.ckpt", map_location='cpu')
    dash.eval()
    dash.freeze()

    rear = Detection()
    rear = rear.load_from_checkpoint("lightning_logs/version_2/checkpoints/epoch=59-step=720.ckpt", map_location='cpu')
    rear.eval()
    rear.freeze()

    right = Detection()
    right = right.load_from_checkpoint("lightning_logs/version_3/checkpoints/epoch=59-step=720.ckpt", map_location='cpu')
    right.eval()
    right.freeze()
    dash_weights = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    rear_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    right_weights = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
    # 3, 18
    agg_weights = torch.tensor(
        [dash_weights, rear_weights, right_weights], dtype=torch.float32
    ).cuda()
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
    c = ['42271', '56306', '65818', '72519', '79336']
    target = c[4]
    for dir in dirs:
        if target not in dir: continue
        # if '42271' in dir or '56306' in dir: continue
        # elif '65818' in dir or '72519' in dir or '79336' in dir: continue
        print(dir)
        end = -1
        for data in eval_from_path(dir, (32 * 2)/24, transform_val_detection):
            # dash.eval(), rear.eval(), right.eval()
            if data['start'] == 0:
                
                if file != None:
                    file.close()
                video_id = video_id_by_user_id[(data['user_id'], data['count'])]
                file = open(f'extracted/{video_id}.txt', 'a')
                
                results = []
            # if data['start'] <= 579:
            #     continue
            # print(data['dash_detection'])
            if data['dash_detection'] == None or data['rear_detection'] == None or data['right_detection'] == None:
                prob = torch.zeros([1, 18]).float()
                prob[:, 0] = 1
            else:
                data['dash'], data['dash_detection'] = process(data['dash'], data['dash_detection'])
                data['rear'], data['rear_detection'] = process(data['rear'], data['rear_detection'])
                data['right'], data['right_detection'] = process(data['right'], data['right_detection'])

                dash_re = dash([d.unsqueeze(0).to(device0) for d in data['dash']], data['dash_detection'].to(device0))
                rear_re = rear([d.unsqueeze(0).to(device1) for d in data['rear']], data['rear_detection'].to(device0))
                right_re = right([d.unsqueeze(0).to(device2) for d in data['right']], data['right_detection'].to(device0))
                
                # prob = F.softmax(dash_re, dim=1)
                # prob = prob.cpu()

                prob = F.softmax(dash_re, dim=1) +\
                    F.softmax(rear_re, dim=1) +\
                    F.softmax(right_re,dim=1)
                prob = prob / 3
                prob = prob.cpu()
            
            video_id = video_id_by_user_id[(data['user_id'], data['count'])]
            start = data['start']
            end = data['end']
            
            if len(results) == 0:
                pred = torch.argmax(prob, dim=1).item()
                prob = prob[0][pred]
                # if prob < 0.6:
                #     pred = 0
                file.write(f'{video_id} {pred} {start} {start+1} {prob}\n')
                results.append(prob)
                results.append(prob)
            else:
                mixed_logits = (results[-1] + prob)/2
                pred = torch.argmax(mixed_logits, dim=1).item()
                prob = mixed_logits[0][pred]
                # mix = (results[-1] + output) / 2
                # index = torch.argmax(mix, dim=1).item()
                # prob = mix[0][index]
                # if prob < 0.6:
                #     pred = 0
                file.write(f'{video_id} {pred} {start} {start+1} {prob}\n')
                results[-1] = mixed_logits
                results.append(prob)
            # results[data['user_id'] + '_' + data['count']].append(
            #     (torch.argmax(dash_re, dim=1).item(), torch.argmax(rear_re, dim=1).item(), torch.argmax(right_re, dim=1).item(), data['start'], data['end']))
            print(f'{video_id} {pred} {start} {start+1} {prob}')


if __name__ == '__main__':
    main()
