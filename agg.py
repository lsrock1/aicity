from glob import glob
import os
from tqdm import tqdm
import time


def duration(a):
    return float(a[2]) - float(a[1])


def main():
    txt = glob('posted/*.txt')
    # if not os.path.exists('posted'):
    #     os.mkdir('posted')
    agg = open('final.txt', 'w')
    for i in txt:
        video_id = i[-5]
        if video_id == '0': video_id = '10'
        with open(i) as f:
            lines = f.readlines()
            for line in lines:
                label, start, end = line.strip().split(' ')
                # start = time.strftime('%M:%S', time.gmtime(int(start)))
                # end = time.strftime('%M:%S', time.gmtime(int(end)))
                if label != '0' and (int(end) - int(start) > 3):
                    agg.write(f'{video_id} {label} {start} {end}\n')
                # f.write(f'{video_id} {label} {start} {end}')
    agg.close()


if __name__ == '__main__':
    main()
