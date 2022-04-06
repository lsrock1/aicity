from glob import glob
import os
from tqdm import tqdm


def duration(a):
    return float(a[2]) - float(a[1])


def main():
    txt = glob('extracted/*.txt')
    if not os.path.exists('posted'):
        os.mkdir('posted')
    for i in txt:
        with open(i) as f:
            lines = f.readlines()
            prev = -1
            prev_start = 0
            # cleansing
            cleaned = []

            for line in tqdm(lines):
                video, label, start, end = line.strip().split(' ')
                if len(cleaned) > 1:
                    if cleaned[-1][0] != label and cleaned[-2][0] == label:
                        cleaned[-1][0] = label
                    
                cleaned.append([label, start, end])
            
            # merge
            phase_2_cleaned = []
            for line in tqdm(cleaned):
                label, start, end = line
                # video, label, start, end = line.strip().split(' ')
                if int(label) == prev:
                    pass
                else:
                    if prev != -1:
                        phase_2_cleaned.append((prev, prev_start, start))
                    prev = int(label)
                    prev_start = start

            results = phase_2_cleaned[:2]
            for idx in range(2, len(phase_2_cleaned)):
                prev_duration = duration(phase_2_cleaned[idx-2])
                post_duration = duration(phase_2_cleaned[idx])

                if prev_duration > 2 and post_duration > 2 and phase_2_cleaned[idx-2][0] == phase_2_cleaned[idx][0] and post_duration + prev_duration > duration(phase_2_cleaned[idx-1]):
                    results.pop(-1)
                    _, start, _ = results.pop(-1)
                    results.append([phase_2_cleaned[idx][0], start, phase_2_cleaned[idx][2]])
                    
                else:
                    results.append(phase_2_cleaned[idx])

        with open(i.replace('extracted', 'posted'), 'w') as f:
            for result in results:
                f.write(f'{result[0]} {result[1]} {result[2]}\n')


if __name__ == '__main__':
    main()
