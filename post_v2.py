from glob import glob
import os
from tqdm import tqdm


def duration(a):
    return float(a[2]) - float(a[1])


def main():
    threshold = 0.3
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
                video, label, start, end, prob = line.strip().split(' ')
                if float(prob) < threshold:
                    label = "0"
                if len(cleaned) > 1:
                    if cleaned[-1][0] != label and cleaned[-2][0] == label:
                        cleaned[-1][0] = label
                    
                cleaned.append([label, start, end])
            # results = cleaned
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
            # results = phase_2_cleaned
            results = phase_2_cleaned[:2]
            phase_2_cleaned[:2] = []

            while phase_2_cleaned:
                current = phase_2_cleaned.pop(0)
                if len(results) > 1: 
                    prev_duration = duration(results[-2])
                    post_duration = duration(current)

                    if (prev_duration > 1 or post_duration > 1) and \
                        results[-2][0] == current[0] and \
                            post_duration + prev_duration > duration(results[-1]) and current[0] != '0':
                        results.pop(-1)
                        _, start, _ = results.pop(-1)
                        results.append([current[0], start, current[2]])
                    else:
                        results.append(current)
                else:
                    results.append(current)
            
            results = [list(r) for r in results if r[0] != 0 and int(r[2]) - int(r[1]) > 3]

            final_merge = results[:1]

            for line in tqdm(results[1:]):
                # print(float(line[1]) - float(final_merge[-1][2]))
                # print(line[0], final_merge[-1][0])
                if line[0] == final_merge[-1][0] and float(line[1]) - float(final_merge[-1][2]) < 8:
                    print('merging')
                    final_merge[-1][2] = line[2]
                else:
                    final_merge.append(line)

        with open(i.replace('extracted', 'posted'), 'w') as f:
            for result in final_merge:
                f.write(f'{result[0]} {result[1]} {result[2]}\n')


if __name__ == '__main__':
    main()
