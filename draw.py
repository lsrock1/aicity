import cv2
from glob import glob
import numpy as np
import os


def main():
    paths = glob('dataset/bboxes/A1/*/*/*')
    
    for path in paths:
        if '38058_0' not in path: continue
        if not os.path.exists(path.replace('bboxes', 'bboxes_draw')):
            os.makedirs(path.replace('bboxes', 'bboxes_draw'))
        for box in glob(os.path.join(path, '*.npy')):
            img = cv2.imread(box.replace('bboxes', 'frames24').replace('.npy', '.png'))
            try:
                boxes = np.load(box)
            except:
                print(box)
                break
            if len(boxes) > 2:
                print(box)
            for b in boxes:
                x1, y1, x2, y2 = b
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
                cv2.imwrite(box.replace('bboxes', 'bboxes_draw').replace('.npy', '.png'), img)


if __name__ == '__main__':
    main()
