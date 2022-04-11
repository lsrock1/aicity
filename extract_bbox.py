import multiprocessing
import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import numpy as np
import os
from glob import glob
import cv2
from functools import partial
from tqdm import tqdm

# This method takes in an image and generates the bounding boxes for people in the image.
def get_person_bboxes(inp_img, predictor):
    predictions = predictor(inp_img)['instances'].to('cpu')
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = np.array(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None)
    predicted_boxes = boxes[np.logical_and(classes==0, scores>0.75 )].tensor.cpu() # only person
    return predicted_boxes.numpy()


def get_iou(bb1, bb2):

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    return intersection_area


def func(path, predictor):
    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # set threshold for this model
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    # predictor = DefaultPredictor(cfg)
    imgs = sorted(glob(os.path.join(path, '*.png')), key=lambda x: int(x.split('/')[-1].split('.')[0]))
    if not os.path.exists(path.replace('frames24', 'bboxes')):
        os.makedirs(path.replace('frames24', 'bboxes'))
    # img = imgs[len(imgs)//2]
    
    for img in imgs:
        img_path = img
        if os.path.exists(img_path.replace('frames24', 'bboxes')[:-4] + '.npy'):
            continue
        
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)#[None, ...]
        bboxes = get_person_bboxes(img, predictor)
        bboxes = [b for b in bboxes]
        if len(bboxes) == 2:
            if get_iou(bboxes[0], bboxes[1]) > 0:
                # select bigger box
                if (bboxes[0][2] - bboxes[0][0]) * (bboxes[0][3] - bboxes[0][1]) > (bboxes[1][2] - bboxes[1][0]) * (bboxes[1][3] - bboxes[1][1]):
                    bboxes = [bboxes[0]]
                else:
                    bboxes = [bboxes[1]]
                print('select bigger')
            else:
                # select right box
                if bboxes[0][2] + bboxes[0][0] > bboxes[1][2] + bboxes[1][0]:
                    bboxes = [bboxes[0]]
                else:
                    bboxes = [bboxes[1]]
                print('select right')
        elif len(bboxes) > 2:
            print('wrong case')
        np.save(img_path.replace('frames24', 'bboxes')[:-4] + '.npy', np.array(bboxes))

def main():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    # use spawn method to run multiple processes
    multiprocessing.set_start_method('spawn')

    paths = glob('dataset/frames24/A1/*_*_*_*/*/*')
    paths = sorted(paths)
    paths = [p for p in paths if 'Rear' in p]
    alloc = partial(func, predictor=predictor)

    # with multiprocessing.Pool(processes=8) as pool:
    #     for _ in tqdm(pool.imap_unordered(alloc, paths), total=len(paths)):
    #         pass

    paths = glob('dataset/frames24/A2/*/*')
    # multiprocessing.Pool(multiprocessing.cpu_count()).map(alloc, paths)
    with multiprocessing.Pool(processes=8) as pool:
        for _ in tqdm(pool.imap_unordered(alloc, paths), total=len(paths)):
            pass


if __name__ == '__main__':
    main()
