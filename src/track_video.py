import argparse
import logging
import time
from collections import deque

import cv2
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

class Target(object):

    def __init__(self, human, number):
        self.human = human
        self.number = number

    
    def d(self, human):
        def _d(bp1, bp2):
            return ((bp1.x - bp2.x)**2 + (bp1.y - bp2.y)**2)**0.5
        indices = set(self.human.body_parts.keys()) & set(human.body_parts.keys())
        avg_d = sum([_d(self.human.body_parts[i], human.body_parts[i]) for i in indices]) / len(indices) if len(indices) != 0 else 0 #np.sqrt(2)
        return avg_d


def connect(humans, previous_targets):
    print('connect start!!!')
    hs = humans
    m = np.array([np.array([p.d(h) for p in previous_targets]) for h in humans])
    targets = []
    for k, h in enumerate(hs):
        if not previous_targets:
            print('previous target is empty')
            targets.append(Target(h, k))
            continue
        i, j = min_element(m)
        print('connect to previous number: {}'.format(previous_targets[j].number))
        
        targets.append(Target(humans[i], previous_targets[j].number))
        m = submatrix(m, i, j)
        humans = list(map(lambda t: t[1], filter(lambda t: t[0] != i, enumerate(humans))))
        previous_targets = list(map(lambda t: t[1], filter(lambda t: t[0] != j, enumerate(previous_targets))))
    return targets


def min_element(m):
    return (0, 0) if m.size == 0 else np.unravel_index(np.argmin(m, axis=None), m.shape)


def submatrix(m, i, j):
    m = list(map(lambda x: np.concatenate([x[:j], x[j+1:]]), m))
    m = list(filter(lambda t: t[0] != i, enumerate(m)))
    m = list(map(lambda x: x[1], m))
    return np.array(m)


def replace2(m, i, j):
    if m.size == 0:
        return m
    m[i, :] = 2
    m[:, j] = 2
    return m


def stringToBool(input_str):
    if input_str.lower() in ('true', '1'):
        return True
    elif input_str.lower() in ('false', '0'):
        return False


def draw_numbers(image, targets):
    if not targets:
        return image
    image_h, image_w = image.shape[:2]
    for t in targets:
        human = t.human
        min_key = sorted(human.body_parts.keys())[0]
        center = (int(human.body_parts[min_key].x * image_w + 0.5), int(human.body_parts[min_key].y * image_h + 0.5))
        cv2.putText(image, str(t.number + 1), center, cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv2.LINE_AA)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=stringToBool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = None
    targets = None
    previous_targets = []
    targets_stock = deque([])
    if (cap.isOpened()== False):
        print("Error opening video stream or file")
    logger.info('file read start')
    while(cap.isOpened()):
        ret_val, image = cap.read()
        
        if ret_val:
            h, w = image.shape[:2]
            humans = e.inference(image)

            if args.showBG == False: image = np.zeros(image.shape)
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            targets = connect(humans, previous_targets)

            # 最大5フレーム分のtargetsを保管しておく
            if len(targets_stock) == 5: targets_stock.popleft()
            targets_stock.append(targets)

            for stock in deque(reversed(targets_stock)):
                # 人数が減っていたら過去の検出情報を採用する
                if len(targets) >= len(stock): continue
                targets = stock
                break

            image = draw_numbers(image, targets)
            logger.debug('show+')
            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            previous_targets = targets
            if not out:
                out = cv2.VideoWriter('../images/output12.avi', fourcc, 15.0, (w, h))
            out.write(image)
            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break
        else:
            break
    logger.info('file read finished')
    cap.release()
    out.release()
    cv2.destroyAllWindows()
logger.debug('finished+')
