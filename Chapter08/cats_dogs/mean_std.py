import os

import cv2
import numpy as np

from multiprocessing import Pool


def meanStd(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, (256, 256))
    img = img / 255.
    _mean = np.mean(img, axis=(0,1))
    _std = np.std(img, axis=(0,1))
    return _mean, _std


if __name__ == '__main__':
    datapath = '/media/john/FastData/cats-dogs-kaggle/train'

    img_list = []
    for path, subdirs, files in os.walk(datapath):
        for name in files:
            filename = os.path.join(path, name)
            if os.path.isfile(filename):
                img_list.append(filename)

    pool = Pool(4).map(meanStd, img_list)

    _mean, _std = zip(*pool)
    img_mean = sum(_mean)
    img_std = sum(_std)
    img_mean = img_mean / 25000.
    img_std = img_std / 25000.
    print(img_mean)
    print(img_std)
