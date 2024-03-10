import sys
import os
import random
import matplotlib.pyplot as plt
from os.path import join, isdir, basename, exists
from glob import glob
from skimage.measure import label as mlabel
import numpy as np
from os import fsync, makedirs
from PIL import Image
import skimage.morphology as morph
import torch.nn.functional as F
from scipy.ndimage.morphology import distance_transform_edt as edt
import torch
import math
import gc
from torchvision.utils import make_grid
import zipfile
import shutil
from datetime import datetime
from shutil import copy2
import argparse

IMG_DTYPE = np.float
SEG_DTYPE = np.uint8
TMP_DIR = "./tmp"
if not os.path.isdir(TMP_DIR):
    os.makedirs(TMP_DIR)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def zipDir(dirpath, outFullName):
    """
    压缩指定文件夹
    :param dirpath: 目标文件夹路径
    :param outFullName: 压缩文件保存路径+xxxx.zip
    :return: 无
    """
    zip = zipfile.ZipFile(outFullName, "w", zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(dirpath):
        # 去掉目标跟路径，只对目标文件夹下边的文件及文件夹进行压缩
        fpath = path.replace(dirpath, '')
        for filename in filenames:
            zip.write(os.path.join(path, filename), os.path.join(fpath, filename))
    zip.close()


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self, shape=1):
        self.shape = shape
        self.reset()

    def reset(self):
        self.val = np.zeros(self.shape)
        self.avg = np.zeros(self.shape)
        self.sum = np.zeros(self.shape)
        self.count = 0

    def update(self, val, n=1):
        val = np.array(val)
        assert val.shape == self.val.shape
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def visualize(data, filename):
    assert (len(data.shape) == 3)  # height*width*channels
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))  # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + '.png')
    return img


def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return:
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files


def create_cityscapes_label_colormap():
    return {
        0: [128, 64, 128],
        1: [244, 35, 232],
        2: [70, 70, 70],
        3: [102, 102, 156],
        4: [190, 153, 153],
        5: [153, 153, 153],
        6: [250, 170, 30],
        7: [220, 220, 0],
        8: [107, 142, 35],
        9: [152, 251, 152],
        10: [70, 130, 180],
        11: [220, 20, 60],
        12: [255, 0, 0],
        13: [0, 0, 142],
        14: [0, 0, 70],
        15: [0, 60, 100],
        16: [0, 80, 100],
        17: [0, 0, 230],
        18: [119, 11, 32],
        255: [255, 255, 255]
    }


def create_binary_colormap():
    return {
        0: [0, 0, 0],
        1: [125, 125, 125],
        2: [255, 255, 255]
    }


def create_pascal_label_colormap():
    def bit_get(val, idx):
        return (val >> idx) & 1

    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    dict_colormap = {}
    for i in range(256):
        dict_colormap[i] = colormap[i, :].tolist()

    return dict_colormap


def get_colormap(dataset):
    if dataset == 'cityscapes' or dataset == 'active_cityscapes_image' or dataset == 'active_cityscapes_region':
        return create_cityscapes_label_colormap()
    elif dataset == 'binary':
        return create_binary_colormap()
    elif dataset == 'pascal' or dataset == 'active_pascal_image' or dataset == 'active_pascal_region':
        return create_pascal_label_colormap()

    raise Exception('No colormap for dataset found')


def map_segmentations_to_colors(segmentations, dataset):
    rgb_masks = []
    for segmentation in segmentations:
        rgb_mask = map_segmentation_to_colors(segmentation, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def map_binary_output_mask_to_colors(binary_segmentation):
    rgb_masks = []
    for segmentation in binary_segmentation:
        rgb_mask = map_segmentation_to_colors(segmentation, 'binary')
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def map_segmentation_to_colors(segmentation, dataset):
    colormap = get_colormap(dataset)
    colored_segmentation = np.zeros((segmentation.shape[0], segmentation.shape[1], 3))

    for label in np.unique(segmentation).tolist():
        colored_segmentation[segmentation == label, :] = colormap[label]

    colored_segmentation /= 255.0
    return colored_segmentation


## Score measure

def numeric_score(y_pred, y_true):
    """Compute True Positive, True Negative, False Positive, False Negative classifications
    between a prediction and its ground truth
    :param y_pred: prediction
    :param y_true: ground truth
    :return: True Positive, True Negative, False Positive, False Negative
    """
    y_pred = y_pred.astype(int)
    y_true = y_true.astype(int)
    FP = float(np.sum((y_pred == 1) & (y_true == 0)))
    FN = float(np.sum((y_pred == 0) & (y_true == 1)))
    TP = float(np.sum((y_pred == 1) & (y_true == 1)))
    TN = float(np.sum((y_pred == 0) & (y_true == 0)))
    return FP, FN, TP, TN


def jaccard_score(y_pred, y_true):
    """Compute Jaccard Score (= Intersection / Union) between a prediction and its ground truth
    :param y_pred: prediction
    :param y_true: ground truth
    :return: Jaccard score value
    """
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    if union == 0:
        return 1.
    else:
        return float(intersection) / union


def pixel_wise_accuracy(y_true, y_pred):
    """Compute Pixel-wise accuracy (= number of well classified pixel / total number of pixel)
    between a prediction and its ground truth
    :param y_pred: prediction
    :param y_true: ground truth
    :return: Pixel-wise accuracy value
    """
    y_true_f = y_true.reshape([1, 224 * 224])
    y_pred_f = y_pred.reshape([1, 224 * 224])
    return 1 - np.count_nonzero(y_pred_f - y_true_f) / y_true_f.shape[1]


def precision_score(y_pred, y_true):
    """Compute precision (= TP / (TP+FP)) between a prediction and its ground truth
    :param y_pred: prediction
    :param y_true: ground truth
    :return: Precision score value
    """
    FP, FN, TP, TN = numeric_score(y_pred, y_true)
    if (TP + FP) <= 0:
        return 0.
    else:
        return np.divide(TP, TP + FP)


def sensitivity_score(y_pred, y_true):
    """Compute sensitivity (= TP / (TP+FN)) between a prediction and its ground truth
    :param y_pred: prediction
    :param y_true: ground truth
    :return: Sensitivity score value
    """
    FP, FN, TP, TN = numeric_score(y_pred, y_true)
    if (TP + FN) <= 0:
        return 0.
    else:
        return np.divide(TP, TP + FN)


def get_random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.random() for i in range(3)]
    return r, g, b


def show_figures(imgs, dir='./tmp'):
    figs, axes = plt.subplots(len(imgs), 1, figsize=(2.5, 2.5 * len(imgs)))
    axes = list(axes.flat)
    for i in range(len(axes)):
        ax = axes[i]
        ax.imshow(imgs[i])
        ax.set_axis_off()
    plt.subplots_adjust(wspace=0.03, hspace=0.01)
    plt.tight_layout()
    plt.savefig(dir + '_batch.png')
