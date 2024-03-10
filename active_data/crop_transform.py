from PIL import Image
from torchvision import transforms
# import monai.transforms as monaintransform
import pandas as pd
import os
import glob as gb
import torch
import numpy as np
import random
import warnings
from scipy import ndimage
import cv2
from driver import std, mean
from os.path import join
from commons.utils import visualize
from active_data.auto_augment import AutoAugment
from skimage import morphology, io


def process_mask(mask_path, radius=1):
    mask = Image.open(mask_path)
    label = np.array(mask)
    new_label = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
    new_label[label[:, :, 0] > 255 * 0.5] = 1  # inside
    boun = morphology.dilation(new_label) & (~morphology.erosion(new_label, morphology.disk(radius)))
    new_label[boun > 0] = 2  # boundary
    target = Image.fromarray(new_label.astype(np.uint8))
    return target


def train_aug_(image, num, output_size=(512, 512), isGT=False, auto_aug=None):
    image_list = []
    h, w = image.size
    if not isGT:
        trans = transforms.Compose([
            transforms.RandomAffine([-90, 90], translate=[0.1, 0.1], shear=[-10, 10],
                                    scale=[0.8, 1.2]),
            transforms.RandomChoice([
                transforms.RandomCrop(output_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]),
            transforms.Resize(output_size),
            auto_aug,
            # transforms.RandomChoice([
            #     transforms.ColorJitter(brightness=0.1),
            #     transforms.ColorJitter(contrast=0.2),
            #     transforms.ColorJitter(saturation=0.1),
            #     transforms.ColorJitter(hue=0.15),
            #     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            #     transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15),
            #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            # ]),
        ])
    else:
        trans = transforms.Compose([
            transforms.RandomAffine([-90, 90], translate=[0.1, 0.1], shear=[-10, 10],
                                    scale=[0.8, 1.2]),
            transforms.RandomChoice([
                transforms.RandomCrop(output_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]),
            transforms.Resize(output_size),
        ])
    for i in range(num):
        img = trans(image)
        image_list.append(img)
    return image_list


from PIL import Image, ImageFilter


class RandomGaussBlur(object):
    """Random GaussBlurring on image by radius parameter.
    Args:
        radius (list, tuple): radius range for selecting from; you'd better set it < 2
    """

    def __init__(self, radius=None):
        if radius is not None:
            assert isinstance(radius, (tuple, list)) and len(radius) == 2, \
                "radius should be a list or tuple and it must be of length 2."
            self.radius = random.uniform(radius[0], radius[1])
        else:
            self.radius = 0.0

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))

    def __repr__(self):
        return self.__class__.__name__ + '(Gaussian Blur radius={0})'.format(self.radius)


from scipy.ndimage.filters import gaussian_filter
import numbers
from scipy.ndimage.interpolation import map_coordinates

class RandomElastic(object):
    """Random Elastic transformation by CV2 method on image by alpha, sigma parameter.
        # you can refer to:  https://blog.csdn.net/qq_27261889/article/details/80720359
        # https://blog.csdn.net/maliang_1993/article/details/82020596
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html#scipy.ndimage.map_coordinates
    Args:
        alpha (float): alpha value for Elastic transformation, factor
        if alpha is 0, output is original whatever the sigma;
        if alpha is 1, output only depends on sigma parameter;
        if alpha < 1 or > 1, it zoom in or out the sigma's Relevant dx, dy.
        sigma (float): sigma value for Elastic transformation, should be \ in (0.05,0.1)
        mask (PIL Image) if not assign, set None.
    """

    def __init__(self, alpha, sigma):
        assert isinstance(alpha, numbers.Number) and isinstance(sigma, numbers.Number), \
            "alpha and sigma should be a single number."
        assert 0.05 <= sigma <= 0.1, \
            "In pathological image, sigma should be in (0.05,0.1)"
        self.alpha = alpha
        self.sigma = sigma

    @staticmethod
    def RandomElasticCV2(img, alpha, sigma, mask=None):
        alpha = img.shape[1] * alpha
        sigma = img.shape[1] * sigma
        if mask is not None:
            mask = np.array(mask).astype(np.uint8)
            img = np.concatenate((img, mask[..., None]), axis=2)

        shape = img.shape

        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        # dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        img = map_coordinates(img, indices, order=0, mode='reflect').reshape(shape)
        if mask is not None:
            return Image.fromarray(img[..., :3]), Image.fromarray(img[..., 3])
        else:
            return Image.fromarray(img)

    def __call__(self, img, mask=None):
        return self.RandomElasticCV2(np.array(img), self.alpha, self.sigma, mask)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(alpha value={0})'.format(self.alpha)
        format_string += ', sigma={0}'.format(self.sigma)
        format_string += ')'
        return format_string


def process_label(gt_list):
    pics = []
    for i in range(len(gt_list)):
        # process label
        label = gt_list[i]
        if isinstance(label, np.ndarray):
            # handle numpy array
            label_tensor = torch.from_numpy(label)
            # backward compatibility
            pics.append(label_tensor.long())

        # handle PIL Image
        if label.mode == 'I':
            label_tensor = torch.from_numpy(np.array(label, np.int32, copy=False))
        elif label.mode == 'I;16':
            label_tensor = torch.from_numpy(np.array(label, np.int16, copy=False))
        else:
            label_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(label.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if label.mode == 'YCbCr':
            nchannel = 3
        elif label.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(label.mode)
        label_tensor = label_tensor.view(label.size[1], label.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        label_tensor = label_tensor.transpose(0, 1).transpose(0, 2).contiguous()
        label_tensor = torch.unsqueeze(label_tensor, dim=0)
        # label_tensor = label_tensor.view(label.size[1], label.size[0])
        pics.append(label_tensor.long())
    return pics


class argumentation_train(object):
    def __init__(self, output_size=(224, 224), aug_num=2):
        self.output_size = output_size
        self.aug_num = aug_num
        self.auto_aug = AutoAugment()

    def __call__(self, image, groundtruth):
        image_list = []
        gt_list = []
        h, w = image.size
        for ii in range(self.aug_num):
            seed = np.random.randint(2147483647)
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            image_list0_img = train_aug_(image, 1, output_size=self.output_size, isGT=False, auto_aug=self.auto_aug)
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            image_list0_gt = train_aug_(groundtruth, 1, output_size=self.output_size, isGT=True)
            image_list += image_list0_img
            gt_list += image_list0_gt
        nomalize_img = transforms.Lambda(lambda crops: torch.stack(
            [transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean,
                                                                             std=std)])(crop) for crop
             in
             crops]))
        image_list = nomalize_img(image_list)
        pics = process_label(gt_list)
        gt_list = torch.cat(pics, dim=0)
        assert gt_list.dtype == torch.int64
        # for i in range(len(image_list)):
        #     visualize(np.clip((np.transpose(np.array(image_list[i]), (1, 2, 0)) * std + mean), 0, 1),
        #               join('./tmp', "image_list_" + str(i) + "_img_images"))
        #     visualize(np.transpose(np.array(gt_list[i]) / 2, (1, 2, 0)),
        #               join('./tmp', "image_list_" + str(i) + "_gt_images"))
        return image_list, gt_list


class argumentation_val_normal(object):
    def __init__(self, output_size=(224, 224)):
        self.output_size = output_size

    def __call__(self, image, groundtruth):
        image_list = [image]
        gt_list = [groundtruth]
        nomalize_img = transforms.Lambda(lambda crops: torch.stack(
            [transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean,
                                                                             std=std)])(crop) for crop
             in
             crops]))
        # process labels:
        pics = process_label(gt_list)
        image_list = nomalize_img(image_list)
        gt_list = torch.cat(pics, dim=0)
        assert gt_list.dtype == torch.int64
        return image_list, gt_list


class argumentation_val_ae(object):
    def __init__(self, output_size=(128, 128)):
        self.output_size = output_size

    def __call__(self, image, groundtruth):
        image_list = [image]
        gt_list = [groundtruth]
        nomalize_img = transforms.Lambda(lambda crops: torch.stack(
            [transforms.Compose(
                [transforms.Resize(self.output_size), transforms.ToTensor(), transforms.Normalize(mean=mean,
                                                                                                  std=std)])(crop) for
             crop
             in
             crops]))
        # process labels:
        pics = process_label(gt_list)
        image_list = nomalize_img(image_list)
        gt_list = torch.cat(pics, dim=0)
        assert gt_list.dtype == torch.int64
        return image_list, gt_list


class argumentation_test_normal(object):
    def __init__(self, output_size=(224, 224)):
        self.output_size = output_size

    def __call__(self, image, groundtruth):
        image_list = [image]
        gt_list = [groundtruth]
        nomalize_img = transforms.Lambda(lambda crops: torch.stack(
            [transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean,
                                                                             std=std)])(crop) for crop
             in
             crops]))
        # process labels:
        image_list = nomalize_img(image_list)
        pics = process_label(gt_list)
        gt_list = torch.cat(pics, dim=0)
        assert gt_list.dtype == torch.int64
        return image_list, gt_list
