# -*- coding: utf-8 -*-
# @Time    : 19/11/7 10:01
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : InstanceDataLoader.py
from torch.utils import data
import numpy as np
import os
from os.path import isdir
from active_data.constant import *
from commons.utils import visualize
from os.path import join
from PIL import Image
import random
import torch
from active_data.preprocess.InstanceDataPreprocess import make_train_dataset, make_test_dataset, \
    make_vali_dataset
from active_data.crop_transform import process_mask

TMP_DIR = "./tmp"
if not isdir(TMP_DIR):
    os.makedirs(TMP_DIR)

DATATYPES = ['labelled', 'valid', 'test', 'unlabeled', 'data_pool']


def img_loader(path, num_channels):
    if num_channels == 1:
        img = Image.open(path)
    else:
        img = Image.open(path).convert('RGB')

    return img


def load_test_data(data_name=GLAND_DATA, root='', partA='A'):
    """
    Load a data source given it's name make_dqn_dataset
    """
    if data_name == GLAND_DATA:
        dir = root
    if data_name == MONUSEG_DATA:
        dir = join(root, 'MoNuSegTestData')
    if data_name == CRAG_DATA:
        dir = join(root, 'valid')
    if data_name == TNBC_DATA:
        dir = join(root, 'test')
    image_paths, image_gt, image_label, image_name = make_test_dataset(dir, data_name=data_name, partA=partA)
    final_data = {}
    final_data['test_x'] = image_paths
    final_data['test_y'] = image_gt
    final_data['test_z'] = image_label
    final_data['test_n'] = image_name
    print('Test images %d ' % (len(image_name)), image_name)
    return final_data


def load_train_vali_data(training_size=1800, data_name=GLAND_DATA, root='', is_full_image=False, seed=666):
    """
    Load a data source given it's name
    """
    post_fix = ['label.png']
    image_paths, image_gt, image_label, image_name = make_train_dataset(root, post_fix=post_fix, data_name=data_name,
                                                                        is_full_image=is_full_image, seed=seed)
    final_data = {}
    final_data[DATA_POOL_X] = image_paths
    final_data[DATA_POOL_Y] = image_gt
    final_data[DATA_POOL_Z] = image_label
    final_data[DATA_POOL_N] = image_name
    train_file_names = [nn[:nn.rfind('_') + 1] for nn in image_name]
    nns = set()
    for nn in train_file_names:
        nns.add(nn)
    print('Total train images in data pool %d ' % (len(nns)), nns)
    image_paths, image_gt, image_label, image_name = make_vali_dataset(root, data_set=data_name)
    final_data['valid_x'] = image_paths
    final_data['valid_y'] = image_gt
    final_data['valid_z'] = image_label
    final_data['valid_n'] = image_name
    print('Total valid images in data pool %d ' % (len(image_name)), image_name)
    # final_data, inital_split = split_for_simulation(final_data, training_size=training_size)
    final_data, inital_split = split_for_simulation_img_base(final_data, nns, training_size=training_size, seed=seed)
    return final_data, inital_split


def split_for_simulation(all_data, training_size=5000):
    """split initial dataset in train, validation and test sets based on indexes
    """
    index_labelled = np.arange(0, training_size, 1)
    index_unlabelled = np.arange(training_size, len(all_data[DATA_POOL_X]), 1)
    inital_split = {}
    inital_split[INDEX_LABELLED_0] = index_labelled
    inital_split[INDEX_UNLABELLED_0] = index_unlabelled
    inital_split[INDEX_LABELLED] = index_labelled
    inital_split[INDEX_UNLABELLED] = index_unlabelled
    return all_data, inital_split


def split_for_simulation_img_base(all_data, nns, training_size=5000, seed=666):
    import math
    """split initial dataset in train, validation and test sets based on indexes
    """
    nn = list(nns)
    nn = sorted(nn)
    per = training_size / len(all_data[DATA_POOL_X])
    len_imgs = round(len(nn) * per)
    np.random.seed(seed)
    np.random.shuffle(nn)
    nns = nn[:len_imgs]
    names = all_data[DATA_POOL_N]
    index_labelled = [ii for ii in range(len(names)) if names[ii][:names[ii].rfind('_') + 1] in nns]
    index_unlabelled = [ii for ii in range(len(names)) if names[ii][:names[ii].rfind('_') + 1] not in nns]
    inital_split = {}
    inital_split[INDEX_LABELLED_0] = index_labelled
    inital_split[INDEX_UNLABELLED_0] = index_unlabelled
    inital_split[INDEX_LABELLED] = index_labelled
    inital_split[INDEX_UNLABELLED] = index_unlabelled
    return all_data, inital_split


class INSTANCELABELDataLoader(data.Dataset):
    """
    Dataloader
    """

    def __init__(self, total_data, total_segment, total_label, transform=None, gt_trans=None):
        self.transform = transform
        self.gt_trains = gt_trans
        self.labels = total_label
        self.images = total_data
        self.segment = total_segment
        print("ISICUNLABELLEDDataLoader data size: " + str(len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_patch = self.images[index]
        image_label = self.labels[index]
        image_segment = self.segment[index]
        if self.transform:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            image_patch = self.transform(np.transpose(image_patch, (1, 2, 0)))
            if self.gt_trains:
                random.seed(seed)
                torch.manual_seed(seed)
                image_segment = self.gt_trains(np.transpose(np.array(image_segment, dtype=np.float32), (1, 2, 0)))
        else:
            image_patch = (image_patch - np.mean(image_patch)) / np.std(image_patch)
            image_patch = (image_patch - np.min(image_patch)) / (np.max(image_patch) - np.min(image_patch))
        return {
            "image_patch": image_patch,
            'image_label': image_label,
            'image_segment': image_segment
        }


class INSTANCEDataset(data.Dataset):
    def __init__(self, total_data, split='labelled', labelled_index=None, transform=None, radius=1):
        self.transforms = transform
        self.total_data = total_data
        self.split = split
        self.radius = radius
        if split not in DATATYPES:
            raise ValueError("not support split type!")
        if split == 'labelled' or split == 'unlabeled':
            self.images = [self.total_data[DATA_POOL_X][i] for i in range(len(self.total_data[DATA_POOL_X])) if
                           i in labelled_index]
            self.segment = [self.total_data[DATA_POOL_Y][i] for i in range(len(self.total_data[DATA_POOL_Y])) if
                            i in labelled_index]
            self.labels = [self.total_data[DATA_POOL_Z][i] for i in range(len(self.total_data[DATA_POOL_Z])) if
                           i in labelled_index]
            self.name = [self.total_data[DATA_POOL_N][i] for i in range(len(self.total_data[DATA_POOL_N])) if
                         i in labelled_index]
            # train_file_names = [nn[:nn.rfind('_')] + '.png' for nn in self.name]
            # nns = set()
            # for nn in train_file_names:
            #     nns.add(nn)
            # print('Selected images %d ' % (len(nns)), nns)

        else:
            self.images = self.total_data[split + '_x']
            self.segment = self.total_data[split + '_y']
            self.labels = self.total_data[split + '_z']
            self.name = self.total_data[split + '_n']

        print("INSTDataLoader " + split + " data size: " + str(len(self.images)))
        train_file_names = [nn[:nn.rfind('_')] + '.png' for nn in self.name]
        nns = set()
        for nn in train_file_names:
            nns.add(nn)
        print(split + ' images %d ' % (len(nns)), sorted(list(nns)))

    def __len__(self):
        if 'Warwick_QU' in self.images[0]:
            return len(self.images)
        if 'MoNuSeg' in self.images[0]:
            return len(self.images)
        if 'CRAG' in self.images[0]:
            return len(self.images)
        return len(self.images)

    def __getitem__(self, index):
        idx = index % len(self.images)
        # print("index %d, idx %d" % (index, idx))
        image_patch = self.images[idx]
        image_segment = self.segment[idx]
        image_label = self.labels[idx]
        image_name = self.name[idx]
        image = self.pil_loader(image_patch, isGT=False)
        # print(image_patch)
        if self.split == 'test':
            target = Image.open(image_segment)
        else:
            target = process_mask(image_segment)
        # TODO dont delete old version generate box
        # mask = np.array(mask)
        # # instances are encoded as different colors
        # obj_ids = np.unique(mask)
        # # first id is the background, so remove it
        # obj_ids = obj_ids[1:]
        #
        # # split the color-encoded mask into a set
        # # of binary masks
        # masks = mask == obj_ids[:, None, None]
        #
        # # get bounding box coordinates for each mask
        # num_objs = len(obj_ids)
        # boxes = []
        # for i in range(num_objs):
        #     pos = np.where(masks[i])
        #     xmin = np.min(pos[1])
        #     xmax = np.max(pos[1])
        #     ymin = np.min(pos[0])
        #     ymax = np.max(pos[0])
        #     boxes.append([xmin, ymin, xmax, ymax])
        #
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # # there is only one class
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        # masks = torch.as_tensor(masks, dtype=torch.uint8)
        #
        # image_id = torch.tensor([index])
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # # suppose all instances are not crowd
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        #
        # target = {}
        # target["boxes"] = boxes
        # target["labels"] = labels
        # target["masks"] = masks
        # target["image_id"] = image_id
        # target["area"] = area
        # target["iscrowd"] = iscrowd
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return {
            "image_patch": image,
            'image_label': image_label,
            'image_segment': target,
            'image_name': image_name,
        }

    def pil_loader(self, path, isGT=False):
        if isGT:
            img = Image.open(path)
        else:
            img = Image.open(path).convert('RGB')
        return img


class INSTANCETrainDataset_DQN(data.Dataset):
    def __init__(self, total_data, transform=None, radius=1):
        self.transforms = transform
        self.total_data = total_data
        self.radius = radius
        self.split = 'train'
        self.images = [self.total_data[i][0] for i in range(len(self.total_data))]
        self.segment = [self.total_data[i][1] for i in range(len(self.total_data))]
        self.labels = [self.total_data[i][2] for i in range(len(self.total_data))]
        self.name = [self.total_data[i][3] for i in range(len(self.total_data))]

        print("FOR DQN INSTDataLoader data size: " + str(len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_patch = self.images[index]
        image_segment = self.segment[index]
        image_label = self.labels[index]
        image_name = self.name[index]
        image = self.pil_loader(image_patch, isGT=False)
        if self.split == 'test':
            target = Image.open(image_segment)
        else:
            target = process_mask(image_segment)
        # TODO dont delete old version generate box
        # mask = np.array(mask)
        # # instances are encoded as different colors
        # obj_ids = np.unique(mask)
        # # first id is the background, so remove it
        # obj_ids = obj_ids[1:]
        #
        # # split the color-encoded mask into a set
        # # of binary masks
        # masks = mask == obj_ids[:, None, None]
        #
        # # get bounding box coordinates for each mask
        # num_objs = len(obj_ids)
        # boxes = []
        # for i in range(num_objs):
        #     pos = np.where(masks[i])
        #     xmin = np.min(pos[1])
        #     xmax = np.max(pos[1])
        #     ymin = np.min(pos[0])
        #     ymax = np.max(pos[0])
        #     boxes.append([xmin, ymin, xmax, ymax])
        #
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # # there is only one class
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        # masks = torch.as_tensor(masks, dtype=torch.uint8)
        #
        # image_id = torch.tensor([index])
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # # suppose all instances are not crowd
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        #
        # target = {}
        # target["boxes"] = boxes
        # target["labels"] = labels
        # target["masks"] = masks
        # target["image_id"] = image_id
        # target["area"] = area
        # target["iscrowd"] = iscrowd
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return {
            "image_patch": image,
            'image_label': image_label,
            'image_segment': target,
            'image_name': image_name,
        }

    def pil_loader(self, path, isGT=False):
        if isGT:
            img = Image.open(path)
        else:
            img = Image.open(path).convert('RGB')
        return img


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from active_data.crop_transform import argumentation_train, argumentation_val_normal
    from torchvision.utils import make_grid, save_image
    from driver import std, mean


    def read_data(dataloader):
        while True:
            for batch in dataloader:
                image = batch['image_patch']
                label = batch['image_label']
                segment = batch['image_segment']
                yield {
                    'image_patch': image,
                    'image_label': label,
                    'image_segment': segment,
                }


    def merge_batch(batch):
        image_patch = [torch.unsqueeze(image, dim=0) for inst in batch for image in inst["image_patch"]]
        image_label = [inst["image_label"] for inst in batch for _ in inst["image_patch"]]
        image_segment = [torch.unsqueeze(image, dim=0) for inst in batch for image in inst["image_segment"]]
        image_patch = torch.cat(image_patch, dim=0)
        image_label = torch.tensor(image_label)
        image_segment = torch.cat(image_segment, dim=0)
        image_name = [inst["image_name"] for inst in batch for _ in inst["image_patch"]]

        return {"image_patch": image_patch,
                "image_label": image_label,
                "image_segment": image_segment,
                "image_name": image_name
                }


    data_sets, inital_split = load_train_vali_data(data_name=MONUSEG_DATA)
    test_dataset = INSTANCEDataset(data_sets, split='test', labelled_index=None,
                                   transform=argumentation_val_normal(output_size=512))
    test_loader_mtl_arl = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=merge_batch)
    data_iter = read_data(test_loader_mtl_arl)
    for i in range(10):
        batch_gen = next(data_iter)
        image_mtl = batch_gen['image_patch']
        label_mtl = batch_gen['image_label']
        seg_mtl = batch_gen['image_segment']
        grid = make_grid(image_mtl, nrow=4, padding=2)
        visualize(np.clip((np.transpose(grid.detach().cpu().numpy(), (1, 2, 0)) * std + mean), 0, 1),
                  join('./tmp', str(i) + "_images"))
        grid = make_grid(seg_mtl, nrow=4, padding=2)
        visualize(np.transpose(grid.detach().cpu().numpy() / 2, (1, 2, 0)),
                  join('./tmp', str(i) + "_gt"))
