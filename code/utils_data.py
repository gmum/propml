import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image
from PIL.ImageFile import ImageFile
from pycocotools.coco import COCO
from torch.utils.data.dataset import Dataset
from torchvision import datasets

ImageFile.LOAD_TRUNCATED_IMAGES = True


class COCO2014_handler(Dataset):
    def __init__(self, X, Y, data_path, transform=None, random_crops=0):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.random_crops = random_crops
        self.data_path = data_path

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')

        if self.random_crops == 0:
            x = self.transform(x)
        else:
            crops = []
            for i in range(self.random_crops):
                crops.append(self.transform(x))
            x = torch.stack(crops)

        y = torch.from_numpy(self.Y[index]).to(torch.float)

        return x, y, index
        # return x, y

    def __len__(self):
        return len(self.X)


class COCO2014_handler_two_augment(Dataset):
    def __init__(self, X, Y, data_path, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.data_path = data_path

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/' + self.X[index]).convert('RGB')

        x = self.transform(x)

        y = torch.from_numpy(self.Y[index]).to(torch.float)

        return x, y, index

    def __len__(self):
        return len(self.X)


def get_COCO2014(train_data_path, test_data_path):
    coco = COCO(train_data_path)
    cat2cat = dict()
    for cat in coco.cats.keys():
        cat2cat[cat] = len(cat2cat)

    img_list = list(coco.imgToAnns.keys())
    # img_list = json.load(open(train_data_path, 'r'))
    names = []
    labels = []

    for idx in img_list:
        item = coco.imgs[idx]
        names.append(item['file_name'])

        tmp_idxs = np.asarray([cat2cat[o['category_id']] for o in coco.imgToAnns[idx]])
        lbl = np.zeros(80)
        lbl[tmp_idxs] = 1
        labels.append(lbl)

    names = np.array(names)
    labels = np.array(labels)

    rand_idxs = np.random.permutation(names.shape[0])
    names = names[rand_idxs]
    labels = labels[rand_idxs]

    train_data = names
    train_labels = labels.astype(np.float)

    coco = COCO(test_data_path)
    img_list = list(coco.imgToAnns.keys())
    names = []
    labels = []

    for idx in img_list:
        item = coco.imgs[idx]
        names.append(item['file_name'])

        tmp_idxs = np.asarray([cat2cat[o['category_id']] for o in coco.imgToAnns[idx]])
        lbl = np.zeros(80)
        lbl[tmp_idxs] = 1
        labels.append(lbl)

    names = np.array(names)
    labels = np.array(labels)

    rand_idxs = np.random.permutation(names.shape[0])
    names = names[rand_idxs]
    labels = labels[rand_idxs]

    test_data = names
    test_labels = labels.astype(np.float)

    return train_data, train_labels, test_data, test_labels


class VOC2007_handler(Dataset):
    def __init__(self, X, Y, data_path, transform=None, random_crops=0):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.random_crops = random_crops
        self.data_path = data_path

    def __getitem__(self, index):
        x = Image.open(self.data_path + '/JPEGImages/' + self.X[index] + '.jpg').convert('RGB')

        if self.random_crops == 0:
            x = self.transform(x)
        else:
            crops = []
            for i in range(self.random_crops):
                crops.append(self.transform(x))
            x = torch.stack(crops)

        y = self.Y[index]

        return x, y, index

    def __len__(self):
        return len(self.X)


class VOC2007_handler_aug(Dataset):
    def __init__(self, X, Y, data_path, transform=None, transform_aug=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.transform_aug = transform_aug
        self.data_path = data_path

    def __getitem__(self, index):
        x_ = Image.open(self.data_path + '/JPEGImages/' + self.X[index] + '.jpg').convert('RGB')
        x_aug = self.transform_aug(x_)
        y = self.Y[index]

        return x_aug, y, index

    def __len__(self):
        return len(self.X)


def __dataset_info(data_path, trainval):
    classes = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
    num_classes = len(classes)
    class_to_ind = dict(zip(classes, range(num_classes)))

    with open(data_path + '/ImageSets/Main/' + trainval + '.txt') as f:
        annotations = f.readlines()

    annotations = [n[:-1] for n in annotations]
    names = []
    labels = []
    for af in annotations:
        if len(af) != 6:
            continue
        filename = os.path.join(data_path, 'Annotations', af)
        tree = ET.parse(filename + '.xml')
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes_cl = np.zeros((num_objs), dtype=np.int32)

        for ix, obj in enumerate(objs):
            cls = class_to_ind[obj.find('name').text.lower().strip()]
            boxes_cl[ix] = cls

        lbl = np.zeros(num_classes)
        lbl[boxes_cl] = 1
        labels.append(lbl)
        names.append(af)

    labels = np.array(labels).astype(np.float32)
    labels = labels[:, 1:]

    return np.array(names), np.array(labels)


def get_VOC2007(train_data_path, test_data_path):
    train_data, train_labels = __dataset_info(train_data_path, 'trainval')
    train_idx = np.arange(train_labels.shape[0])
    np.random.shuffle(train_idx)
    train_data, train_labels = train_data[train_idx], train_labels[train_idx]

    test_data, test_labels = __dataset_info(test_data_path, 'test')
    test_idx = np.arange(test_labels.shape[0])
    np.random.shuffle(test_idx)
    test_data, test_labels = test_data[test_idx], test_labels[test_idx]

    return train_data, train_labels, test_data, test_labels


def generate_noisy_labels(labels, noise_rate=0.2):
    N, C = labels.shape

    if isinstance(noise_rate, list):
        if noise_rate[1] == 0:
            noise_rate = noise_rate[0]
        else:
            print('CCMN')
            rand_mat = np.random.rand(N, C)
            mask = np.zeros((N, C), dtype=np.float)
            mask[labels != 1] = rand_mat[labels != 1] < noise_rate[0]
            mask[labels == 1] = rand_mat[labels == 1] < noise_rate[1]
            noisy_labels = np.copy(labels)

            noisy_labels[mask == 1] = 1 - noisy_labels[mask == 1]

    if isinstance(noise_rate, float):
        print('PML')
        alpha_mat = np.ones_like(labels) * noise_rate
        rand_mat = np.random.rand(N, C)

        mask = np.zeros((N, C), dtype=np.float)
        mask[labels != 1] = rand_mat[labels != 1] < alpha_mat[labels != 1]

        noisy_labels = labels.copy()
        noisy_labels[mask == 1] = 1

    return noisy_labels


class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, noise=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)
        self.gen_targets(noise)

    def gen_targets(self, noise):
        coco = self.coco
        targets = []
        for idx in range(len(self)):
            img_id = self.ids[idx]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)
            output = torch.zeros((80), dtype=torch.long)
            for obj in target:
                output[self.cat2cat[obj['category_id']]] = 1
            targets.append(output)
        self.targets = torch.stack(targets, dim=0)
        if noise is not None:
            self.true_targets = self.targets
            self.targets = torch.from_numpy(generate_noisy_labels(self.targets.numpy(), noise))
            print((self.true_targets == self.targets).sum() / self.targets.shape[0] / self.targets.shape[1])
        print(self.targets.shape)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        target = self.targets[index]

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

