import io
import os
import torch
from math import floor

import numpy as np
from torch.utils.data import Dataset
from skimage import io
from skimage.transform import resize
from sklearn.cluster import KMeans
import pandas as pd


# Grid size 13x13
# IN 416x416x3
# OUT 64X64x5(p, bx, by, bw, bh)
class YoloDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, num_of_classes: int = 1, transform=None):
        self.boxes = pd.read_csv(csv_file, names=["filename", "top", "right", "bottom", "left"])
        self.root_dir = root_dir
        self.transform = transform
        # self.boxes_kmeans = self._get_boxes_kmeans(num_of_classes)
        self.anchors = 1
        self.IN = 416
        self.FILTER_SIZE = 13
        assert self.IN % self.FILTER_SIZE == 0
        self.OUT = int(self.IN / self.FILTER_SIZE)
        self.OUT_CHANNELS = 5

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, idx):
        box = self.boxes.iloc[idx].values
        img = io.imread(os.path.join(self.root_dir, box[0]))
        in_size = img.shape
        try:
            if in_size[2] == 4:
                img = img[..., :3]  # reduce transparent channel
                in_size = img.shape
        except IndexError:
            print(f"Remove from csv white/black image {box[0]}")
        img = resize(img, (self.IN, self.IN))
        box_rescaled = self._rescale_boxes(box[1:], in_size)
        y = self._get_y_to_img(box_rescaled)
        sample = {"x": img, 'y': y}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _get_y_to_img(self, box) -> np.ndarray:
        y = np.zeros((self.OUT, self.OUT, self.OUT_CHANNELS))
        box_middle_point = (box[2] + box[0]) / 2, (box[3] + box[1]) / 2  # y,x
        height = (box[2] - box[0]) / self.FILTER_SIZE
        width = (box[1] - box[3]) / self.FILTER_SIZE
        y[floor(box_middle_point[0] / self.FILTER_SIZE), floor(box_middle_point[1] / self.FILTER_SIZE)] = [1,  # p
                                                                                                           (
                                                                                                                   box_middle_point[
                                                                                                                       1] % self.FILTER_SIZE) / self.FILTER_SIZE,
                                                                                                           # bx
                                                                                                           (
                                                                                                                   box_middle_point[
                                                                                                                       0] % self.FILTER_SIZE) / self.FILTER_SIZE,
                                                                                                           # by
                                                                                                           height,  # bh
                                                                                                           width]  # bw
        return y

    def _rescale_boxes(self, box, in_size) -> []:
        vertical_ratio = self.IN / in_size[0]
        horizontal_ratio = self.IN / in_size[1]
        new_box = [box[0] * vertical_ratio, box[1] * horizontal_ratio, box[2] * vertical_ratio,
                   box[3] * horizontal_ratio]
        return [round(i, 1) for i in new_box]

    def _get_boxes_kmeans(self, num_of_classes):
        # In this task we will have only one class face so it's redndant
        # classes = [x[1] for x in os.walk(self.root_dir)][0]
        column_names = list(self.boxes)
        class_boxes = self.boxes[column_names[1:]].values
        kmeans = KMeans(n_clusters=num_of_classes).fit(class_boxes)
        return kmeans
        # classes = [x[1] for x in os.walk(self.root_dir)][0]
        # for _class_name in classes:
        #     column_names = list(self.boxes)
        #
        #     class_boxes = self.boxes[self.boxes[column_names[0]].str.startswith(_class_name)][column_names[1:]].values
        #
        #     class_kmean = KMeans(n_clusters=len(this_dataset.classes)).fit(X)
        #     # class_boxes = self.boxes[self.boxes['filename'].str.startswith(_class)]["top"].values
        #     break


class ToTensor:
    def __call__(self, sample):
        img, y = sample['x'], sample['y']
        img = img.transpose((2, 0, 1))
        img /= 255.0
        y = y.transpose((2, 0, 1))
        return {'img': torch.FloatTensor(img),
                'y': torch.FloatTensor(y)}


if __name__ == '__main__':
    yolo_dataset = YoloDataset('./raw_dataset/raw_dataset/faces.csv', './raw_dataset/raw_dataset')