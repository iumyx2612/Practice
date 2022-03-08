import torch
from torch.utils.data import Dataset, DataLoader

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET

from utils.draw import draw_boxes_with_label, draw_one_box
from utils.boxes import xywh_to_xyxy, scale_box, xyxy_to_xywh, normalize_box


IMAGE_EXT = [".jpg", ".jpeg", ".png"]

def inspect_dataset(root, image_dir, label_dir, ext='.txt', no_label=False):
    image_dir = os.path.join(root, image_dir)
    label_dir = os.path.join(root, label_dir)

    label_dict = {}
    label_counter = 0

    for image_file in os.listdir(image_dir):
        for image_ext in IMAGE_EXT:
            if image_ext in image_file:
                image = cv2.imread(os.path.join(image_dir, image_file))
                image_name = image_file[:image_file.find(image_ext)]
                for label_file in os.listdir(label_dir):
                    if image_name in label_file and ext in label_file:
                        if ext == '.txt': # YOLO
                            data = np.loadtxt(os.path.join(label_dir, label_file))
                            if data.ndim == 1:
                                data = np.expand_dims(data, axis=0)
                            if data.size:
                                labels = data[:, 0]
                                bboxes = data[:, 1:]
                                xywhs = scale_box(image,
                                                  bboxes)  # Update 27/10: change to scale_box instead of scale_xywh
                                xyxys = xywh_to_xyxy(xywhs)
                                image = draw_boxes_with_label(image, xyxys, labels, no_label=no_label)
                        if ext == '.xml':
                            tree = ET.parse(os.path.join(label_dir, label_file))
                            root = tree.getroot()
                            for member in root.findall('object'):
                                # bbox contains 4 coordinate of format [xmin, ymin, xmax, ymax]
                                bbox = member.find("bndbox")

                                # if object is None, ignore
                                if member.find("name") is None:
                                    continue

                                name = member.find("name").text
                                if name not in label_dict.keys():
                                    label_dict[name] = label_counter
                                    label_counter += 1
                                label = label_dict[name]
                                xmin = int(float(bbox.find('xmin').text))
                                ymin = int(float(bbox.find('ymin').text))
                                xmax = int(float(bbox.find('xmax').text))
                                ymax = int(float(bbox.find('ymax').text))
                                xyxy = np.asarray([xmin, ymin, xmax, ymax])
                                image = draw_one_box(image, xyxy, label, no_label=no_label)
                        plt.imshow(image[:, :, ::-1])
                        plt.title(image_file)
                        plt.show()
                        break


def convert_YOLO(root, label_folder, image_folder, dst):
    label_dict = {}
    label_counter = 0
    for label_file in tqdm(os.listdir(os.path.join(root, label_folder))):
        file_name = label_file[:label_file.find('.xml')]
        for image_file in os.listdir(os.path.join(root, image_folder)):
            if file_name in image_file:
                image_path = os.path.join(root, image_folder, image_file)
                image = cv2.imread(image_path)

                xyxys = []
                labels = []
                tree = ET.parse(os.path.join(os.path.join(root, label_folder), label_file))
                _root = tree.getroot()
                for member in _root.findall('object'):
                    # bbox contains 4 coordinate of format [xmin, ymin, xmax, ymax]
                    bbox = member.find("bndbox")

                    # if object is None, ignore
                    if member.find("name") is None:
                        continue

                    name = member.find("name").text
                    if name not in label_dict.keys():
                        label_dict[name] = label_counter
                        label_counter += 1
                    label = label_dict[name]
                    xmin = int(float(bbox.find('xmin').text))
                    ymin = int(float(bbox.find('ymin').text))
                    xmax = int(float(bbox.find('xmax').text))
                    ymax = int(float(bbox.find('ymax').text))
                    xyxy = np.asarray([xmin, ymin, xmax, ymax])
                    xyxys.append(xyxy)
                    labels.append(label)
                xyxys = np.asarray(xyxys)
                xywhs = xyxy_to_xywh(xyxys)
                xywhs_n = normalize_box(image, xywhs)
                lines = ""
                for i in range(len(labels)):
                    l = labels[i]
                    t = ""
                    for j in range(xywhs_n.shape[1]):
                        t += f"{xywhs_n[i][j]} "
                    lines += f"{l} {t} \n"
                with open(f"{root}/{dst}/{file_name}.txt", 'w') as f:
                    f.write(f"{lines}")


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


class LoadImagesAndLabels(Dataset):
    def __init__(self, root, C, S=7, B=2, image_size=448, augment=False, augment_cfg=None):
        self.root = root
        self.C = C
        self.S = S
        self.B = B
        self.image_size = image_size
        self.augment = augment
        self.augment_cfg = augment_cfg
        self.image_files = []

        # get image and label paths
        for image_file in os.listdir(os.path.join(root, "images")):
            self.image_files.append(os.path.join(root, "images", image_file))
        self.label_files = img2label_paths(self.image_files)

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, index):
        label_path = self.label_files[index]
        labels = []
        with open(label_path) as f:
            for label in f.readlines():
                c, x, y, w, h = [float(a) for a in label.replace("\n", '').split()]
                labels.append([c, x, y, w, h])
        image_path = self.image_files[index]
        image = cv2.imread(image_path)[:, :, ::-1] #BGR -> RGB

        # convert to cell
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for label in labels:
            c, x, y, w, h = label

            # find which cell has object
            cell_x, cell_y = int(self.S * x), int(self.S * y)

            # cal x, y, w, h respective to cell
            x_coord, y_coord = self.S * x - cell_x, self.S * y - cell_y
            w_cell, h_cell = w * self.S, h * self.S

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[cell_x, cell_y, self.C] == 0:
                # Set that there exists an object
                label_matrix[cell_x, cell_y, self.C] = 1

                # box coords
                label_matrix[cell_x, cell_y, (self.C + 1):(self.C + 5)] = torch.tensor([x_coord, y_coord, w_cell, h_cell])

                # one hot encoding for class label
                label_matrix[cell_x, cell_y, int(c)] = 1

        return image, label_matrix


if __name__ == '__main__':
    inspect_dataset("../Dataset", "Asirra_ cat vs dogs", "Asirra_ cat vs dogs", ext='.xml')