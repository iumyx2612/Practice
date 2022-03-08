import cv2
import numpy as np

import torch


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


def draw_boxes(img, xyxys, color=(255, 255, 255), thickness=2):
    """
    Draw boxes of xyxy format on an image
    """
    for xyxy in xyxys:
        if isinstance(xyxy, torch.Tensor):
            xyxy = xyxy.cpu().numpy()
        cv2.rectangle(img, pt1=xyxy[:2], pt2=xyxy[2:], color=color, thickness=thickness)
    return img


def draw_one_box(img, xyxy, label, thickness=2, no_label=False):
    """ Draw one box of xyxy format on an image"""
    if isinstance(xyxy, torch.Tensor):
        xyxy = xyxy.cpu().numpy()
    id = label
    color = colors(id)
    label = '{}{:d}'.format("", id)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness=thickness)
    if not no_label:
        cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[0] + t_size[0] + 3, xyxy[1] + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (xyxy[0], xyxy[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def draw_boxes_with_label(img, xyxys, labels, thickness=2, no_label=False):
    for i, xyxy in enumerate(xyxys):
        if isinstance(xyxy, torch.Tensor):
            xyxy = xyxy.cpu().numpy()
        id = int(labels[i])
        color = colors(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness=thickness)
        if not no_label:
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[0] + t_size[0] + 3, xyxy[1] + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (xyxy[0], xyxy[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

if __name__ == '__main__':
    img = np.random.rand(200, 200, 3)
    xyxys = np.asarray([
        [20, 20, 50, 50]
    ])
    img = draw_boxes(img, xyxys)
    cv2.imshow('a', img)
    cv2.waitKey(0)