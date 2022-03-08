import torch
from torch import nn

import yaml
import numpy as np

from YOLO.models.common import *


class YOLOv1(nn.Module):
    def __init__(self, cfg, nc, s_grid, num_boxes, input_channels=3):
        super(YOLOv1, self).__init__()
        if isinstance(cfg, dict):
            self.cfg = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.cfg = yaml.safe_load(f)  # model dict
        self.darknet = self._parse_model(self.cfg, input_channels)
        self.fc = self._build_fc(nc, s_grid, num_boxes)

    def forward(self, x):
        x = self.darknet(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)

    def extract_features(self, x):
        x = self.darknet(x)
        return x

    def _parse_model(self, cfg, input_channels):
        layers, in_ch = [], [input_channels]
        for i, (repeats, module, args) in enumerate(cfg["backbone"]):
            module = eval(module) if isinstance(module, str) else module  # eval strings
            for j, a in enumerate(args):
                try:
                    args[j] = eval(a) if isinstance(a, str) else a  # eval strings
                except NameError:
                    pass
            if module in [ConvBlock]:
                in_c, out_ch = in_ch[args[0]], args[1]
                args = [in_c, *args[1:]]
                in_ch.append(out_ch)
            if module in [nn.MaxPool2d]:
                args = args
            m = nn.Sequential(*(module(*args) for _ in range(repeats))) if repeats > 1 else module(*args) # build layer
            layers.append(m)
        return nn.Sequential(*layers)

    def _build_fc(self, nc, s_grid, num_boxes):
        S, B, C = s_grid, num_boxes, nc
        return nn.Sequential(
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5))
        )


if __name__ == '__main__':
    net = YOLOv1("darknet.yaml", nc=2, s_grid=7, num_boxes=2)
    image = torch.randn((2, 3, 448, 448))
    print(net.fc)
    x = net.extract_features(image)
    y = net(image)
    print(x.shape)
    print(y.shape)