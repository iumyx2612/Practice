import torch
from torch import nn

from utils.metrics import bbox_iou


class Loss(nn.Module):
    def __init__(self, C, B=2, S=7, lcoord=5, lnoobj=0.5):
        super(Loss, self).__init__()
        self.C = C
        self.B = B
        self.S = S
        self.lcoord = lcoord
        self.lnoobj = lnoobj
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, predictions, targets):
        # predictions: (batch, S * S * (C+5*B)) -> ((batch, S, S, (C+5*B))
        predictions = predictions.reshape(-1, self.S, self.S, self.C + 5 * self.B)




