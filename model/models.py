import torch.nn as nn
import torch.nn.functional as F
from model.ptv1 import PointTransformerCls, PointTransformerReg


class Classifier(nn.Module):
    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.cls = PointTransformerCls(cfg)

    def forward(self, x):
        x = self.cls(x)
        x = F.log_softmax(x, -1)

        return x


class Regressor(nn.Module):
    def __init__(self, cfg):
        super(Regressor, self).__init__()
        self.feat = PointTransformerReg(cfg)
        self.fc = nn.Linear(256, cfg.output_dim)
        
    def forward(self, x):
        x = self.feat(x)
        x = self.fc(x)
        return x
