import torch.nn as nn
import torch.nn.functional as F
from model.dgcnn import DGCNNEmbedding
from model.fcnn import MinkowskiFCNN


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.embedding = DGCNNEmbedding()
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        x = F.log_softmax(x, -1)

        return x


class PlaneRegressor(nn.Module):
    def __init__(self):
        super(PlaneRegressor, self).__init__()
        self.embedding = MinkowskiFCNN(in_channel=3, out_channel=3)
        
    def forward(self, x):
        return self.embedding(x)


class CylinderRegressor(nn.Module):
    def __init__(self):
        super(CylinderRegressor, self).__init__()
        self.embedding = MinkowskiFCNN(in_channel=3, out_channel=7)

    def forward(self, x):
        return self.embedding(x)


class SphereRegressor(nn.Module):
    def __init__(self):
        super(SphereRegressor, self).__init__()
        self.embedding = MinkowskiFCNN(in_channel=3, out_channel=4)

    def forward(self, x):
        return self.embedding(x)


class ConeRegressor(nn.Module):
    def __init__(self):
        super(ConeRegressor, self).__init__()
        self.embedding = MinkowskiFCNN(in_channel=3, out_channel=7)

    def forward(self, x):
        return self.embedding(x)

class TorusRegressor(nn.Module):
    def __init__(self):
        super(TorusRegressor, self).__init__()
        self.embedding = MinkowskiFCNN(in_channel=3, out_channel=8)

    def forward(self, x):
        return self.embedding(x)
