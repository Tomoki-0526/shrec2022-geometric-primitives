import torch.nn as nn
from model.dgcnn import DGCNNEmbedding
import torch.nn.functional as F


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
        self.embedding = DGCNNEmbedding()
        self.fc1 = nn.Linear(256, 3)    # normal
        self.fc2 = nn.Linear(256, 3)    # xyz
        
    def forward(self, x):
        x = self.embedding(x)
        normal = self.fc1(x)
        xyz = self.fc2(x)

        return normal, xyz


class CylinderRegressor(nn.Module):
    def __init__(self):
        super(CylinderRegressor, self).__init__()
        self.embedding = DGCNNEmbedding()
        self.fc1 = nn.Linear(256, 3)    # normal
        self.fc2 = nn.Linear(256, 3)    # center
        self.fc3 = nn.Linear(256, 1)    # radius

    def forward(self, x):
        x =  self.embedding(x)
        normal = self.fc1(x)
        center = self.fc2(x)
        radius = self.fc3(x)

        return normal, center, radius


class SphereRegressor(nn.Module):
    def __init__(self):
        super(SphereRegressor, self).__init__()
        self.embedding = DGCNNEmbedding()
        self.fc1 = nn.Linear(256, 3)    # center
        self.fc2 = nn.Linear(256, 1)    # radius

    def forward(self, x):
        x = self.embedding(x)
        center = self.fc1(x)
        radius = self.fc2(x)

        return center, radius


class ConeRegressor(nn.Module):
    def __init__(self):
        super(ConeRegressor, self).__init__()
        self.embedding = DGCNNEmbedding()
        self.fc1 = nn.Linear(256, 3)    # normal
        self.fc2 = nn.Linear(256, 1)    # aperture
        self.fc3 = nn.Linear(256, 3)    # vertex

    def forward(self, x):
        x = self.embedding(x)
        normal = self.fc1(x)
        aperture = self.fc2(x)
        vertex = self.fc3(x)

        return normal, vertex, aperture

class TorusRegressor(nn.Module):
    def __init__(self):
        super(TorusRegressor, self).__init__()
        self.embedding = DGCNNEmbedding()
        self.fc1 = nn.Linear(256, 3)    # normal
        self.fc2 = nn.Linear(256, 3)    # center
        self.fc3 = nn.Linear(256, 1)    # minR
        self.fc4 = nn.Linear(256, 1)    # maxR

    def forward(self, x):
        x = self.embedding(x)
        normal = self.fc1(x)
        center = self.fc2(x)
        minR = self.fc3(x)
        maxR = self.fc4(x)

        return normal, center, minR, maxR
