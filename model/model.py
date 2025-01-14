import torch.nn as nn
from model.dgcnn import DGCNN


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.model = DGCNN(config='cls', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


class PlaneRegressor(nn.Module):
    def __init__(self):
        super(PlaneRegressor, self).__init__()
        self.model = DGCNN(config='reg_pla')
        
    def forward(self, x):
        return self.model(x)


class CylinderRegressor(nn.Module):
    def __init__(self):
        super(CylinderRegressor, self).__init__()
        self.model = DGCNN(config='reg_cyl')

    def forward(self, x):
        return self.model(x)


class SphereRegressor(nn.Module):
    def __init__(self):
        super(SphereRegressor, self).__init__()
        self.model = DGCNN(config='reg_sph')

    def forward(self, x):
        return self.model(x)


class ConeRegressor(nn.Module):
    def __init__(self):
        super(ConeRegressor, self).__init__()
        self.model = DGCNN(config='reg_con')

    def forward(self, x):
        return self.model(x)
    

class TorusRegressor(nn.Module):
    def __init__(self):
        super(TorusRegressor, self).__init__()
        self.model = DGCNN(config='reg_tor')

    def forward(self, x):
        return self.model(x)
