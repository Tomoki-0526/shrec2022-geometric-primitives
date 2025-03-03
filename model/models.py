import torch.nn as nn
import torch.nn.functional as F
from model.dgcnn import DGCNNEmbedding
from model.pointnet import PointNetfeat


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


class PlaneNet(nn.Module):
    def __init__(self):
        super(PlaneNet, self).__init__()
        self.feat = PointNetfeat(global_feat=True, feature_transform=False)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3) #Normal
        self.fc4 = nn.Linear(256, 3) #Point
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        normal = self.fc3(x)
        point = self.fc4(x)
        return normal, point


class CylinderNet(nn.Module):
    def __init__(self):
        super(CylinderNet, self).__init__()
        self.feat = PointNetfeat(global_feat=True, feature_transform=False)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3) #normal
        self.fc4 = nn.Linear(256, 3) #center
        self.fc5 = nn.Linear(256, 1) #radius
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
            
    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        normal = self.fc3(x)
        center = self.fc4(x)
        radius = self.fc5(x)

        return normal, center, radius


class SphereNet(nn.Module):
    def __init__(self):
        super(SphereNet, self).__init__()
        self.feat = PointNetfeat(global_feat=True, feature_transform=False)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3) #center
        self.fc4 = nn.Linear(256, 1)

        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        center = self.fc3(x)
        radius = self.fc4(x)
        return center, radius


class ConeNet(nn.Module):
    def __init__(self):
        super(ConeNet, self).__init__()
        self.feat = PointNetfeat(global_feat=True, feature_transform=False)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3) #Normal
        self.fc4 = nn.Linear(256, 1) #aperture
        self.fc5 = nn.Linear(256, 3) #vertex
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
       
    
    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        normal = self.fc3(x)
        aperture = self.fc4(x)
        vertex = self.fc5(x)
        return normal, vertex, aperture


class TorusNet(nn.Module):
    def __init__(self):
        super(TorusNet, self).__init__()
        self.feat = PointNetfeat(global_feat=True, feature_transform=False)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3) #normal
        self.fc4 = nn.Linear(256, 3) #center
        self.fc5 = nn.Linear(256, 1) #minR
        self.fc6 = nn.Linear(256, 1) #maxR
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        normal = self.fc3(x)
        center = self.fc4(x)
        minR = self.fc5(x)
        maxR = self.fc6(x)
        return normal , center, minR, maxR
