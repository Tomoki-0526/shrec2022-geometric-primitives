import numpy as np
import torch
import torch.utils.data as data
import os
import glob
from numpy import linalg as LA
import random
import math
import einops
import transforms as t

#####################################################################
# M4 methods
train_transforms = [t.KeepInitialPoints(),
                    t.Translate(), 
                    t.SphereNormalization(), 
                    t.Initialization(),
                    t.RandomRotate(180, 0),
                    t.RandomRotate(180, 1),
                    t.RandomRotate(180, 2),
                    t.GaussianNoise(),
                    t.GetMean()]

valid_transforms = [t.KeepInitialPoints(),
                    t.Translate(), 
                    t.SphereNormalization(),
                    t.GetMean()]

def norm(x):
    return (x * x).sum(-1).sqrt()

def parse_point_cloud(fname):

    file = open(fname)
    points = []

    for line in file.readlines():
        pts = torch.Tensor(list(map(float, line.split(","))))
        points.append(pts)

    return einops.rearrange(points, "n d -> n d")

def parse_plane(lines):
    
    normal = torch.Tensor(list(map(float, lines[1:4])))
    assert norm(normal) > 0.9999

    vertex = torch.Tensor(list(map(float, lines[4:])))
    data = torch.Tensor([0] + list(map(float, lines[1:])) + [-1, -1])

    return {"type": "plane", "class": 0, "vertex": vertex, "normal":normal, "data": data}

def parse_cylinder(lines):
    
    radius = float(lines[1])
    axis = torch.Tensor(list(map(float, lines[2:5])))
    assert norm(axis) > 0.9999
    vertex = torch.Tensor(list(map(float, lines[5:])))
    data = torch.Tensor([1] + list(map(float, lines[1:])) + [-1])

    return {"type": "cylinder", "class": 1, "radius": radius, "axis": axis, "vertex": vertex, "data": data}

def parse_sphere(lines):
    
    radius = float(lines[1])
    center = torch.Tensor(list(map(float, lines[2:])))
    data = torch.Tensor([2] + list(map(float, lines[1:])) + [-1]*4)

    return {"type": "sphere", "class": 2, "radius": radius, "center": center, "data": data}

def parse_cone(lines):
    
    angle = float(lines[1])
    axis = torch.Tensor(list(map(float, lines[2:5])))
    assert norm(axis) > 0.9999
    vertex = torch.Tensor(list(map(float, lines[5:])))
    data = torch.Tensor([3] + list(map(float, lines[1:])) + [-1])

    return {"type": "cone", "class": 3, "angle": angle, "axis": axis, "vertex": vertex, "data": data}

def parse_torus(lines):
    
    major_radius = float(lines[1])
    minor_radius = float(lines[2])
    axis = torch.Tensor(list(map(float, lines[3:6])))
    assert norm(axis) > 0.9999
    center = torch.Tensor(list(map(float, lines[6:])))
    data = torch.Tensor([4] + list(map(float, lines[1:])))

    return {"type": "torus", "class": 4, "major_radius": major_radius, "minor_radius": minor_radius, "axis": axis, "center": center, "data": data}

def parse_label(fname):
    
    file = open(fname)
    
    #assigning a distinct function to handle each type of primitive
    handlers ={
                "1": parse_plane,
                "2": parse_cylinder,
                "3": parse_sphere,
                "4": parse_cone,
                "5": parse_torus
                }
    
    #parsing the contents of the file. The first character corresponds to a specific type of primitive
    contents =  file.readlines()
    
    #handling the primitive and returning the label
    return handlers[contents[0][0]](contents)
#####################################################################

def get_rotation_x(teta):
    return np.array([
        np.array([1,    0,                0]),
        np.array([0,    math.cos(teta),   -math.sin(teta)]),
        np.array([0,    math.sin(teta),    math.cos(teta)])
    ])

def get_rotation_y(teta):
    return np.array([
        np.array([math.cos(teta),  0,       math.sin(teta)]),
        np.array([0,               1,       0]),
        np.array([-math.sin(teta), 0,       math.cos(teta)])
    ])


def get_rotation_z(teta):
    return np.array([
        np.array([math.cos(teta),  -math.sin(teta),       0]),
        np.array([math.sin(teta),  math.cos(teta),        0]),
        np.array([0,               0,                     1])
    ])


def add_rotation_to_pcloud(pcloud, r_rotation):
    # r_rotation = rand_rotation_matrix()
    if len(pcloud.shape) == 2:
        return pcloud.dot(r_rotation)
    else:
        return np.asarray([e.dot(r_rotation) for e in pcloud])

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

def origin_mass_center(pcd):
    expectation = np.mean(pcd, axis = 0)
    centered_pcd = pcd - expectation
    return centered_pcd

def origin_mass_center2(pcd):
    expectation = np.mean(pcd, axis = 0)
    centered_pcd = pcd - expectation
    return centered_pcd, expectation

def normalize(points, unit_ball = False):
    normalized_points = origin_mass_center(points)
    #normalized_points = points
    l2_norm = LA.norm(normalized_points,axis=1)
    max_distance = max(l2_norm)

    if unit_ball:
        scale = max_distance
        normalized_points = normalized_points/(max_distance)
    else:
        scale = 2 * max_distance
        normalized_points = normalized_points/(2 * max_distance)

    return normalized_points, scale
    
def normalize2(points, unit_ball = False):
    normalized_points, center = origin_mass_center2(points)
    #normalized_points = points
    l2_norm = LA.norm(normalized_points,axis=1)
    max_distance = max(l2_norm)

    if unit_ball:
        scale = max_distance
        normalized_points = normalized_points/(max_distance)
    else:
        scale = 2 * max_distance
        normalized_points = normalized_points/(2 * max_distance)

    return normalized_points, center, scale

# Dataset class for the classification problem    
class DatasetSHREC2022(data.Dataset):
    def __init__(self, root, npoints=2048, split='train'):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.filepaths = sorted(glob.glob(self.root+'/pointCloud/*.txt'))
        self.filesplit = []
        

        self.objectClass = dict()
        
        for i in range(5):
            self.objectClass[i] = []
            
        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            with open(gtfile, 'r') as f:
                cl = f.readline()
            self.objectClass[int(cl)-1].append(filename)

        self.classes = []

        if self.split == 'train':
            for i in range(5):
                self.filesplit.extend(self.objectClass[i][:7360])
                self.classes.extend([i for j in range(7360)])
        elif self.split == 'val':
            for i in range(5):
                self.filesplit.extend(self.objectClass[i][7360:])
                self.classes.extend([i for j in range(1840)])

    def __len__(self):
        return len(self.filesplit)
    
    def __getitem__(self, idx):
        filename = self.filesplit[idx]

        pcd = np.loadtxt(filename, delimiter=',')
        
        if self.npoints != 0:
            pcd = resample_pcd(pcd, self.npoints)
        
        norm_points, center, scale = normalize2(pcd, unit_ball=True)
        
        return self.classes[idx],norm_points

#Dataset class for the plane regression
class DatasetPlane(data.Dataset):
    def __init__(self, root, npoints=2048, split='train', transform=[]):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.filepaths = sorted(glob.glob(self.root+'/pointCloud/*.txt'))
        self.filesplit = []
        self.transform = transform
        

        self.objectClass = dict()
        
        for i in range(5):
            self.objectClass[i] = []
            
        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            with open(gtfile, 'r') as f:
                cl = f.readline()
            self.objectClass[int(cl)-1].append(filename)

        self.classes = []

        if self.split == 'train':
            self.filesplit.extend(self.objectClass[0][:7360])
        elif self.split == 'val':
            self.filesplit.extend(self.objectClass[0][7360:])
            
    def __len__(self):
        return len(self.filesplit)
    
    def __getitem__(self, idx):
        filename = self.filesplit[idx]

        pcd = np.loadtxt(filename, delimiter=',')
        
        if self.npoints != 0:
            pcd = resample_pcd(pcd, self.npoints)
        pcd = torch.from_numpy(pcd).float()
        
        gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
        label = parse_label(gtfile)

        data = {'x': pcd, 'y': label['data'], 'index': idx+1}
        
        for t in self.transform:
            data = t(data)

        return data

#Dataset class for the cylinder regression
class DatasetCylinder(data.Dataset):
    def __init__(self, root, npoints=2048, split='train', transform=[]):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.filepaths = [os.path.normpath(fi) for fi in sorted(glob.glob(self.root+'/pointCloud/*.txt'))]
        self.filesplit = []
        self.transform = transform
        
        print(len(self.filepaths))
        self.objectClass = dict()
        
        for i in range(5):
            self.objectClass[i] = []
            
        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            gtfile = os.path.normpath(gtfile)
            
            with open(gtfile, 'r') as f:
                cl = f.readline()
            self.objectClass[int(cl)-1].append(filename)

        self.classes = []

        if self.split == 'train':
            self.filesplit.extend(self.objectClass[1][:7360])
        elif self.split == 'val':
            self.filesplit.extend(self.objectClass[1][7360:])
            
    def __len__(self):
        return len(self.filesplit)
    
    def __getitem__(self, idx):
        filename = self.filesplit[idx]

        pcd = np.loadtxt(filename, delimiter=',')
        
        if self.npoints != 0:
            pcd = resample_pcd(pcd, self.npoints)
        
        gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
        gtfile = os.path.normpath(gtfile)
        label = parse_label(gtfile) if self.split == 'train' else {'data': None}

        data = {'x': pcd, 'y': label['data'], 'index': idx+1}
        
        for t in self.transform:
            data = t(data)

        return data

#Dataset class for the cone regression
class DatasetCone(data.Dataset):
    def __init__(self, root, npoints=2048, split='train', transform=[]):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.transform = transform
        self.filepaths = [os.path.normpath(fi) for fi in sorted(glob.glob(self.root+'/pointCloud/*.txt'))]
        self.filesplit = []
        
        print(len(self.filepaths))
        self.objectClass = dict()
        
        for i in range(5):
            self.objectClass[i] = []
            
        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            gtfile = os.path.normpath(gtfile)
            
            with open(gtfile, 'r') as f:
                cl = f.readline()
            self.objectClass[int(cl)-1].append(filename)

        self.classes = []

        if self.split == 'train':
            self.filesplit.extend(self.objectClass[3][:7360])
        elif self.split == 'val':
            self.filesplit.extend(self.objectClass[3][7360:])
            
    def __len__(self):
        return len(self.filesplit)
    
    def __getitem__(self, idx):
        filename = self.filesplit[idx]

        pcd = np.loadtxt(filename, delimiter=',')
        
        if self.npoints != 0:
            pcd = resample_pcd(pcd, self.npoints)
        
        gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
        gtfile = os.path.normpath(gtfile)
        label = parse_label(gtfile) if self.split == 'train' else {'data': None}

        data = {'x': pcd, 'y': label['data'], 'index': idx+1}
        
        for t in self.transform:
            data = t(data)

        return data

#Dataset class for the sphere regression
class DatasetSphere(data.Dataset):
    def __init__(self, root, npoints=2048, split='train', transform=[]):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.transform = transform
        self.filepaths = [os.path.normpath(fi) for fi in sorted(glob.glob(self.root+'/pointCloud/*.txt'))]
        self.filesplit = []
        
        print(len(self.filepaths))
        self.objectClass = dict()
        
        for i in range(5):
            self.objectClass[i] = []
            
        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            gtfile = os.path.normpath(gtfile)
            
            with open(gtfile, 'r') as f:
                cl = f.readline()
            self.objectClass[int(cl)-1].append(filename)

        self.classes = []

        if self.split == 'train':
            self.filesplit.extend(self.objectClass[2][:7360])
        elif self.split == 'val':
            self.filesplit.extend(self.objectClass[2][7360:])
            
    def __len__(self):
        return len(self.filesplit)
    
    def __getitem__(self, idx):
        filename = self.filesplit[idx]

        pcd = np.loadtxt(filename, delimiter=',')
        
        if self.npoints != 0:
            pcd = resample_pcd(pcd, self.npoints)
        
        gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
        gtfile = os.path.normpath(gtfile)
        label = parse_label(gtfile) if self.split == 'train' else {'data': None}

        data = {'x': pcd, 'y': label['data'], 'index': idx+1}
        
        for t in self.transform:
            data = t(data)

        return data

#Dataset class for the torus regression
class DatasetTorus(data.Dataset):
    def __init__(self, root, npoints=2048, split='train', transform=[]):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.transform = transform
        self.filepaths = [os.path.normpath(fi) for fi in sorted(glob.glob(self.root+'/pointCloud/*.txt'))]
        self.filesplit = []
        
        print(len(self.filepaths))
        self.objectClass = dict()
        
        for i in range(5):
            self.objectClass[i] = []
            
        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            gtfile = os.path.normpath(gtfile)
            
            with open(gtfile, 'r') as f:
                cl = f.readline()
            self.objectClass[int(cl)-1].append(filename)

        self.classes = []

        if self.split == 'train':
            self.filesplit.extend(self.objectClass[4][:7360])
        elif self.split == 'val':
            self.filesplit.extend(self.objectClass[4][7360:])
            
    def __len__(self):
        return len(self.filesplit)
    
    def __getitem__(self, idx):
        filename = self.filesplit[idx]

        pcd = np.loadtxt(filename, delimiter=',')
        
        if self.npoints != 0:
            pcd = resample_pcd(pcd, self.npoints)
        
        gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
        gtfile = os.path.normpath(gtfile)

        label = parse_label(gtfile) if self.split == 'train' else {'data': None}

        data = {'x': pcd, 'y': label['data'], 'index': idx+1}
        
        for t in self.transform:
            data = t(data)

        return data
