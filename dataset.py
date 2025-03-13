import numpy as np
import torch
import torch.utils.data as data
import os
import glob
from numpy import linalg as LA
import math
import random
import re

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
    def __init__(self, root, npoints=2048, split='train', num_classes=5):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.filepaths = sorted(glob.glob(self.root+'/pointCloud/*.txt'))
        # input_files = glob.glob(self.root+'/pointCloud/pointCloud*.txt')
        
        # def extract_number(f):
        #     s = re.search(r'(\d+)', os.path.basename(f))
        #     return int(s[0]) if s else 0
        # self.filepaths = sorted(input_files, key=extract_number)

        self.filesplit = []

        self.objectClass = dict()
        
        for i in range(num_classes):
            self.objectClass[i] = []
            
        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            gt = np.loadtxt(gtfile)
            cl = gt[0]
            self.objectClass[int(cl)-1].append(filename)

        self.classes = []

        if self.split == 'train':
            for i in range(num_classes):
                self.filesplit.extend(self.objectClass[i][:7360])
                self.classes.extend([i for j in range(7360)])
        elif self.split == 'val':
            for i in range(num_classes):
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
        
        return self.classes[idx], norm_points

# Dataset class for the plane regression
class DatasetPlane(data.Dataset):
    def __init__(self, root, npoints=2048, split='train', num_classes=5):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.filepaths = sorted(glob.glob(self.root+'/pointCloud/*.txt'))
        self.filesplit = []
        

        self.objectClass = dict()
        
        for i in range(num_classes):
            self.objectClass[i] = []
            
        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            gt = np.loadtxt(gtfile)
            cl = gt[0]
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
        
        gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
        with open(gtfile, 'r') as f:
            cl = f.readline()
            n1 = f.readline()
            n2 = f.readline()
            n3 = f.readline()
            p1 = f.readline()
            p2 = f.readline()
            p3 = f.readline()
        
        normal = np.array([float(n1), float(n2), float(n3)])
        point = np.array([float(p1), float(p2), float(p3)])

        norm_points, center, scale = normalize2(pcd, unit_ball=True)
        point = (point - center) / scale

        save_dict = dict(
            coord=norm_points.astype(np.float32),
            color=np.zeros_like(norm_points).astype(np.uint8),
            feat=np.zeros_like(norm_points).astype(np.float32),
            segment=np.zeros(norm_points.shape[0]).astype(np.int32),
            instance=np.zeros(norm_points.shape[0]).astype(np.int32),
            offset=np.array([norm_points.shape[0]], dtype=np.int32),
            grid_size=0.01,
            params=np.concatenate((normal, point)),
        )
        
        return save_dict

# Dataset class for the cylinder regression
class DatasetCylinder(data.Dataset):
    def __init__(self, root, npoints=2048, split='train', num_classes=5):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.filepaths = [os.path.normpath(fi) for fi in sorted(glob.glob(self.root+'/pointCloud/*.txt'))]
        self.filesplit = []
        
        print(len(self.filepaths))
        self.objectClass = dict()
        
        for i in range(num_classes):
            self.objectClass[i] = []
            
        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            gtfile = os.path.normpath(gtfile)
            
            gt = np.loadtxt(gtfile)
            cl = gt[0]
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

        with open(gtfile, 'r') as f:
            cl = f.readline()
            radius = f.readline()
            n1 = f.readline()
            n2 = f.readline()
            n3 = f.readline()
            c1 = f.readline()
            c2 = f.readline()
            c3 = f.readline()
            
        
        target_normal = np.array([float(n1), float(n2), float(n3)])
        target_point = np.array([float(c1), float(c2), float(c3)])
        radius = np.float(radius)

        norm_points, center, scale = normalize2(pcd, unit_ball=True)
        
        return target_normal, target_point, radius, norm_points, center, scale
    
# Dataset class for the sphere regression
class DatasetSphere(data.Dataset):
    def __init__(self, root, npoints=2048, split='train', transform=True, num_classes=5):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.transform = transform
        self.filepaths = [os.path.normpath(fi) for fi in sorted(glob.glob(self.root+'/pointCloud/*.txt'))]
        self.filesplit = []
        
        print(len(self.filepaths))
        self.objectClass = dict()
        
        for i in range(num_classes):
            self.objectClass[i] = []
            
        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            gtfile = os.path.normpath(gtfile)
            
            gt = np.loadtxt(gtfile)
            cl = gt[0]
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
        
        #Apply a perturbation in rotation
        if self.transform:
            rot_x = get_rotation_x(np.deg2rad(random.uniform(25, 45)))
            rot_y = get_rotation_y(np.deg2rad(random.uniform(25, 45)))
            rot_z = get_rotation_z(np.deg2rad(random.uniform(25, 45)))
            rotation_mat = np.dot(rot_x, rot_y)
            rotation_mat = np.dot(rotation_mat, rot_z)
        
            pcd = add_rotation_to_pcloud(pcd, rotation_mat)
        
        gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
        gtfile = os.path.normpath(gtfile)

        with open(gtfile, 'r') as f:
            cl = f.readline()
            radius = f.readline()
            c1 = f.readline()
            c2 = f.readline()
            c3 = f.readline()
        
        point = np.array([float(c1), float(c2), float(c3)])
        radius = np.float(radius)

        if self.transform:
            rotation_norm = np.transpose(np.linalg.inv(rotation_mat))
        
            normal = np.dot(rotation_norm, normal)
            normal = normal/np.linalg.norm(normal)

        norm_points, centerp, scale = normalize2(pcd, unit_ball=True)
        
        return point, radius, norm_points, centerp, scale

# Dataset class for the cone regression
class DatasetCone(data.Dataset):
    def __init__(self, root, npoints=2048, split='train', transform=True, num_classes=5):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.transform = transform
        self.filepaths = [os.path.normpath(fi) for fi in sorted(glob.glob(self.root+'/pointCloud/*.txt'))]
        self.filesplit = []
        
        print(len(self.filepaths))
        self.objectClass = dict()
        
        for i in range(num_classes):
            self.objectClass[i] = []
            
        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            gtfile = os.path.normpath(gtfile)
            
            gt = np.loadtxt(gtfile)
            cl = gt[0]
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
        
        #Apply a perturbation in rotation
        if self.transform:
            rot_x = get_rotation_x(np.deg2rad(random.uniform(25, 45)))
            rot_y = get_rotation_y(np.deg2rad(random.uniform(25, 45)))
            rot_z = get_rotation_z(np.deg2rad(random.uniform(25, 45)))
            rotation_mat = np.dot(rot_x, rot_y)
            rotation_mat = np.dot(rotation_mat, rot_z)
        
            pcd = add_rotation_to_pcloud(pcd, rotation_mat)
        
        gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
        gtfile = os.path.normpath(gtfile)

        with open(gtfile, 'r') as f:
            cl = f.readline()
            aperture = f.readline()
            n1 = f.readline()
            n2 = f.readline()
            n3 = f.readline()
            v1 = f.readline()
            v2 = f.readline()
            v3 = f.readline()
        
        normal = np.array([float(n1), float(n2), float(n3)])
        vertex = np.array([float(v1), float(v2), float(v3)])
        aperture = np.float(aperture)

        if self.transform:
            rotation_norm = np.transpose(np.linalg.inv(rotation_mat))
        
            normal = np.dot(rotation_norm, normal)
            normal = normal/np.linalg.norm(normal)

        norm_points, center, scale = normalize2(pcd, unit_ball=True)
        
        return normal, vertex, aperture, norm_points, center, scale

# Dataset class for the torus regression
class DatasetTorus(data.Dataset):
    def __init__(self, root, npoints=2048, split='train', transform=True, num_classes=5):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.transform = transform
        self.filepaths = [os.path.normpath(fi) for fi in sorted(glob.glob(self.root+'/pointCloud/*.txt'))]
        self.filesplit = []
        
        print(len(self.filepaths))
        self.objectClass = dict()
        
        for i in range(num_classes):
            self.objectClass[i] = []
            
        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            gtfile = os.path.normpath(gtfile)
            
            gt = np.loadtxt(gtfile)
            cl = gt[0]
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
        
        #Apply a perturbation in rotation
        if self.transform:
            rot_x = get_rotation_x(np.deg2rad(random.uniform(25, 45)))
            rot_y = get_rotation_y(np.deg2rad(random.uniform(25, 45)))
            rot_z = get_rotation_z(np.deg2rad(random.uniform(25, 45)))
            rotation_mat = np.dot(rot_x, rot_y)
            rotation_mat = np.dot(rotation_mat, rot_z)
        
            pcd = add_rotation_to_pcloud(pcd, rotation_mat)
        
        gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
        gtfile = os.path.normpath(gtfile)

        with open(gtfile, 'r') as f:
            cl = f.readline()
            major_radius = f.readline()
            minor_radius = f.readline()
            n1 = f.readline()
            n2 = f.readline()
            n3 = f.readline()
            c1 = f.readline()
            c2 = f.readline()
            c3 = f.readline()
        
        normal = np.array([float(n1), float(n2), float(n3)])
        point = np.array([float(c1), float(c2), float(c3)])
        major_radius = np.float(major_radius)
        minor_radius = np.float(minor_radius)

        if self.transform:
            rotation_norm = np.transpose(np.linalg.inv(rotation_mat))
        
            normal = np.dot(rotation_norm, normal)
            normal = normal/np.linalg.norm(normal)

        norm_points, centerp, scale = normalize2(pcd, unit_ball=True)
        
        return normal, point, minor_radius, major_radius, norm_points, centerp, scale

# Dataset class for the cuboid regression
class DatasetCuboid(data.Dataset):
    def __init__(self, root, npoints=2048, split='train', num_classes=5):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.filepaths = [os.path.normpath(fi) for fi in sorted(glob.glob(self.root+'/pointCloud/*.txt'))]
        self.filesplit = []

        self.objectClass = dict()

        for i in range(num_classes):
            self.objectClass[i] = []

        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            gtfile = os.path.normpath(gtfile)
            
            gt = np.loadtxt(gtfile)
            cl = gt[0]
            self.objectClass[int(cl)-1].append(filename)

        self.classes = []

        if self.split == 'train':
            self.filesplit.extend(self.objectClass[5][:7360])
        elif self.split == 'val':
            self.filesplit.extend(self.objectClass[5][7360:])

    def __len__(self):
        return len(self.filesplit)
    
    def __getitem__(self, idx):
        filename = self.filesplit[idx]

        pcd = np.loadtxt(filename, delimiter=',')
        
        if self.npoints != 0:
            pcd = resample_pcd(pcd, self.npoints)
        
        gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
        gtfile = os.path.normpath(gtfile)

        with open(gtfile, 'r') as f:
            cl = f.readline()
            a = f.readline()
            b = f.readline()
            n1 = f.readline()
            n2 = f.readline()
            n3 = f.readline()
            u1 = f.readline()
            u2 = f.readline()
            u3 = f.readline()
            c1 = f.readline()
            c2 = f.readline()
            c3 = f.readline()

        target_axis = np.array([float(n1), float(n2), float(n3)])
        target_uaxis = np.array([float(u1), float(u2), float(u3)])
        point = np.array([float(c1), float(c2), float(c3)])
        a = np.float(a)
        b = np.float(b)

        norm_points, center, scale = normalize2(pcd, unit_ball=True)

        return target_axis, target_uaxis, point, a, b, norm_points, center, scale
    
# Dataset class for the tee regression
class DatasetTee(data.Dataset):
    def __init__(self, root, npoints=2048, split='train', num_classes=5):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.filepaths = [os.path.normpath(fi) for fi in sorted(glob.glob(self.root+'/pointCloud/*.txt'))]
        self.filesplit = []

        self.objectClass = dict()

        for i in range(num_classes):
            self.objectClass[i] = []

        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            gtfile = os.path.normpath(gtfile)
            
            gt = np.loadtxt(gtfile)
            cl = gt[0]
            self.objectClass[int(cl)-1].append(filename)

        self.classes = []

        if self.split == 'train':
            self.filesplit.extend(self.objectClass[6][:7360])
        elif self.split == 'val':
            self.filesplit.extend(self.objectClass[6][7360:])

    def __len__(self):
        return len(self.filesplit)
    
    def __getitem__(self, idx):
        filename = self.filesplit[idx]

        pcd = np.loadtxt(filename, delimiter=',')
        
        if self.npoints != 0:
            pcd = resample_pcd(pcd, self.npoints)
        
        gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
        gtfile = os.path.normpath(gtfile)

        with open(gtfile, 'r') as f:
            cl = f.readline()
            ar = f.readline()
            br = f.readline()
            al = f.readline()
            bl = f.readline()
            c1 = f.readline()
            c2 = f.readline()
            c3 = f.readline()
            u1 = f.readline()
            u2 = f.readline()
            u3 = f.readline()
            v1 = f.readline()
            v2 = f.readline()
            v3 = f.readline()

        target_uaxis = np.array([float(u1), float(u2), float(u3)])
        target_vaxis = np.array([float(v1), float(v2), float(v3)])
        point = np.array([float(c1), float(c2), float(c3)])
        ar = np.float(ar)
        br = np.float(br)
        al = np.float(al)
        bl = np.float(bl)

        norm_points, center, scale = normalize2(pcd, unit_ball=True)

        return target_uaxis, target_vaxis, point, ar, br, al, bl, norm_points, center, scale
    
# Dataset class for the cross regression
class DatasetCross(data.Dataset):
    def __init__(self, root, npoints=2048, split='train', num_classes=5):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.filepaths = [os.path.normpath(fi) for fi in sorted(glob.glob(self.root+'/pointCloud/*.txt'))]
        self.filesplit = []

        self.objectClass = dict()

        for i in range(num_classes):
            self.objectClass[i] = []

        for filename in self.filepaths:
            gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
            gtfile = os.path.normpath(gtfile)
            
            gt = np.loadtxt(gtfile)
            cl = gt[0]
            self.objectClass[int(cl)-1].append(filename)

        self.classes = []

        if self.split == 'train':
            self.filesplit.extend(self.objectClass[7][:7360])
        elif self.split == 'val':
            self.filesplit.extend(self.objectClass[7][7360:])

    def __len__(self):
        return len(self.filesplit)
    
    def __getitem__(self, idx):
        filename = self.filesplit[idx]

        pcd = np.loadtxt(filename, delimiter=',')
        
        if self.npoints != 0:
            pcd = resample_pcd(pcd, self.npoints)
        
        gtfile = self.root + '/GTpointCloud/GT' + filename.split('/')[-1]
        gtfile = os.path.normpath(gtfile)

        with open(gtfile, 'r') as f:
            cl = f.readline()
            ar = f.readline()
            br = f.readline()
            al = f.readline()
            bl = f.readline()
            c1 = f.readline()
            c2 = f.readline()
            c3 = f.readline()
            u1 = f.readline()
            u2 = f.readline()
            u3 = f.readline()
            v1 = f.readline()
            v2 = f.readline()
            v3 = f.readline()

        target_uaxis = np.array([float(u1), float(u2), float(u3)])
        target_vaxis = np.array([float(v1), float(v2), float(v3)])
        point = np.array([float(c1), float(c2), float(c3)])
        ar = np.float(ar)
        br = np.float(br)
        al = np.float(al)
        bl = np.float(bl)

        norm_points, center, scale = normalize2(pcd, unit_ball=True)

        return target_uaxis, target_vaxis, point, ar, br, al, bl, norm_points, center, scale