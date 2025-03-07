import numpy as np
import torch
import os
from numpy import linalg as LA
import argparse
from model.models import *
from time import time

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

def origin_mass_center2(pcd):
    expectation = np.mean(pcd, axis = 0)
    centered_pcd = pcd - expectation
    return centered_pcd, expectation

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

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='', help='input file')
parser.add_argument('--outf', type=str, default='', help='output folder')

opt = parser.parse_args()

#generate output filename
filename = opt.file
output_filename = os.path.splitext(os.path.split(filename)[1])[0]
output_filename = output_filename+'_prediction.txt'

t1 = time()
#Read and sample the point cloud
pcd = np.loadtxt(opt.file, delimiter=',')
pcd = resample_pcd(pcd, 2048)

input_pts, center, scale = normalize2(pcd, unit_ball=True)
#print(f'Center:{center}, Scale: {scale}')
input_pts = torch.unsqueeze(torch.from_numpy(input_pts), 0)

classifier = torch.nn.DataParallel(Classifier(num_classes=8))
classifier.load_state_dict(torch.load('/home/szj/SHREC2022/log/ultra/classification/cls_model_249.pth'))
classifier.cuda()

input_pts = input_pts.transpose(2, 1)
input_pts = input_pts.cuda().float()
classifier = classifier.eval()
pred = classifier(input_pts)
pred_choice = pred.detach().max(1)[1].cpu().numpy()[0]

classifier.cpu()

with open(os.path.join(opt.outf, output_filename), 'wt') as f:
    f.write(str(pred_choice+1)+'\n')

    if pred_choice==0: #Plane
        #print('Shape is a plane')
        network = PlaneNet()
        network.load_state_dict(torch.load("/home/szj/SHREC2022/log/pointnet/plane/pla_model_249.pth"))
        network.cuda()

        network = network.eval()
        pred_normal, pred_point = network(input_pts)
    
        pred_normal = torch.squeeze(pred_normal).cpu().detach().numpy()
        pred_point = torch.squeeze(pred_point).cpu().detach().numpy()

        pred_point = pred_point*scale + center
        mag = LA.norm(pred_normal)
        pred_normal = pred_normal/mag

        f.write(str(pred_normal[0])+'\n')
        f.write(str(pred_normal[1])+'\n')
        f.write(str(pred_normal[2])+'\n')
        f.write(str(pred_point[0])+'\n')
        f.write(str(pred_point[1])+'\n')
        f.write(str(pred_point[2])+'\n')
        
    
        #print(f'Parameters: {pred_normal}->{pred_point}')
    elif pred_choice==1: #Cylinder
        #print('Shape is a cylinder')
        network = CylinderNet()
        network.load_state_dict(torch.load("/home/szj/SHREC2022/log/pointnet/cylinder/cyl_model_249.pth"))
        network.cuda()

        network = network.eval()
        pred_normal, pred_point, pred_radius = network(input_pts)

        pred_normal = torch.squeeze(pred_normal).cpu().detach().numpy()
        pred_point = torch.squeeze(pred_point).cpu().detach().numpy()
        pred_radius = torch.squeeze(pred_radius).cpu().detach().numpy()

        pred_point = pred_point*scale + center
        mag = LA.norm(pred_normal)
        pred_normal = pred_normal/mag
        pred_radius = pred_radius*scale

        f.write(str(pred_radius)+'\n')
        f.write(str(pred_normal[0])+'\n')
        f.write(str(pred_normal[1])+'\n')
        f.write(str(pred_normal[2])+'\n')
        f.write(str(pred_point[0])+'\n')
        f.write(str(pred_point[1])+'\n')
        f.write(str(pred_point[2])+'\n')

   
        #print(f'Parameters: {pred_normal}->{pred_point}->{pred_radius}')
    elif pred_choice==2: #Sphere
        #print('Shape is a sphere')
        network = SphereNet()
        network.load_state_dict(torch.load("/home/szj/SHREC2022/log/pointnet/sphere/sph_model_249.pth"))
        network.cuda()

        network = network.eval()
        pred_point, pred_radius = network(input_pts)

        pred_point = torch.squeeze(pred_point).cpu().detach().numpy()
        pred_radius = torch.squeeze(pred_radius).cpu().detach().numpy()
    
        pred_point = pred_point*scale + center
        pred_radius = pred_radius*scale

        f.write(str(pred_radius)+'\n')
        f.write(str(pred_point[0])+'\n')
        f.write(str(pred_point[1])+'\n')
        f.write(str(pred_point[2])+'\n')

        #print(f'Parameters: {pred_point}->{pred_radius}')
    elif pred_choice==3: #Cone
        #print('Shape is a cone')
        network = ConeNet()
        network.load_state_dict(torch.load("/home/szj/SHREC2022/log/pointnet/cone/con_model_249.pth"))
        network.cuda()

        network = network.eval()
        pred_normal, pred_point, pred_aperture = network(input_pts)

        pred_normal = torch.squeeze(pred_normal).cpu().detach().numpy()
        pred_point = torch.squeeze(pred_point).cpu().detach().numpy()
        pred_aperture = torch.squeeze(pred_aperture).cpu().detach().numpy()

        pred_point = pred_point*scale + center
        mag = LA.norm(pred_normal)
        pred_normal = pred_normal/mag

        f.write(str(pred_aperture)+'\n')
        f.write(str(pred_normal[0])+'\n')
        f.write(str(pred_normal[1])+'\n')
        f.write(str(pred_normal[2])+'\n')
        f.write(str(pred_point[0])+'\n')
        f.write(str(pred_point[1])+'\n')
        f.write(str(pred_point[2])+'\n')
    
        #print(f'Parameters: {pred_normal}->{pred_point}->{pred_aperture}')
    elif pred_choice==4: # Torus
        #print('Shape is a torus')
        network = TorusNet()
        network.load_state_dict(torch.load("/home/szj/SHREC2022/log/pointnet/torus/tor_model_249.pth"))
        network.cuda()

        network = network.eval()
        pred_normal, pred_point, pred_min, pred_max = network(input_pts)

        pred_normal = torch.squeeze(pred_normal).cpu().detach().numpy()
        pred_point = torch.squeeze(pred_point).cpu().detach().numpy()
        pred_min = torch.squeeze(pred_min).cpu().detach().numpy()
        pred_max = torch.squeeze(pred_max).cpu().detach().numpy()

        pred_point = pred_point*scale + center
        mag = LA.norm(pred_normal)
        pred_normal = pred_normal/mag
        pred_min = pred_min*scale
        pred_max = pred_max*scale

        f.write(str(pred_max)+'\n')
        f.write(str(pred_min)+'\n')
        f.write(str(pred_normal[0])+'\n')
        f.write(str(pred_normal[1])+'\n')
        f.write(str(pred_normal[2])+'\n')
        f.write(str(pred_point[0])+'\n')
        f.write(str(pred_point[1])+'\n')
        f.write(str(pred_point[2])+'\n')

        #print(f'Parameters: {pred_normal}->{pred_point}->{pred_min}->{pred_max}')

    elif pred_choice == 5:  # cuboid
        network = CuboidNet()
        network.load_state_dict(torch.load("/home/szj/SHREC2022/log/pointnet/cuboid/cub_model_249.pth"))
        network.cuda()

        network = network.eval()
        pred_axis, pred_uaxis, pred_point, pred_length, pred_width = network(input_pts)

        pred_axis = torch.squeeze(pred_axis).cpu().detach().numpy()
        pred_uaxis = torch.squeeze(pred_uaxis).cpu().detach().numpy()
        pred_point = torch.squeeze(pred_point).cpu().detach().numpy()
        pred_length = torch.squeeze(pred_length).cpu().detach().numpy()
        pred_width = torch.squeeze(pred_width).cpu().detach().numpy()

        pred_point = pred_point * scale + center
        pred_axis = pred_axis / LA.norm(pred_axis)
        pred_uaxis = pred_uaxis / LA.norm(pred_uaxis)
        pred_length = pred_length * scale
        pred_width = pred_width * scale

        f.write(str(pred_length)+'\n')
        f.write(str(pred_width)+'\n')
        f.write(str(pred_axis[0])+'\n')
        f.write(str(pred_axis[1])+'\n')
        f.write(str(pred_axis[2])+'\n')
        f.write(str(pred_uaxis[0])+'\n')
        f.write(str(pred_uaxis[1])+'\n')
        f.write(str(pred_uaxis[2])+'\n')
        f.write(str(pred_point[0])+'\n')
        f.write(str(pred_point[1])+'\n')
        f.write(str(pred_point[2])+'\n')

    elif pred_choice == 6:  # tee
        network = TeeNet()
        network.load_state_dict(torch.load("/home/szj/SHREC2022/log/pointnet/tee/tee_model_249.pth"))
        network.cuda()

        network = network.eval()

        pred_uaxis, pred_vaxis, pred_point, pred_main_radius, pred_vice_radius, pred_main_length, pred_vice_length = network(input_pts)

        pred_uaxis = torch.squeeze(pred_uaxis).cpu().detach().numpy()
        pred_vaxis = torch.squeeze(pred_vaxis).cpu().detach().numpy()
        pred_point = torch.squeeze(pred_point).cpu().detach().numpy()
        pred_main_radius = torch.squeeze(pred_main_radius).cpu().detach().numpy()
        pred_vice_radius = torch.squeeze(pred_vice_radius).cpu().detach().numpy()
        pred_main_length = torch.squeeze(pred_main_length).cpu().detach().numpy()
        pred_vice_length = torch.squeeze(pred_vice_length).cpu().detach().numpy()

        pred_point = pred_point * scale + center
        pred_uaxis = pred_uaxis / LA.norm(pred_uaxis)
        pred_vaxis = pred_vaxis / LA.norm(pred_vaxis)
        pred_main_radius = pred_main_radius * scale
        pred_vice_radius = pred_vice_radius * scale
        pred_main_length = pred_main_length * scale
        pred_vice_length = pred_vice_length * scale

        f.write(str(pred_main_radius)+'\n')
        f.write(str(pred_vice_radius)+'\n')
        f.write(str(pred_main_length)+'\n')
        f.write(str(pred_vice_length)+'\n')
        f.write(str(pred_point[0])+'\n')
        f.write(str(pred_point[1])+'\n')
        f.write(str(pred_point[2])+'\n')
        f.write(str(pred_uaxis[0])+'\n')
        f.write(str(pred_uaxis[1])+'\n')
        f.write(str(pred_uaxis[2])+'\n')
        f.write(str(pred_vaxis[0])+'\n')
        f.write(str(pred_vaxis[1])+'\n')
        f.write(str(pred_vaxis[2])+'\n')

    elif pred_choice == 7:  # cross
        network = CrossNet()
        network.load_state_dict(torch.load("/home/szj/SHREC2022/log/pointnet/cross/xos_model_249.pth"))
        network.cuda()

        network = network.eval()

        pred_uaxis, pred_vaxis, pred_point, pred_main_radius, pred_vice_radius, pred_main_length, pred_vice_length = network(input_pts)

        pred_uaxis = torch.squeeze(pred_uaxis).cpu().detach().numpy()
        pred_vaxis = torch.squeeze(pred_vaxis).cpu().detach().numpy()
        pred_point = torch.squeeze(pred_point).cpu().detach().numpy()
        pred_main_radius = torch.squeeze(pred_main_radius).cpu().detach().numpy()
        pred_vice_radius = torch.squeeze(pred_vice_radius).cpu().detach().numpy()
        pred_main_length = torch.squeeze(pred_main_length).cpu().detach().numpy()
        pred_vice_length = torch.squeeze(pred_vice_length).cpu().detach().numpy()

        pred_point = pred_point * scale + center
        pred_uaxis = pred_uaxis / LA.norm(pred_uaxis)
        pred_vaxis = pred_vaxis / LA.norm(pred_vaxis)
        pred_main_radius = pred_main_radius * scale
        pred_vice_radius = pred_vice_radius * scale
        pred_main_length = pred_main_length * scale
        pred_vice_length = pred_vice_length * scale

        f.write(str(pred_main_radius)+'\n')
        f.write(str(pred_vice_radius)+'\n')
        f.write(str(pred_main_length)+'\n')
        f.write(str(pred_vice_length)+'\n')
        f.write(str(pred_point[0])+'\n')
        f.write(str(pred_point[1])+'\n')
        f.write(str(pred_point[2])+'\n')
        f.write(str(pred_uaxis[0])+'\n')
        f.write(str(pred_uaxis[1])+'\n')
        f.write(str(pred_uaxis[2])+'\n')
        f.write(str(pred_vaxis[0])+'\n')
        f.write(str(pred_vaxis[1])+'\n')
        f.write(str(pred_vaxis[2])+'\n')

t2 = time()
print(f'{output_filename} {(t2-t1)}')