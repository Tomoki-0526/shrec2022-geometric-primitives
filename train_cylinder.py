from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import argparse
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import DatasetCylinder
from model.model import CylinderRegressor
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from loss import CylinderLoss

def vis_curve(curve, title, filename):
    plt.clf()
    X=np.arange(len(curve))
    Y=np.array(curve)
    plt.xlabel('epochs')
    plt.ylabel('value')
    plt.plot(X, Y)
    plt.title(title)
    plt.savefig(filename)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2048, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = DatasetCylinder(
        root=opt.dataset,
        npoints=opt.num_points,
        split='train')

test_dataset = DatasetCylinder(
        root=opt.dataset,
        split='val',
        npoints=opt.num_points)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))

try:
    os.makedirs(opt.outf)
except OSError:
    pass

regressor = CylinderRegressor()
if torch.cuda.device_count() > 1:
    regressor = torch.nn.DataParallel(regressor)
    print(f'Let\'s use {torch.cuda.device_count()} gpus!')

if opt.model != '':
    regressor.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(regressor.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
regressor.cuda()

cylinder_loss = CylinderLoss()

num_batch = len(dataset) / opt.batchSize

lossTrainValues = []
lossTestValues = []
lossLoss1 = []
lossLoss2 = []
lossLoss3 = []
lossLoss4 = []
delta = 1/256

for epoch in range(opt.nepoch):
    running_loss = 0
    cont = 0
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        target_normal, target_center, target_radius, points = data
        points = points.transpose(2, 1)
        points, target_normal = points.cuda().float(), target_normal.cuda().float()
        target_center, target_radius = target_center.cuda().float(), target_radius.cuda().float()
        
        optimizer.zero_grad()
        regressor = regressor.train()
        pred_normal, pred_center, pred_radius = regressor(points)

        pred = torch.cat([pred_radius, pred_normal, pred_center], dim=1)
        gt = torch.cat([target_radius, target_normal, target_center], dim=1)
        loss_normal, loss_center, loss_radius = cylinder_loss(pred, gt, None)
        loss = loss_normal.mean(0) + loss_center.mean(0) + loss_radius.mean(0)

        loss.backward()
        optimizer.step()
        print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))

        running_loss += loss.mean().item()
        cont += 1

    lossTrainValues.append(running_loss / float(cont))
   
    #Validation after one epoch
    running_loss = 0

    cont = 0
    for i,data in tqdm(enumerate(testdataloader, 0)):
        target_normal, target_center, target_radius, points = data
        points = points.transpose(2, 1)
        points, target_normal = points.cuda().float(), target_normal.cuda().float()
        target_center, target_radius = target_center.cuda().float(), target_radius.cuda().float()

        regressor = regressor.eval()
        pred_normal, pred_center, pred_radius = regressor(points)

        pred = torch.cat([pred_radius, pred_normal, pred_center], dim=1)
        gt = torch.cat([target_radius, target_normal, target_center], dim=1)
        loss_normal, loss_center, loss_radius = cylinder_loss(pred, gt, None)
        loss = loss_normal.mean(0) + loss_center.mean(0) + loss_radius.mean(0)
        running_loss += loss.item()
        cont = cont + 1
    
    lossTestValues.append(running_loss/float(cont))

    if epoch == opt.nepoch - 1:
        torch.save(regressor.state_dict(), '%s/cyl_model_%d.pth' % (opt.outf, epoch))

vis_curve(lossTrainValues, 'cylinder train loss', os.path.join(opt.outf, 'cyl_train_loss.png'))
vis_curve(lossTestValues, 'cylinder test loss - all', os.path.join(opt.outf, 'cyl_test_loss_all.png'))

angle_err = 0
point_err = 0
rad_err = 0

angles = []
distances = []
radii = []

cont = 0

for i,data in tqdm(enumerate(testdataloader, 0)):
    target_normal, target_center, target_radius, points = data
    points = points.transpose(2, 1)
    points, target_normal = points.cuda().float(), target_normal.cuda().float()
    target_center, target_radius = target_center.cuda().float(), target_radius.cuda().float()

    regressor = regressor.eval()
    pred_normal, pred_center, pred_radius = regressor(points)
    
    t = np.squeeze(target_normal.detach().cpu().numpy())
    p = np.squeeze(pred_normal.detach().cpu().numpy()) 
    norm_p = np.linalg.norm(p)
    p = p/norm_p
    angle = 180*np.arccos(t.dot(p))/np.pi
    angles.append(angle)
    angle_err += angle

    t1 = np.squeeze(target_center.detach().cpu().numpy())
    p1 = np.squeeze(pred_center.detach().cpu().numpy()) 
    dist = np.linalg.norm(t1-p1)
    distances.append(dist)
    point_err += dist

    t2 = np.squeeze(target_radius.detach().cpu().numpy())
    p2 = np.squeeze(pred_radius.detach().cpu().numpy()) 
    dist2 = np.linalg.norm(t2-p2)
    radii.append(dist2)
    rad_err += dist2

    print(f'{t} -> {p}->{angle}')
    cont = cont + 1
    
print("average angle error {}".format(angle_err / float(cont)))
print("average point error {}".format(point_err / float(cont)))
print("average radii error {}".format(rad_err / float(cont)))


fig,axes = plt.subplots(1,3)
axes[0].hist(angles, 50)
axes[1].hist(distances, 50)
axes[2].hist(radii, 50)
plt.show()
