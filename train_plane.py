from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import argparse
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import DatasetPlane
from model.model import PlaneRegressor
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from loss import PlaneLoss

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

dataset = DatasetPlane(
        root=opt.dataset,
        npoints=opt.num_points,
        split='train')

test_dataset = DatasetPlane(
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

regressor = PlaneRegressor()
if torch.cuda.device_count() > 1:
    regressor = torch.nn.DataParallel(regressor)
    print(f'Let\'s use {torch.cuda.device_count()} gpus!')

if opt.model != '':
    regressor.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(regressor.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
regressor.cuda()


num_batch = len(dataset) / opt.batchSize

lossTrainValues = []
lossTestValues = []

plane_loss = PlaneLoss()

for epoch in range(opt.nepoch):
    running_loss = 0
    cont = 0
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        target_normal, target_xyz, points = data
        points = points[0].transpose(2, 1)
        points, target_normal, target_xyz = \
            points.cuda().float(), target_normal.cuda().float(), target_xyz.cuda().float()
        optimizer.zero_grad()
        regressor = regressor.train()
        pred_normal, pred_xyz = regressor(points)
        
        pred = torch.cat([pred_normal, pred_xyz], dim=1)
        gt = torch.cat([target_normal, target_xyz], dim=1)
        loss_normal, loss_xyz = plane_loss(pred, gt, None)
        loss = loss_normal.mean(0)# + loss_xyz.mean(0)
        loss.backward()
        optimizer.step()
        print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.mean().item()))
        running_loss += loss.mean().item()
        cont += 1

    lossTrainValues.append(running_loss / float(cont))

    #Validation after one epoch
    running_loss = 0
    cont = 0
    for i,data in tqdm(enumerate(testdataloader, 0)):
        target_normal, target_xyz, points = data
        #target = target[:, 0]
        points = points[0].transpose(2, 1)
        points, target_normal, target_xyz = \
            points.cuda().float(), target_normal.cuda().float(), target_xyz.cuda().float()
        regressor = regressor.eval()
        pred_normal, pred_xyz = regressor(points)

        pred = torch.cat([pred_normal, pred_xyz], dim=1)
        gt = torch.cat([target_normal, target_xyz], dim=1)
        loss_normal, loss_xyz = plane_loss(pred, gt, None)
        loss = loss_normal.mean(0)# + loss_xyz.mean(0)
        running_loss += loss.item()
        cont += 1
    
    lossTestValues.append(running_loss/float(cont))

    if epoch == opt.nepoch - 1:
        torch.save(regressor.state_dict(), '%s/pla_model_%d.pth' % (opt.outf, epoch))

vis_curve(lossTrainValues, 'plane train loss', os.path.join(opt.outf, 'pla_train_loss.png'))
vis_curve(lossTestValues, 'plane test loss - all (normal cosine)', os.path.join(opt.outf, 'pla_test_loss_all.png'))

running_loss = 0
cont = 0

for i,data in tqdm(enumerate(testdataloader, 0)):
    target_normal, target_xyz, points = data
    points = points[0].transpose(2, 1)
    points, target_normal, target_xyz = \
        points.cuda().float(), target_normal.cuda().float(), target_xyz.cuda().float()
    regressor = regressor.eval()
    pred_normal, pred_xyz = regressor(points)
    
    t = np.squeeze(target_normal.detach().cpu().numpy())
    p = np.squeeze(pred_normal.detach().cpu().numpy())
    norm_p = np.linalg.norm(p)
    p = p/norm_p
    angle = 180*np.arccos(t.dot(p))/np.pi
    print(f'{t} -> {p}->{angle}')

    pred = torch.cat([pred_normal, pred_xyz], dim=1)
    gt = torch.cat([target_normal, target_xyz], dim=1)
    loss_normal, loss_xyz = plane_loss(pred, gt, None)
    loss = loss_normal.mean(0)# + loss_xyz.mean(0)
    running_loss += loss.item()
    cont = cont + 1
    
print("final loss {}".format(running_loss / float(cont)))
