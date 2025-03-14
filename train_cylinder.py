from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import DatasetCylinder
from model.models import Regressor
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from losses import CylinderLoss

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

train_dataset = DatasetCylinder(
        root=opt.dataset,
        npoints=opt.num_points,
        split='train',
        num_classes=8)

valid_dataset = DatasetCylinder(
        root=opt.dataset,
        split='val',
        npoints=opt.num_points,
        num_classes=8)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(train_dataset), len(valid_dataset))

try:
    os.makedirs(opt.outf)
except OSError:
    pass

opt.input_dim = 3
opt.output_dim = 7

net = Regressor(opt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net)
print(f'Let\'s use {torch.cuda.device_count()} gpu(s)!')

if opt.model != '':
    net.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
net.cuda()

cylinder_loss = CylinderLoss()

num_batch = len(train_dataset) / opt.batchSize

lossTrainValues = []
lossTrainAxisValues = []
lossTrainVertexValues = []
lossTrainRadiusValues = []
lossValidValues = []
lossValidAxisValues = []
lossValidVertexValues = []
lossValidRadiusValues = []

for epoch in range(opt.nepoch):
    m_loss = 0
    m_axis_loss = 0
    m_vertex_loss = 0
    m_radius_loss = 0
    
    for i, data in tqdm(enumerate(train_loader, 0)):
        optimizer.zero_grad()
        net = net.train()

        gt_normal, gt_xyz, gt_radius, input_pts = data
        gt_radius = gt_radius.view(-1, 1)
        input_pts, gt_normal, gt_xyz, gt_radius = \
            input_pts.to(device).float(), gt_normal.to(device).float(), gt_xyz.to(device).float(), gt_radius.to(device).float()
        pred = net(input_pts)
        gt = torch.cat([gt_radius, gt_normal, gt_xyz], dim=1)

        a_loss, v_loss, r_loss = cylinder_loss(pred, gt, None)
        a_loss = a_loss.mean(0)
        v_loss = v_loss.mean(0)
        r_loss = r_loss.mean(0)

        loss = a_loss + v_loss + r_loss
        loss.backward()
        optimizer.step()

        m_axis_loss += a_loss.item()
        m_vertex_loss += v_loss.item()
        m_radius_loss += r_loss.item()
        m_loss += loss.item()

    scheduler.step()

    m_loss        /= len(train_loader)
    m_axis_loss   /= len(train_loader)
    m_vertex_loss /= len(train_loader)
    m_radius_loss /= len(train_loader)
    print(f" Epoch: {epoch} | Training: Total loss = {m_loss}, Axis loss: {m_axis_loss}, Vertex loss: {m_vertex_loss}, Radius loss: {m_radius_loss}")

    lossTrainValues.append(m_loss)
    lossTrainAxisValues.append(m_axis_loss)
    lossTrainVertexValues.append(m_vertex_loss)
    lossTrainRadiusValues.append(m_radius_loss)
   
    # Validation after one epoch
    best_loss = 100000
    best_epoch = 0
    with torch.no_grad():
        m_loss = 0
        m_axis_loss = 0
        m_vertex_loss = 0
        m_radius_loss = 0

        net = net.eval()
    
        for i, data in enumerate(valid_loader, 0):
            gt_normal, gt_xyz, gt_radius, input_pts = data
            gt_radius = gt_radius.view(-1, 1)
            input_pts, gt_normal, gt_xyz, gt_radius = \
                input_pts.to(device).float(), gt_normal.to(device).float(), gt_xyz.to(device).float(), gt_radius.to(device).float()
            pred = net(input_pts)
            gt = torch.cat([gt_radius, gt_normal, gt_xyz], dim=1)

            a_loss, v_loss, r_loss = cylinder_loss(pred, gt, None)
            a_loss = a_loss.mean(0)
            v_loss = v_loss.mean(0)
            r_loss = r_loss.mean(0)

            loss = a_loss + v_loss + r_loss

            m_axis_loss += a_loss.item()
            m_vertex_loss += v_loss.item()
            m_radius_loss += r_loss.item()
            m_loss += loss.item()

        m_loss        /= len(valid_loader)
        m_axis_loss   /= len(valid_loader)
        m_vertex_loss /= len(valid_loader)
        m_radius_loss /= len(valid_loader)
        print(f" -------- | Validation: Total loss = {m_loss}, Axis loss: {m_axis_loss}, Vertex loss: {m_vertex_loss}, Radius loss: {m_radius_loss}")

        if m_loss < best_loss:
            best_loss = m_loss
            best_epoch = epoch
            torch.save(net.state_dict(), '%s/cyl_model_best.pth' % (opt.outf))
        
        lossValidValues.append(m_loss)
        lossValidAxisValues.append(m_axis_loss)
        lossValidVertexValues.append(m_vertex_loss)
        lossValidRadiusValues.append(m_radius_loss)

print(f"Best epoch: {best_epoch}, Best loss: {best_loss}")

vis_curve(lossTrainValues, 'cylinder train loss', os.path.join(opt.outf, 'cyl_train_loss.png'))
vis_curve(lossTrainAxisValues, 'cylinder train axis loss', os.path.join(opt.outf, 'cyl_train_axis_loss.png'))
vis_curve(lossTrainVertexValues, 'cylinder train vertex loss', os.path.join(opt.outf, 'cyl_train_vertex_loss.png'))
vis_curve(lossTrainRadiusValues, 'cylinder train radius loss', os.path.join(opt.outf, 'cyl_train_radius_loss.png'))
vis_curve(lossValidValues, 'cylinder validation loss', os.path.join(opt.outf, 'cyl_valid_loss.png'))
vis_curve(lossValidAxisValues, 'cylinder validation axis loss', os.path.join(opt.outf, 'cyl_valid_axis_loss.png'))
vis_curve(lossValidVertexValues, 'cylinder validation vertex loss', os.path.join(opt.outf, 'cyl_valid_vertex_loss.png'))
vis_curve(lossValidRadiusValues, 'cylinder validation radius loss', os.path.join(opt.outf, 'cyl_valid_radius_loss.png'))
