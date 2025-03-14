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
from model.models import PlaneNet
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from losses import PlaneLoss

from functools import partial
from pointcept.engines.defaults import worker_init_fn
from pointcept.datasets import point_collate_fn
import pointcept.utils.comm as comm

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

train_dataset = DatasetPlane(
        root=opt.dataset,
        npoints=opt.num_points,
        split='train',
        num_classes=8)

valid_dataset = DatasetPlane(
        root=opt.dataset,
        split='val',
        npoints=opt.num_points,
        num_classes=8)

init_fn = (
    partial(
        worker_init_fn,
        num_workers=opt.workers,
        rank=comm.get_rank(),
        seed=opt.manualSeed,
    )
    if opt.manualSeed is not None
    else None
)

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

net = PlaneNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net)
print(f'Let\'s use {torch.cuda.device_count()} gpu(s)!')

if opt.model != '':
    net.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
net.cuda()


num_batch = len(train_dataset) / opt.batchSize

lossTrainValues = []
lossValidValues = []

plane_loss = PlaneLoss()

for epoch in range(opt.nepoch):
    m_loss = 0
    m_xyz_loss = 0
    m_normal_loss = 0

    for i, data in tqdm(enumerate(train_loader, 0)):
        optimizer.zero_grad()
        net = net.train()

        gt_normal, gt_xyz, input_pts = data
        input_pts = input_pts.transpose(2, 1)
        input_pts, gt_normal, gt_xyz = \
            input_pts.to(device).float(), gt_normal.to(device).float(), gt_xyz.to(device).float()
        pred_normal, pred_xyz = net(input_pts)

        pred = torch.cat([pred_normal, pred_xyz], dim=1)
        gt = torch.cat([gt_normal, gt_xyz], dim=1)

        a_loss, v_loss = plane_loss(pred, gt, None)

        loss = a_loss.mean(0) + v_loss.mean(0)
        loss.backward()
        optimizer.step()

        m_normal_loss += a_loss.mean(0).item()
        m_xyz_loss += v_loss.mean(0).item()
        m_loss += loss.item()
    
    scheduler.step()

    m_loss /= len(train_loader)
    m_normal_loss /= len(train_loader)
    m_xyz_loss /= len(train_loader)
    print(f" Epoch: {epoch} | Training: Total loss = {m_loss}, Normal loss: {m_normal_loss}, Vertex loss: {m_xyz_loss}")

    lossTrainValues.append(m_loss)

    # Validation after one epoch
    with torch.no_grad():
        m_loss = 0
        m_xyz_loss = 0
        m_normal_loss = 0

        net = net.eval()
    
        for i, data in enumerate(valid_loader, 0):
            gt_normal, gt_xyz, input_pts = data
            input_pts = input_pts.transpose(2, 1)
            input_pts, gt_normal, gt_xyz = \
                input_pts.to(device).float(), gt_normal.to(device).float(), gt_xyz.to(device).float()
            pred_normal, pred_xyz = net(input_pts)

            pred = torch.cat([pred_normal, pred_xyz], dim=1)
            gt = torch.cat([gt_normal, gt_xyz], dim=1)

            # calculating the loss
            a_loss, v_loss = plane_loss(pred, gt, None)

            # tracking progress
            m_normal_loss += a_loss.mean(0).item()
            m_xyz_loss += v_loss.mean(0).item()
            m_loss += (a_loss.mean(0).item() + v_loss.mean(0).item())

        # epoch average scores   
        m_loss        /= len(valid_loader)
        m_normal_loss /= len(valid_loader)
        m_xyz_loss /= len(valid_loader)
        print(f" -------- | Validation: Total loss = {m_loss}, Normal loss: {m_normal_loss}, Vertex loss: {m_xyz_loss}")
        
        lossValidValues.append(m_loss)

        if epoch == opt.nepoch - 1:
            torch.save(net.state_dict(), '%s/pla_model_%d.pth' % (opt.outf, epoch))

vis_curve(lossTrainValues, 'plane train loss', os.path.join(opt.outf, 'pla_train_loss.png'))
vis_curve(lossValidValues, 'plane validation loss', os.path.join(opt.outf, 'pla_valid_loss.png'))
