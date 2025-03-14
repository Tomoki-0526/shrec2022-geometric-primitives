from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import argparse
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import DatasetTorus
from model.models import Regressor
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from losses import TorusLoss

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

train_dataset = DatasetTorus(
        root=opt.dataset,
        npoints=opt.num_points,
        split='train',
        num_classes=8)

valid_dataset = DatasetTorus(
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
opt.output_dim = 8

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

torus_loss = TorusLoss()

num_batch = len(train_dataset) / opt.batchSize

lossTrainValues = []
lossTrainCenterValues = []
lossTrainAxisValues = []
lossTrainMinorValues = []
lossTrainMajorValues = []
lossValidValues = []
lossValidCenterValues = []
lossValidAxisValues = []
lossValidMinorValues = []
lossValidMajorValues = []

for epoch in range(opt.nepoch):
    m_loss = 0
    m_center_loss = 0
    m_axis_loss   = 0
    m_minor_loss = 0
    m_major_loss = 0

    for i, data in tqdm(enumerate(train_loader, 0)):
        optimizer.zero_grad()
        net = net.train()

        gt_normal, gt_xyz, gt_minor, gt_major, input_pts = data
        gt_minor, gt_major = gt_minor.view(-1, 1), gt_major.view(-1, 1)
        input_pts, gt_normal, gt_xyz, gt_minor, gt_major = \
            input_pts.to(device).float(), gt_normal.to(device).float(), gt_xyz.to(device).float(), gt_minor.to(device).float(), gt_major.to(device).float()
        pred = net(input_pts)
        gt = torch.cat([gt_major, gt_minor, gt_normal, gt_xyz], dim=1)

        a_loss, c_loss, R_loss, r_loss = torus_loss(pred, gt, None)
        a_loss = a_loss.mean(0)
        c_loss = c_loss.mean(0)
        r_loss = r_loss.mean(0)
        R_loss = R_loss.mean(0)

        loss = R_loss + r_loss + c_loss + a_loss
        
        loss.backward()
        optimizer.step()
        
        m_center_loss += c_loss.item()
        m_major_loss += R_loss.item()
        m_minor_loss += r_loss.item()
        m_axis_loss   += a_loss.item()
        m_loss += loss.item()

    scheduler.step()

    m_loss /= len(train_loader)
    m_center_loss /= len(train_loader)
    m_axis_loss /= len(train_loader)
    m_minor_loss /= len(train_loader)
    m_major_loss /= len(train_loader)
    print(f" Epoch: {epoch} | Training: Total loss = {m_loss}, Center loss: {m_center_loss}, Axis loss: {m_axis_loss}, Major radius loss: {m_major_loss}, Minor radius loss: {m_minor_loss}")
    
    lossTrainValues.append(m_loss)
    lossTrainCenterValues.append(m_center_loss)
    lossTrainAxisValues.append(m_axis_loss)
    lossTrainMinorValues.append(m_minor_loss)
    lossTrainMajorValues.append(m_major_loss)

    # Validation after one epoch
    best_loss = 100000
    best_epoch = 0
    with torch.no_grad():
        m_loss = 0
        m_center_loss = 0
        m_axis_loss   = 0
        m_minor_loss = 0
        m_major_loss = 0

        net = net.eval()

        for i,data in enumerate(valid_loader, 0):
            gt_normal, gt_xyz, gt_minor, gt_major, input_pts = data
            gt_minor, gt_major = gt_minor.view(-1, 1), gt_major.view(-1, 1)
            input_pts, gt_normal, gt_xyz, gt_minor, gt_major = \
                input_pts.to(device).float(), gt_normal.to(device).float(), gt_xyz.to(device).float(), gt_minor.to(device).float(), gt_major.to(device).float()
            pred = net(input_pts)
            gt = torch.cat([gt_major, gt_minor, gt_normal, gt_xyz], dim=1)

            a_loss, c_loss, R_loss, r_loss = torus_loss(pred, gt, None)
            c_loss = c_loss.mean(0)
            r_loss = r_loss.mean(0)
            R_loss = R_loss.mean(0)
            a_loss = a_loss.mean(0)

            loss = R_loss + r_loss + c_loss + a_loss

            m_center_loss += c_loss.item()
            m_major_loss += R_loss.item()
            m_minor_loss += r_loss.item()
            m_axis_loss   += a_loss.item()
            m_loss += loss.item()
        
        m_loss        /= len(valid_loader)
        m_center_loss /= len(valid_loader)
        m_axis_loss   /= len(valid_loader)
        m_minor_loss  /= len(valid_loader)
        m_major_loss  /= len(valid_loader)
        print(f" -------- | Validation: loss = {m_loss}, Center loss: {m_center_loss}, Axis loss: {m_axis_loss}, Major radius loss: {m_major_loss}, Minor radius loss: {m_minor_loss}")
        
        if m_loss < best_loss:
            best_loss = m_loss
            best_epoch = epoch
            torch.save(net.state_dict(), '%s/tor_model_best.pth' % (opt.outf))
        
        lossValidValues.append(m_loss)
        lossValidCenterValues.append(m_center_loss)
        lossValidAxisValues.append(m_axis_loss)
        lossValidMajorValues.append(m_major_loss)
        lossValidMinorValues.append(m_minor_loss)

print(f"Best epoch: {best_epoch}, Best loss: {best_loss}")

vis_curve(lossTrainValues, 'torus train loss', os.path.join(opt.outf, 'tor_train_loss.png'))
vis_curve(lossTrainCenterValues, 'torus train center loss', os.path.join(opt.outf, 'tor_train_center_loss.png'))
vis_curve(lossTrainAxisValues, 'torus train axis loss', os.path.join(opt.outf, 'tor_train_axis_loss.png'))
vis_curve(lossTrainMajorValues, 'torus train major radius loss', os.path.join(opt.outf, 'tor_train_major_loss.png'))
vis_curve(lossTrainMinorValues, 'torus train minor radius loss', os.path.join(opt.outf, 'tor_train_minor_loss.png'))
vis_curve(lossValidValues, 'torus validation loss', os.path.join(opt.outf, 'tor_valid_loss.png'))
vis_curve(lossValidCenterValues, 'torus validation center loss', os.path.join(opt.outf, 'tor_valid_center_loss.png'))
vis_curve(lossValidAxisValues, 'torus validation axis loss', os.path.join(opt.outf, 'tor_valid_axis_loss.png'))
vis_curve(lossValidMajorValues, 'torus validation major radius loss', os.path.join(opt.outf, 'tor_valid_major_loss.png'))
vis_curve(lossValidMinorValues, 'torus validation minor radius loss', os.path.join(opt.outf, 'tor_valid_minor_loss.png'))
