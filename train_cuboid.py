from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import argparse
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import DatasetCuboid
from model.models import CuboidNet
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from losses import CuboidLoss

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

train_dataset = DatasetCuboid(
    root=opt.dataset,
    npoints=opt.num_points,
    split='train',
    num_classes=8,
)

valid_dataset = DatasetCuboid(
    root=opt.dataset,
    npoints=opt.num_points,
    split='val',
    num_classes=8,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=int(opt.workers),
)

print(len(train_dataset), len(valid_dataset))

try:
    os.makedirs(opt.outf)
except OSError:
    pass

net = CuboidNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net)
print(f'Let\'s use {torch.cuda.device_count()} gpu(s)!')

if opt.model != '':
    net.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
net.cuda()

cuboid_loss = CuboidLoss()

lossTrainValues = []
lossTrainAxisValues = []
lossTrainUAxisValues = []
lossTrainPointValues = []
lossTrainLengthValues = []
lossTrainWidthValues = []
lossValidValues = []
lossValidAxisValues = []
lossValidUAxisValues = []
lossValidPointValues = []
lossValidLengthValues = []
lossValidWidthValues = []

for epoch in range(opt.nepoch):
    m_loss = 0
    m_axis_loss = 0
    m_uaxis_loss = 0
    m_point_loss = 0
    m_length_loss = 0
    m_width_loss = 0

    for i, data in tqdm(enumerate(train_loader, 0)):
        optimizer.zero_grad()
        net = net.train()

        gt_axis, gt_uaxis, gt_point, gt_a, gt_b, input_pts, center, scale = data
        input_pts = input_pts.transpose(2, 1)
        gt_a, gt_b, scale = gt_a.view(-1, 1), gt_b.view(-1, 1), scale.view(-1, 1)
        input_pts, gt_axis, gt_uaxis, gt_point, gt_a, gt_b, center, scale = \
            input_pts.to(device).float(), gt_axis.to(device).float(), gt_uaxis.to(device).float(), gt_point.to(device).float(), gt_a.to(device).float(), gt_b.to(device).float(), center.to(device).float(), scale.to(device).float()
        pred_axis, pred_uaxis, pred_point, pred_a, pred_b = net(input_pts)
        pred_point = pred_point * scale + center
        pred_a = pred_a.view(-1, 1) * scale
        pred_b = pred_b.view(-1, 1) * scale

        pred = torch.cat([pred_axis, pred_uaxis, pred_point, pred_a, pred_b], dim=1)
        gt = torch.cat([gt_axis, gt_uaxis, gt_point, gt_a, gt_b], dim=1)

        n_loss, u_loss, c_loss, a_loss, b_loss = cuboid_loss(pred, gt)
        n_loss = n_loss.mean(0)
        u_loss = u_loss.mean(0)
        c_loss = c_loss.mean(0)
        a_loss = a_loss.mean(0)
        b_loss = b_loss.mean(0)

        loss = n_loss + u_loss + c_loss + a_loss + b_loss
        loss.backward()
        optimizer.step()

        m_axis_loss += n_loss.item()
        m_uaxis_loss += u_loss.item()
        m_point_loss += c_loss.item()
        m_length_loss += a_loss.item()
        m_width_loss += b_loss.item()
        m_loss += loss.item()

    scheduler.step()

    m_loss          /= len(train_loader)
    m_axis_loss     /= len(train_loader)
    m_uaxis_loss    /= len(train_loader)
    m_point_loss    /= len(train_loader)
    m_length_loss   /= len(train_loader)
    m_width_loss    /= len(train_loader)
    print(f" Epoch: {epoch} | Training: Total loss = {m_loss}, Axis loss: {m_axis_loss}, UAxis loss: {m_uaxis_loss}, Vertex loss: {m_point_loss}, Length loss: {m_length_loss}, Width loss: {m_width_loss}")

    lossTrainValues.append(m_loss)
    lossTrainAxisValues.append(m_axis_loss)
    lossTrainUAxisValues.append(m_uaxis_loss)
    lossTrainPointValues.append(m_point_loss)
    lossTrainLengthValues.append(m_length_loss)
    lossTrainWidthValues.append(m_width_loss)

    with torch.no_grad():
        m_loss = 0
        m_axis_loss = 0
        m_uaxis_loss = 0
        m_point_loss = 0
        m_length_loss = 0
        m_width_loss = 0

        net = net.eval()

        for i, data in enumerate(valid_loader, 0):
            gt_axis, gt_uaxis, gt_point, gt_a, gt_b, input_pts, center, scale = data
            input_pts = input_pts.transpose(2, 1)
            gt_a, gt_b, scale = gt_a.view(-1, 1), gt_b.view(-1, 1), scale.view(-1, 1)
            input_pts, gt_axis, gt_uaxis, gt_point, gt_a, gt_b, center, scale = \
                input_pts.to(device).float(), gt_axis.to(device).float(), gt_uaxis.to(device).float(), gt_point.to(device).float(), gt_a.to(device).float(), gt_b.to(device).float(), center.to(device).float(), scale.to(device).float()
            pred_axis, pred_uaxis, pred_point, pred_a, pred_b = net(input_pts)
            pred_point = pred_point * scale + center
            pred_a = pred_a.view(-1, 1) * scale
            pred_b = pred_b.view(-1, 1) * scale

            pred = torch.cat([pred_axis, pred_uaxis, pred_point, pred_a, pred_b], dim=1)
            gt = torch.cat([gt_axis, gt_uaxis, gt_point, gt_a, gt_b], dim=1)

            n_loss, u_loss, c_loss, a_loss, b_loss = cuboid_loss(pred, gt)
            n_loss = n_loss.mean(0)
            u_loss = u_loss.mean(0)
            c_loss = c_loss.mean(0)
            a_loss = a_loss.mean(0)
            b_loss = b_loss.mean(0)

            loss = n_loss + u_loss + c_loss + a_loss + b_loss

            m_axis_loss += n_loss.item()
            m_uaxis_loss += u_loss.item()
            m_point_loss += c_loss.item()
            m_length_loss += a_loss.item()
            m_width_loss += b_loss.item()
            m_loss += loss.item()

        m_loss          /= len(valid_loader)
        m_axis_loss     /= len(valid_loader)
        m_uaxis_loss    /= len(valid_loader)
        m_point_loss    /= len(valid_loader)
        m_length_loss   /= len(valid_loader)
        m_width_loss    /= len(valid_loader)
        print(f" --------- | Validation: Total loss = {m_loss}, Axis loss: {m_axis_loss}, UAxis loss: {m_uaxis_loss}, Vertex loss: {m_point_loss}, Length loss: {m_length_loss}, Width loss: {m_width_loss}")

    lossValidValues.append(m_loss)
    lossValidAxisValues.append(m_axis_loss)
    lossValidUAxisValues.append(m_uaxis_loss)
    lossValidPointValues.append(m_point_loss)
    lossValidLengthValues.append(m_length_loss)
    lossValidWidthValues.append(m_width_loss)

    if epoch == opt.nepoch - 1:
        torch.save(net.state_dict(), '%s/cub_model_%d.pth' % (opt.outf, epoch))

vis_curve(lossTrainValues, 'cuboid train loss', os.path.join(opt.outf, 'cub_train_loss.png'))
vis_curve(lossTrainAxisValues, 'cuboid train axis loss', os.path.join(opt.outf, 'cub_train_axis_loss.png'))
vis_curve(lossTrainUAxisValues, 'cuboid train u axis loss', os.path.join(opt.outf, 'cub_train_uaxis_loss.png'))
vis_curve(lossTrainPointValues, 'cuboid train point loss', os.path.join(opt.outf, 'cub_train_point_loss.png'))
vis_curve(lossTrainLengthValues, 'cuboid train length loss', os.path.join(opt.outf, 'cub_train_length_loss.png'))
vis_curve(lossTrainWidthValues, 'cuboid train width loss', os.path.join(opt.outf, 'cub_train_width_loss.png'))
vis_curve(lossValidValues, 'cuboid validation loss', os.path.join(opt.outf, 'cub_valid_loss.png'))
vis_curve(lossValidAxisValues, 'cuboid validation axis loss', os.path.join(opt.outf, 'cub_valid_axis_loss.png'))
vis_curve(lossValidUAxisValues, 'cuboid validation u axis loss', os.path.join(opt.outf, 'cub_valid_uaxis_loss.png'))
vis_curve(lossValidPointValues, 'cuboid validation point loss', os.path.join(opt.outf, 'cub_valid_point_loss.png'))
vis_curve(lossValidLengthValues, 'cuboid validation length loss', os.path.join(opt.outf, 'cub_valid_length_loss.png'))
vis_curve(lossValidWidthValues, 'cuboid validation width loss', os.path.join(opt.outf, 'cub_valid_width_loss.png'))
