from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import argparse
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import DatasetTee
from model.models import Regressor
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from losses import TeeLoss

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

train_dataset = DatasetTee(
    root=opt.dataset,
    npoints=opt.num_points,
    split='train',
    num_classes=8,
)

valid_dataset = DatasetTee(
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

opt.input_dim = 3
opt.output_dim = 13

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

tee_loss = TeeLoss()

lossTrainValues = []
lossTrainUAxisValues = []
lossTrainVAxisValues = []
lossTrainPointValues = []
lossTrainMainRadiusValues = []
lossTrainViceRadiusValues = []
lossTrainMainLengthValues = []
lossTrainViceLengthValues = []

lossValidValues = []
lossValidUAxisValues = []
lossValidVAxisValues = []
lossValidPointValues = []
lossValidMainRadiusValues = []
lossValidViceRadiusValues = []
lossValidMainLengthValues = []
lossValidViceLengthValues = []

for epoch in range(opt.nepoch):
    m_loss = 0
    m_uaxis_loss = 0
    m_vaxis_loss = 0
    m_point_loss = 0
    m_main_radius_loss = 0
    m_vice_radius_loss = 0
    m_main_length_loss = 0
    m_vice_length_loss = 0

    for i, data in tqdm(enumerate(train_loader, 0)):
        optimizer.zero_grad()
        net = net.train()

        gt_uaxis, gt_vaxis, gt_point, gt_ar, gt_br, gt_al, gt_bl, input_pts = data
        gt_ar, gt_br, gt_al, gt_bl = gt_ar.view(-1, 1), gt_br.view(-1, 1), gt_al.view(-1, 1), gt_bl.view(-1, 1)
        input_pts, gt_uaxis, gt_vaxis, gt_point, gt_ar, gt_br, gt_al, gt_bl = \
            input_pts.to(device).float(), gt_uaxis.to(device).float(), gt_vaxis.to(device).float(), gt_point.to(device).float(), gt_ar.to(device).float(), gt_br.to(device).float(), gt_al.to(device).float(), gt_bl.to(device).float()
        pred = net(input_pts)
        gt = torch.cat([gt_uaxis, gt_vaxis, gt_point, gt_ar, gt_br, gt_al, gt_bl], dim=1)

        u_loss, v_loss, c_loss, ar_loss, br_loss, al_loss, bl_loss = tee_loss(pred, gt)
        u_loss = u_loss.mean(0)
        v_loss = v_loss.mean(0)
        c_loss = c_loss.mean(0)
        ar_loss = ar_loss.mean(0)
        br_loss = br_loss.mean(0)
        al_loss = al_loss.mean(0)
        bl_loss = bl_loss.mean(0)

        loss = u_loss + v_loss + c_loss + ar_loss + br_loss + al_loss + bl_loss
        loss.backward()
        optimizer.step()

        m_uaxis_loss += u_loss.item()
        m_vaxis_loss += v_loss.item()
        m_point_loss += c_loss.item()
        m_main_radius_loss += ar_loss.item()
        m_vice_radius_loss += br_loss.item()
        m_main_length_loss += al_loss.item()
        m_vice_length_loss += bl_loss.item()
        m_loss += u_loss.item()

    scheduler.step()

    m_loss /= len(train_loader)
    m_uaxis_loss /= len(train_loader)
    m_vaxis_loss /= len(train_loader)
    m_point_loss /= len(train_loader)
    m_main_radius_loss /= len(train_loader)
    m_vice_radius_loss /= len(train_loader)
    m_main_length_loss /= len(train_loader)
    m_vice_length_loss /= len(train_loader)
    print(f" Epoch: {epoch} | Training: Total loss = {m_loss}, UAxis loss: {m_uaxis_loss}, VAxis loss: {m_vaxis_loss}, Point loss: {m_point_loss}, Main Radius loss: {m_main_radius_loss}, Vice Radius loss: {m_vice_radius_loss}, Main Length Loss: {m_main_length_loss}, Vice Length loss: {m_vice_length_loss}")

    lossTrainValues.append(m_loss)
    lossTrainUAxisValues.append(m_uaxis_loss)
    lossTrainVAxisValues.append(m_vaxis_loss)
    lossTrainPointValues.append(m_point_loss)
    lossTrainMainRadiusValues.append(m_main_radius_loss)
    lossTrainViceRadiusValues.append(m_vice_radius_loss)
    lossTrainMainLengthValues.append(m_main_length_loss)
    lossTrainViceLengthValues.append(m_vice_length_loss)

    best_loss = 100000
    best_epoch = 0
    with torch.no_grad():
        m_loss = 0
        m_uaxis_loss = 0
        m_vaxis_loss = 0
        m_point_loss = 0
        m_main_radius_loss = 0
        m_vice_radius_loss = 0
        m_main_length_loss = 0
        m_vice_length_loss = 0

        net = net.eval()

        for i, data in enumerate(valid_loader, 0):
            gt_uaxis, gt_vaxis, gt_point, gt_ar, gt_br, gt_al, gt_bl, input_pts = data
            gt_ar, gt_br, gt_al, gt_bl = gt_ar.view(-1, 1), gt_br.view(-1, 1), gt_al.view(-1, 1), gt_bl.view(-1, 1)
            input_pts, gt_uaxis, gt_vaxis, gt_point, gt_ar, gt_br, gt_al, gt_bl = \
                input_pts.to(device).float(), gt_uaxis.to(device).float(), gt_vaxis.to(device).float(), gt_point.to(device).float(), gt_ar.to(device).float(), gt_br.to(device).float(), gt_al.to(device).float(), gt_bl.to(device).float()
            pred = net(input_pts)
            gt = torch.cat([gt_uaxis, gt_vaxis, gt_point, gt_ar, gt_br, gt_al, gt_bl], dim=1)

            u_loss, v_loss, c_loss, ar_loss, br_loss, al_loss, bl_loss = tee_loss(pred, gt)
            u_loss = u_loss.mean(0)
            v_loss = v_loss.mean(0)
            c_loss = c_loss.mean(0)
            ar_loss = ar_loss.mean(0)
            br_loss = br_loss.mean(0)
            al_loss = al_loss.mean(0)
            bl_loss = bl_loss.mean(0)

            loss = u_loss + v_loss + c_loss + ar_loss + br_loss + al_loss + bl_loss

            m_uaxis_loss += u_loss.item()
            m_vaxis_loss += v_loss.item()
            m_point_loss += c_loss.item()
            m_main_radius_loss += ar_loss.item()
            m_vice_radius_loss += br_loss.item()
            m_main_length_loss += al_loss.item()
            m_vice_length_loss += bl_loss.item()
            m_loss += u_loss.item()

        m_loss /= len(valid_loader)
        m_uaxis_loss /= len(valid_loader)
        m_vaxis_loss /= len(valid_loader)
        m_point_loss /= len(valid_loader)
        m_main_radius_loss /= len(valid_loader)
        m_vice_radius_loss /= len(valid_loader)
        m_main_length_loss /= len(valid_loader)
        m_vice_length_loss /= len(valid_loader)
        print(f" -------- | Validation: Total loss = {m_loss}, UAxis loss: {m_uaxis_loss}, VAxis loss: {m_vaxis_loss}, Point loss: {m_point_loss}, Main Radius loss: {m_main_radius_loss}, Vice Radius loss: {m_vice_radius_loss}, Main Length Loss: {m_main_length_loss}, Vice Length loss: {m_vice_length_loss}")

        if m_loss < best_loss:
            best_loss = m_loss
            best_epoch = epoch
            torch.save(net.state_dict(), '%s/tee_model_best.pth' % (opt.outf))

        lossValidValues.append(m_loss)
        lossValidUAxisValues.append(m_uaxis_loss)
        lossValidVAxisValues.append(m_vaxis_loss)
        lossValidPointValues.append(m_point_loss)
        lossValidMainRadiusValues.append(m_main_radius_loss)
        lossValidViceRadiusValues.append(m_vice_radius_loss)
        lossValidMainLengthValues.append(m_main_length_loss)
        lossValidViceLengthValues.append(m_vice_length_loss)

print(f"Best epoch: {best_epoch}, Best loss: {best_loss}")

vis_curve(lossTrainValues, 'tee train loss', os.path.join(opt.outf, 'tee_train_loss.png'))
vis_curve(lossTrainUAxisValues, 'tee train u axis loss', os.path.join(opt.outf, 'tee_train_uaxis_loss.png'))
vis_curve(lossTrainVAxisValues, 'tee train v axis loss', os.path.join(opt.outf, 'tee_train_vaxis_loss.png'))
vis_curve(lossTrainPointValues, 'tee train point loss', os.path.join(opt.outf, 'tee_train_point_loss.png'))
vis_curve(lossTrainMainRadiusValues, 'tee train main radius loss', os.path.join(opt.outf, 'tee_train_main_radius_loss.png'))
vis_curve(lossTrainViceRadiusValues, 'tee train vice radius loss', os.path.join(opt.outf, 'tee_train_vice_radius_loss.png'))
vis_curve(lossTrainMainLengthValues, 'tee train main length loss', os.path.join(opt.outf, 'tee_train_main_length_loss.png'))
vis_curve(lossTrainViceLengthValues, 'tee train vice length loss', os.path.join(opt.outf, 'tee_train_vice_length_loss.png'))

vis_curve(lossValidValues, 'tee validation loss', os.path.join(opt.outf, 'tee_valid_loss.png'))
vis_curve(lossValidUAxisValues, 'tee validation u axis loss', os.path.join(opt.outf, 'tee_valid_uaxis_loss.png'))
vis_curve(lossValidVAxisValues, 'tee validation v axis loss', os.path.join(opt.outf, 'tee_valid_vaxis_loss.png'))
vis_curve(lossValidPointValues, 'tee validation point loss', os.path.join(opt.outf, 'tee_valid_point_loss.png'))
vis_curve(lossValidMainRadiusValues, 'tee validation main radius loss', os.path.join(opt.outf, 'tee_valid_main_radius_loss.png'))
vis_curve(lossValidViceRadiusValues, 'tee validation vice radius loss', os.path.join(opt.outf, 'tee_valid_vice_radius_loss.png'))
vis_curve(lossValidMainLengthValues, 'tee validation main length loss', os.path.join(opt.outf, 'tee_valid_main_length_loss.png'))
vis_curve(lossValidViceLengthValues, 'tee validation vice length loss', os.path.join(opt.outf, 'tee_valid_vice_length_loss.png'))