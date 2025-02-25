from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import DatasetPlane, train_transforms, valid_transforms
from model.model import PlaneRegressor
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from loss import PlaneLoss
from utils import minkowski_collate, create_input_batch

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
        transform=train_transforms)

valid_dataset = DatasetPlane(
        root=opt.dataset,
        split='val',
        npoints=opt.num_points,
        transform=valid_transforms)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    collate_fn=minkowski_collate,
    num_workers=int(opt.workers))

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=minkowski_collate,
    num_workers=int(opt.workers))

print(len(train_dataset), len(valid_dataset))

try:
    os.makedirs(opt.outf)
except OSError:
    pass

regressor = PlaneRegressor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    regressor = torch.nn.DataParallel(regressor)
print(f'Let\'s use {torch.cuda.device_count()} gpu(s)!')

if opt.model != '':
    regressor.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(regressor.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
regressor.cuda()


num_batch = len(train_dataset) / opt.batchSize

lossTrainValues = []
lossValidValues = []

plane_loss = PlaneLoss()

for epoch in range(opt.nepoch):
    # for loss and accuracy tracking the training set
    m_loss = 0
    m_vertex_loss = 0
    m_normal_loss = 0

    for i, data in tqdm(enumerate(train_loader, 0)):
        optimizer.zero_grad()
        regressor = regressor.train()

        # reading the data and formating them
        labels = data['labels'].to(device)
        gt = labels[:, 1:]
        minknet_input = create_input_batch(
            data, 
            device=device,
            quantization_size=0.05
        )

        # predicting the normal vector
        pred = regressor(minknet_input)
        # getting the average of the points to get the plane vertex
        pred_vertex = data['means'].to(device)
        # adding to one vector
        pred = torch.cat([pred, pred_vertex], dim=-1)

        # calculating the loss
        normal_loss, vertex_loss = plane_loss(pred, gt, data['trans'])

        loss = normal_loss.mean(0) #+ vertex_loss.mean()
        loss.backward()
        optimizer.step()

        # tracking progress
        m_normal_loss += normal_loss.mean(0).item()
        m_vertex_loss += vertex_loss.mean(0).item()
        m_loss += loss.item()
    
    # stepping the scheduler
    scheduler.step()

    # epoch average scores   
    m_loss /= len(train_loader)
    m_normal_loss /= len(train_loader)
    m_vertex_loss /= len(train_loader)
    print(f" Epoch: {epoch} | Training: Total loss = {m_loss}, Normal loss: {m_normal_loss}, Vertex loss: {m_vertex_loss}")

    lossTrainValues.append(m_loss)

    # Validation after one epoch
    with torch.no_grad():
        # for loss and accuracy tracking the training set
        m_loss = 0
        m_vertex_loss = 0
        m_normal_loss = 0

        regressor = regressor.eval()
    
        for i, data in tqdm(enumerate(valid_loader, 0)):
            # reading the data and formating them
            labels = data['labels'].to(device)
            gt = labels[:, 1:]
            minknet_input = create_input_batch(
                data, 
                device=device,
                quantization_size=0.05
            )

            # predicting the normal vector
            pred = regressor(minknet_input)

            # getting the average of the points to get the plane vertex
            pred_vertex = data['means'].to(device)    
            # adding to one vector
            pred = torch.cat([pred, pred_vertex], dim=-1)

            # calculating the loss
            normal_loss, vertex_loss = plane_loss(pred, gt, data['trans'])

            # tracking progress
            m_normal_loss += normal_loss.mean(0).item()
            m_vertex_loss += vertex_loss.mean(0).item()
            m_loss += (normal_loss.mean(0).item() + vertex_loss.mean(0).item())

        # epoch average scores   
        m_loss        /= len(valid_loader)
        m_normal_loss /= len(valid_loader)
        m_vertex_loss /= len(valid_loader)
        print(f" -------- | Validation: Total loss = {m_loss}, Normal loss: {m_normal_loss}, Vertex loss: {m_vertex_loss}")
        
        lossValidValues.append(m_loss)

        if epoch == opt.nepoch - 1:
            torch.save(regressor.state_dict(), '%s/pla_model_%d.pth' % (opt.outf, epoch))

vis_curve(lossTrainValues, 'plane train loss', os.path.join(opt.outf, 'pla_train_loss.png'))
vis_curve(lossValidValues, 'plane validation loss', os.path.join(opt.outf, 'pla_valid_loss.png'))
