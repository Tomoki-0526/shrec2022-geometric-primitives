from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import DatasetSphere
from model.model import SphereRegressor
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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

dataset = DatasetSphere(
        root=opt.dataset,
        npoints=opt.num_points,
        split='train', transform=False)

test_dataset = DatasetSphere(
        root=opt.dataset,
        split='val',
        npoints=opt.num_points, transform=False)

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

regressor = SphereRegressor()
if torch.cuda.device_count() > 1:
    regressor = torch.nn.DataParallel(regressor)

if opt.model != '':
    regressor.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(regressor.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
regressor.cuda()
myloss = torch.nn.MSELoss()
mylossCen = torch.nn.MSELoss()


num_batch = len(dataset) / opt.batchSize

lossTrainValues = []
lossTestValues = []
lossLoss1 = []
lossLoss2 = []


for epoch in range(opt.nepoch):
    running_loss = 0
    cont = 0

    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        target_center, target_radius, points = data
        
        points = points.transpose(2, 1)
        points = points.cuda().float()
        target_center = target_center.cuda().float()
        target_radius = target_radius.cuda().float()
        
        optimizer.zero_grad()
        regressor = regressor.train()
        pred_center, pred_radius = regressor(points)
        loss = mylossCen(pred_center, target_center) + myloss(pred_radius, target_radius)
        loss.mean().backward()
        optimizer.step()
        print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.mean().item()))
        running_loss += loss.mean().item()
        cont += 1

    lossTrainValues.append(running_loss / float(cont))

    #Validation after one epoch
    running_loss = 0
    
    runningCenter = 0
    runningRadius = 0
    
    cont = 0
    for i,data in tqdm(enumerate(testdataloader, 0)):
        target_center, target_radius, points = data
        
        points = points.transpose(2, 1)
        points = points.cuda().float()
        target_center = target_center.cuda().float()
        target_radius = target_radius.cuda().float()
        regressor = regressor.eval()
        pred_center, pred_radius  = regressor(points)

        lossCenter = mylossCen(target_center, pred_center)
        lossRadius = myloss(target_radius, pred_radius)
        loss = lossCenter + lossRadius

        running_loss += loss.item()
        runningCenter += lossCenter.item()
        runningRadius += lossRadius.item()
        
        cont = cont + 1
    
    lossTestValues.append(running_loss/float(cont))
    lossLoss1.append(runningCenter/float(cont))
    lossLoss2.append(runningRadius/float(cont))
    
    if epoch == opt.nepoch - 1:
        torch.save(regressor.state_dict(), '%s/sph_model_%d.pth' % (opt.outf, epoch))

vis_curve(lossTrainValues, 'sphere train loss', os.path.join(opt.outf, 'sph_train_loss.png'))
vis_curve(lossTestValues, 'sphere test loss - all', os.path.join(opt.outf, 'sph_test_loss_all.png'))
vis_curve(lossLoss1, 'sphere test loss - center', os.path.join(opt.outf, 'sph_test_loss_c.png'))
vis_curve(lossLoss2, 'sphere test loss - radius', os.path.join(opt.outf, 'sph_test_loss_r.png'))

running_loss = 0

cont = 0
center_err = 0
radius_err = 0

centers = []
radii = []

for i,data in tqdm(enumerate(testdataloader, 0)):
    target_center, target_radius, points = data
    print(points.shape)
    
    points = points.transpose(2, 1)
    points = points.cuda().float()
    target_center = target_center.cuda().float()
    target_radius = target_radius.cuda().float()
    
    regressor = regressor.eval()
    pred_center, pred_radius = regressor(points)
    
    lossCenter = mylossCen(target_center, pred_center)
    lossRadius = myloss(target_radius, pred_radius)
    loss = lossCenter + lossRadius
    
    t = np.squeeze(target_center.detach().cpu().numpy())
    p = np.squeeze(pred_center.detach().cpu().numpy())

    t1 = np.squeeze(target_radius.detach().cpu().numpy())
    p1 = np.squeeze(pred_radius.detach().cpu().numpy())

    c1 = np.linalg.norm(t-p)
    c2 = np.linalg.norm(t1-p1)

    center_err += c1
    radius_err += c2
    centers.append(c1)
    radii.append(c2)

    print(f'{t} -> {p}->({t1},{p1})')

    running_loss += loss.item()
    cont = cont + 1
    
print("final accuracy {}".format(running_loss / float(cont)))
print("average center error {}".format(center_err / float(cont)))
print("average radius error {}".format(radius_err / float(cont)))

fig1, axes = plt.subplots(1,2)
axes[0].hist(centers, 50)

axes[1].hist(radii, 50)
plt.show()



