from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import DatasetSHREC2022
from model.model import Classifier
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
parser.add_argument('--dataset_type', type=str, default='shrec2022', help="dataset type shapenet|modelnet40|shrec2022")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shrec2022':
    dataset = DatasetSHREC2022(
        root=opt.dataset,
        npoints=opt.num_points,
        split='train')

    test_dataset = DatasetSHREC2022(
        root=opt.dataset,
        split='val',
        npoints=opt.num_points)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

num_classes = 5

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = Classifier(num_classes=num_classes)
if torch.cuda.device_count() > 1:
    classifier = torch.nn.DataParallel(classifier)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize

lossTrain = []
lossTest = []
accTrain = []
accTest = []

for epoch in range(opt.nepoch):
    running_loss = 0
    running_acc = 0
    cont = 0
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        target, points = data
        points = points.transpose(2, 1)
        points, target = points.cuda().float(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, _ = classifier(points)
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))
        running_loss += loss.item()
        running_acc += correct.item() / float(opt.batchSize)
        cont += 1

    lossTrain.append(running_loss / float(cont))
    accTrain.append(running_acc / float(cont))

    running_loss = 0
    running_acc = 0
    cont = 0
    for i,data in tqdm(enumerate(testdataloader, 0)):
        target, points = data
        points = points.transpose(2, 1)
        points, target = points.cuda().float(), target.cuda()
        classifier = classifier.eval()
        pred, _ = classifier(points)
        loss = F.nll_loss(pred, target)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))
        running_loss += loss.item()
        running_acc += correct.item() / float(opt.batchSize)
        cont += 1

    lossTest.append(running_loss / float(cont))
    accTest.append(running_acc / float(cont))

    if epoch == opt.nepoch - 1:
        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

vis_curve(lossTrain, 'classification train loss', os.path.join(opt.outf, 'cls_train_loss.png'))
vis_curve(accTrain, 'classification train accuracy', os.path.join(opt.outf, 'cls_train_acc.png'))
vis_curve(lossTest, 'classification test loss', os.path.join(opt.outf, 'cls_test_loss.png'))
vis_curve(accTest, 'classification test accuracy', os.path.join(opt.outf, 'cls_test_acc.png'))

total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(testdataloader, 0)):
    target, points = data
    points = points.transpose(2, 1)
    points, target = points.cuda().float(), target.cuda()
    classifier = classifier.eval()
    pred, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))