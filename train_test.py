# -*- coding: utf-8 -*-
# @Author  : Mumu
# @Time    : 2021/11/18 9:40
'''
@Function:
 
'''
import os
import shutil
import time

import cv2
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from model_resnet import *
from torchsummary import summary


torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = {'lr': 0.1,
        'prefix': 'RESNET50_IMAGENET_CBAM',
        'arch': 'resnet',
        'checkpoint': './checkpoints/RESNET50_IMAGENET_CBAM_checkpoint.pth.tar',
        'start_epoch': 0,
        'epochs': 10,
        'Batch_size': 4}
best_prec1 = 0


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args['lr'] * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class DataLoad(Dataset):
    def __init__(self, path, regno_data):
        self.path = path
        self.data = regno_data
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((224, 224)),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        img_name = self.path + self.data[index][0] + '.jpg'
        img = cv2.imread(img_name)
        img = self.transform(img).to(device)
        return img, np.array(self.data[index][1], dtype='int64')

    def __len__(self):
        return len(self.data)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.to(device)
        img_val = torch.autograd.Variable(img)
        target_val = torch.autograd.Variable(target)

        output = model(img_val)
        loss = criterion(output, target_val)

        preac1, preac5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(preac1[0], img.size(0))
        top5.update(preac5[0], img.size(0))

        optimizer.zero_grad()
        loss.backward()  # 梯度反传
        optimizer.step()  # 保留梯度

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, prefix):
    filename = './checkpoints/%s_checkpoint.pth.tar' % prefix
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoints/%s_model_best.pth.tar' % prefix)


def main():
    global best_prec1
    from tensorboardX import SummaryWriter
    data_df = pd.read_csv('./data/index.csv')

    train_data = data_df.sample(frac=0.8)
    val_data = data_df.sample(frac=0.2)
    train_data = np.array(train_data).tolist()
    val_data = np.array(val_data).tolist()


    # print(train_data[1][0], train_data[1][1])
    url = './data/img_crop/'
    train_loader = DataLoader(DataLoad(url, train_data), batch_size=args['Batch_size'], shuffle=True)
    val_loader = DataLoader(DataLoad(url, val_data), batch_size=args['Batch_size'])

    # build model
    model = ResidualNet('ImageNet', 50, 5, 'CBAM')
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer = torch.optim.Adam(model.parameters(),)
    data = torch.Tensor(8, 3, 224,224)
    data = data.to(device)
    writer = SummaryWriter()
    writer.add_graph(model=model, input_to_model=data,verbose=True)
    writer.close()
    # summary(model, (3, 224, 224))

    #load the checkpoint
    if args['checkpoint']:
        if os.path.isfile(args['checkpoint']):
            print("===> loading checkpoint '{}'".format(args['checkpoint']))
            checkpoint = torch.load(args['checkpoint'])
            args['start_epoch'] = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("==> loaded checkpoint '{}' (epoch {})"
                  .format(args['checkpoint'], checkpoint['epoch']))
        else:
            print("==> no checkpoint found at '{}'".format(args['checkpoint']))

    for _ in range(args['start_epoch'], args['epochs']):
        adjust_learning_rate(optimizer, _)

        train(train_loader, model, criterion, optimizer, _)

        prec1 = validate(val_loader, model, criterion, _)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': _ + 1,
            'arch': args['arch'],
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args['prefix'])


if __name__ == '__main__':
    main()