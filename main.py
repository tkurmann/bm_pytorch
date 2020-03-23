from __future__ import print_function, division

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
import time
import sklearn.metrics
from datasets.octhdf5 import OCTHDF5Dataset
from models.resnet import resnet50
#from models.mobilenet import mobilenet_v2
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='OCT Biomarker Classification ')
parser.add_argument('training_data', metavar='TRAINFILE',
                    help='path to training dataset')
parser.add_argument('val_data', metavar='VALFILE',
                    help='path to training dataset')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 8), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')





class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'





def train(train_loader, model, criterion, optimizer, epoch):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Gender Accuracy', ':6.2f')


    # switch to train mode
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))
    model.train()
    end = time.time()
    for i, (sample) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # print()
        images = sample["images"].cuda(0, non_blocking=True)
        labels = sample["labels"].cuda(0, non_blocking=True)


        # compute output
        output = model(images)
        loss = criterion(output, labels)


        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))
        # top1.update(acc1[0], images.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time

        batch_time.update(time.time() - end)
        end = time.time()
        if i % 10 == 0:
            progress.display(i)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Gender Accuracy', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    labels_all = []
    pred_all = []

    with torch.no_grad():
        end = time.time()
        for i, (sample) in enumerate(val_loader):

            images = sample["images"].cuda(0, non_blocking=True)
            labels = sample["labels"].cuda(0, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, labels)
            if(i == 0):
                labels_all = labels.data.cpu().numpy()
                pred_all = output.data.cpu().numpy()
            else:
                labels_all = np.vstack([labels_all, labels.data.cpu().numpy()])
                pred_all = np.vstack([pred_all, output.data.cpu().numpy()])

            #labels_all.append(labels.data.cpu().numpy())
            #pred_all.append(output.data.cpu().numpy())

            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))
            # top1.update(acc1[0], images.size(0))
            # top1.update(acc1[0], images/.size(0))
            # top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        print(labels_all)
        labels_all = np.array(labels_all)
        pred_all = np.array(pred_all)
        print(labels_all.shape)
        print(pred_all.shape)
        roc_macro = sklearn.metrics.roc_auc_score(y_true=labels_all, y_score=pred_all, average='macro')
        roc_micro = sklearn.metrics.roc_auc_score(y_true=labels_all, y_score=pred_all, average='micro')

        map_macro = sklearn.metrics.average_precision_score(y_true=labels_all, y_score=pred_all, average='macro')
        map_micro = sklearn.metrics.average_precision_score(y_true=labels_all, y_score=pred_all, average='micro')

        print(' *roc macro {0} roc micro {1} map macro {2} map micro {3}'.format(roc_macro, roc_micro, map_macro, map_micro))

    return 1






def main():

    args = parser.parse_args()



    image_train_transform = transform_image=transforms.Compose([
                                                             transforms.ToPILImage(),
                                                             transforms.Resize([512,512]),
                                                             transforms.RandomHorizontalFlip(p=0.5),
                                                             transforms.RandomVerticalFlip(p=0.5),
                                                             # transforms.RandomApply([transforms.RandomAffine(degrees=(-90, 90))], p=0.1),
                                                             # transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.0, 0.25))], p=0.1),
                                                             # transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0)]),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    image_val_transform = transform_image=transforms.Compose([
                                                             transforms.ToPILImage(),
                                                             transforms.Resize([512,512]),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    train_dataset = OCTHDF5Dataset(args.training_data, image_set="data/slices",
                                                       label_set="data/markers",
                                                       transform_image=image_train_transform)
    val_dataset = OCTHDF5Dataset(args.val_data, image_set="data/slices",
                                                label_set="data/markers",
                                                transform_image=image_val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None)


    model = resnet50(pretrained=True, num_classes=11)
    model = torch.nn.DataParallel(model).cuda()



    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion= torch.nn.BCEWithLogitsLoss()




    for epoch in range(100):
        train(train_loader, model, criterion, optimizer, epoch)

        score1 = validate(val_loader, model, criterion)






if __name__== "__main__":
  main()
