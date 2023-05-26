from argparse import ArgumentParser
import os
import time
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn

import nets as models
import functions as fns

_NUM_CLASSES = 10

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
   

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def get_avg(self):
        return self.avg
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    
def compute_accuracy(output, target):
    output = output.argmax(dim=1)
    acc = 0.0
    acc = torch.sum(target == output).item()
    acc = acc/output.size(0)*100
    return acc
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(train_loader, model, criterion, optimizer, epoch, args, num_classes):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    
    print('===================================================================')
    end = time.time()
    
    for i, (images, target) in enumerate(train_loader):
        target.unsqueeze_(1)
        target_onehot = torch.FloatTensor(target.shape[0], num_classes)
        target_onehot.zero_()
        target_onehot.scatter_(1, target, 1)
        target.squeeze_(1)
        
        if not args.no_cuda:
            images = images.cuda()
            target_onehot = target_onehot.cuda()
            target = target.cuda()

        # compute output and loss
        output = model(images)
        if args.dataset == 'cifar10':
            loss = criterion(output, target_onehot)
        else:
            loss = criterion(output, target)
        
        # measure accuracy and record loss
        batch_acc = compute_accuracy(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        acc.update(batch_acc, images.size(0))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Update statistics
        estimated_time_remained = batch_time.get_avg()*(len(train_loader)-i-1)
        fns.update_progress(i, len(train_loader), 
            ESA='{:8.2f}'.format(estimated_time_remained)+'s',
            loss='{:4.2f}'.format(loss.item()),
            acc='{:4.2f}%'.format(float(batch_acc)),
            top1='{:4.2f}%'.format(float(acc1[0])),
            top5='{:4.2f}%'.format(float(acc5[0]))
            )

    print()
    print('Finish epoch {}: time = {:8.2f}s, loss = {:4.2f}, acc = {:4.2f}%, top1 = {:4.2f}%, top5 = {:4.2f}%'.format(
            epoch+1, batch_time.get_avg()*len(train_loader), 
            float(losses.get_avg()), float(acc.get_avg()),
            float(top1.get_avg()),float(top5.get_avg())))
    print('===================================================================')
    return


def eval(test_loader, model, args):
    batch_time = AverageMeter()
    acc = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to eval mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            if not args.no_cuda:
                images = images.cuda()
                target = target.cuda()
            output = model(images)
            batch_acc = compute_accuracy(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            acc.update(batch_acc, images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            # Update statistics
            estimated_time_remained = batch_time.get_avg()*(len(test_loader)-i-1)
            fns.update_progress(i, len(test_loader), 
                ESA='{:8.2f}'.format(estimated_time_remained)+'s',
                acc='{:4.2f}'.format(float(batch_acc)),
                top1='{:4.2f}%'.format(float(acc1[0])),
                top5='{:4.2f}%'.format(float(acc5[0]))
                )
    print()
    print('Test accuracy: {:4.2f}%, top1 = {:4.2f}%, top5 = {:4.2f}%, (time = {:8.2f}s)'.format(
            float(acc.get_avg()), float(top1.get_avg()),float(top5.get_avg()), batch_time.get_avg()*len(test_loader)))
    print('===================================================================')
    return float(acc.get_avg())
            

if __name__ == '__main__':
    # Parse the input arguments.
    arg_parser = ArgumentParser()
    arg_parser.add_argument('data', metavar='DIR', help='path to dataset')
    arg_parser.add_argument('--dataset', type=str, default='imagenet', help='dataset cifar10 or cifar100')
    arg_parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
    arg_parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run (default: 150)')
    arg_parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    arg_parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
    arg_parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='batch size (default: 128)')
    arg_parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (defult: 0.1)', dest='lr')
    arg_parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
    arg_parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)',
                    dest='weight_decay')
    arg_parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    arg_parser.add_argument('--dir', type=str, default='models/', dest='save_dir', 
                            help='path to save models (default: models/')
    arg_parser.add_argument('--no-cuda', action='store_true', default=False, dest='no_cuda',
                    help='disables training on GPU')
    arg_parser.add_argument('--pretrained', action='store_true', default=False)
    args = arg_parser.parse_args()
    print(args)
    
    path = os.path.dirname(args.save_dir)
    if not os.path.exists(path):
        os.makedirs(path)
        print('Create new directory `{}`'.format(path))        

    # Data loader
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.data, train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4), 
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]))
        
        test_dataset = datasets.CIFAR10(root=args.data, train=False, download=True,
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]))
        num_class = 10
        criterion = nn.BCEWithLogitsLoss()
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=args.data, train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
            ]))
        
        test_dataset = datasets.CIFAR100(root=args.data, train=False, download=True,
        transform=transforms.Compose([
                #transforms.Resize(32), 
                transforms.ToTensor(),
                transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), 
                                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        ]))
        num_class = 100
        criterion = nn.CrossEntropyLoss()
    elif args.dataset == 'imagenet':
        from PIL import Image
        if args.arch == 'efficientnet':
            interpolation = Image.BICUBIC
        else:
            interpolation = Image.BILINEAR
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        train_dataset = datasets.folder.ImageFolder(root='/home/keti/workspace/Dataset/imagenet/train/',
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224,interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,]))
        test_dataset = datasets.folder.ImageFolder(root='/home/keti/workspace/Dataset/imagenet/val/',
        transform=transforms.Compose([
            transforms.Resize(256,interpolation=interpolation),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,]))
        num_class = 1000
        criterion = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    # Network
    cudnn.benchmark = True
    num_classes = num_class
    model_arch = args.arch
    model = models.__dict__[model_arch](pretrained = args.pretrained, num_classes=num_classes)
    
    if not args.no_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))
            model = torch.load(args.resume)
            model = torch.nn.DataParallel(model)
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Train & evaluation
    best_acc = 0
    filename = os.path.join(args.save_dir)
    
    for epoch in range(args.start_epoch, args.epochs):
        print('Epoch [{}/{}]'.format(epoch+1, args.epochs - args.start_epoch))
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, num_classes)
        acc = eval(test_loader, model, args)
        
        if acc > best_acc:
            torch.save(model, filename)
            best_acc = acc
            print('Save model: ' + filename)
        print(' ')
    print('Best accuracy:', best_acc)
    
    model = torch.load(filename)
    print(model)
        
    best_acc = eval(test_loader, model, args)
    print('Best accuracy:', best_acc)