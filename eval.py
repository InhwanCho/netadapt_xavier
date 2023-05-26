from argparse import ArgumentParser
import os
import time
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import pickle

import nets as models
import functions as fns

_NUM_CLASSES = 10

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def compute_topk_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    
def compute_accuracy(output, target):
    output = output.argmax(dim=1)
    acc = 0.0
    acc = torch.sum(target == output).item()
    acc = acc/output.size(0)*100
    return acc


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


def eval(test_loader, model, args):
    batch_time = AverageMeter()
    acc = AverageMeter()

    # switch to eval mode
    model.eval()
    #with torch.no_grad():
    end = time.time()
    for i, (images, target) in enumerate(test_loader):
        if not args.no_cuda:
            images = images.cuda()
            target = target.cuda()
        output = model(images)
        batch_acc = compute_accuracy(output, target)
        acc.update(batch_acc, images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Update statistics
        estimated_time_remained = batch_time.get_avg()*(len(test_loader)-i-1)
        fns.update_progress(i, len(test_loader), 
            ESA='{:8.2f}'.format(estimated_time_remained)+'s',
            acc='{:4.2f}'.format(float(batch_acc))
            )
    print()
    print('Test accuracy: {:4.2f}% (time = {:8.2f}s)'.format(
            float(acc.get_avg()), batch_time.get_avg()*len(test_loader)))
    print('===================================================================')
    return float(acc.get_avg())
            

if __name__ == '__main__':
    # Parse the input arguments.
    arg_parser = ArgumentParser()
    arg_parser.add_argument('data', metavar='DIR', help='path to dataset')
    arg_parser.add_argument('--dataset',default = 'cifar10')
    arg_parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    arg_parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
    arg_parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='batch size (default: 128)')
    arg_parser.add_argument('--dir', type=str, default='models/', dest='save_dir', 
                            help='path to save models (default: models/')
    arg_parser.add_argument('--no-cuda', action='store_true', default=False, dest='no_cuda',
                    help='disables training on GPU')
    
    args = arg_parser.parse_args()
    print(args)
    
    # Data loader
    if args.dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(root=args.data, train=False, download=True,
            transform=transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]))
        num_class = 10
    elif args.dataset == 'cifar100':
        test_dataset = datasets.CIFAR100(root=args.data, train=False, download=True,
            transform=transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ]))
        num_class = 100
    
    # Image Preprocessing 
    elif args.dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        if args.arch =='efficientnet_es_postech':
            normalize = transforms.Normalize(mean=[0.5019, 0.5019, 0.5019], std=[0.5019, 0.5019, 0.5019])
       
        test_dataset = datasets.folder.ImageFolder(root='/home/keti/workspace/Dataset/imagenet/val/',
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,]))
        num_class = 1000

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    # Network
    model_arch = args.arch
    cudnn.benchmark = True
    num_classes = num_class
    model = models.__dict__[model_arch](num_classes=num_classes)

    if not args.no_cuda:
        model = model.cuda()          
    
    # Evaluation
    filename = os.path.join(args.save_dir)
    #torch.save(model, filename)
    model = torch.load(filename) 
   
    #print(model)
        
    best_acc = eval(test_loader, model, args)
    print('Testing accuracy:', best_acc)