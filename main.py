# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import time
import shutil
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import os.path as osp
import csv
import numpy as np

np.random.seed(1337)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from model import SGN
from data import NTUDataLoaders, AverageMeter
import fit
from util import make_dir, get_num_classes

parser = argparse.ArgumentParser(description='Skeleton-Based Action Recgnition')
fit.add_fit_args(parser)
parser.set_defaults(
    network='SGN',
    dataset = 'NTU',
    case = 0,
    batch_size=64,
    max_epochs=120,
    monitor='val_acc',
    lr=0.001,
    weight_decay=0.0001,
    lr_factor=0.1,
    workers=16,
    print_freq = 20,
    train = 0,
    seg = 20,
    tag='ar',
    mask=[],
    load_dir=None,
    smart_noise=0,
    smart_masking=0,
    group_noise=0,
    naive_noise=0,
    alpha=0.9,
    beta=0.2,
    sigma=0.01,
    total_epsilon=1.0,
    )
args = parser.parse_args()

exclusives = [args.smart_masking, args.smart_noise, args.group_noise, args.naive_noise]
assert sum(exclusives) <= 1, "Only one of smart masking, smart noise, group noise, or naive noise can be applied"
assert sum(exclusives) == 0 or 'maskidx' not in args, "Masking cannot be applied with smart masking, smart noise, group noise, or naive noise"
assert args.smart_masking == 0 or 'maskidx' not in args, "Smart masking cannot be applied with masking"
assert args.smart_noise == 0 or args.alpha > 0, "Alpha must be greater than 0 for smart noise"
assert args.dataset in ['NTU', 'NTU120', 'ETRI'], "Dataset not found"
# assert args.dataset == 'ETRI' and args.tag in ['ar', 'ri'], "ETRI dataset only supports ar and ri tags"

def main():

    args.num_classes = get_num_classes(args.dataset, args.tag)
    model = SGN(args.num_classes, args.dataset, args.seg, args)

    total = get_n_params(model)
    print('The number of parameters: ', total)
    print('The modes is:', args.network)

    if torch.cuda.is_available():
        model = model.cuda()

    criterion = LabelSmoothingLoss(args.num_classes, smoothing=0.1).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.monitor == 'val_acc':
        mode = 'max'
        monitor_op = np.greater
        best = -np.Inf
        str_op = 'improve'
    elif args.monitor == 'val_loss':
        mode = 'min'
        monitor_op = np.less
        best = np.Inf
        str_op = 'reduce'

    scheduler = MultiStepLR(optimizer, milestones=[60, 90, 110], gamma=0.1)
    # Data loading
    ntu_loaders = NTUDataLoaders(args.dataset, args.case, seg=args.seg, tag=args.tag, \
                                    maskidx=args.mask, naive_noise=args.naive_noise==1, \
                                    smart_noise=args.smart_noise==1, smart_masking=args.smart_masking==1, \
                                    group_noise=args.group_noise==1, alpha=args.alpha, beta=args.beta, \
                                    sigma=args.sigma, total_epsilon=args.total_epsilon)
    train_loader = ntu_loaders.get_train_loader(args.batch_size, args.workers)
    val_loader = ntu_loaders.get_val_loader(args.batch_size, args.workers)
    train_size = ntu_loaders.get_train_size()
    val_size = ntu_loaders.get_val_size()

    test_loader = ntu_loaders.get_test_loader(32, args.workers)

    if args.train: print(f'Training on {train_size} samples')
    print(f'Testing on {val_size} samples')

    best_epoch = 0
    if args.load_dir:
        output_dir = os.path.join(args.load_dir)
    else:
        output_dir = make_dir(args.dataset, args.tag)

    save_path = os.path.join(output_dir, args.network)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    checkpoint = osp.join(save_path, '%s_best.pth' % args.case)
    earlystop_cnt = 0
    csv_file = osp.join(save_path, '%s_log.csv' % args.case)
    log_res = list()

    lable_path = osp.join(save_path, '%s_lable.txt'% args.case)
    pred_path = osp.join(save_path, '%s_pred.txt' % args.case)
    
    # Training
    if args.train ==1:
        for epoch in range(args.start_epoch, args.max_epochs):

            print(epoch, optimizer.param_groups[0]['lr'])

            t_start = time.time()
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
            val_loss, val_acc = validate(val_loader, model, criterion)
            log_res += [[train_loss, train_acc.cpu().numpy(),\
                         val_loss, val_acc.cpu().numpy()]]

            print('Epoch-{:<3d} {:.1f}s\t'
                  'Train: loss {:.4f}\taccu {:.4f}\tValid: loss {:.4f}\taccu {:.4f}'
                  .format(epoch + 1, time.time() - t_start, train_loss, train_acc, val_loss, val_acc))

            current = val_loss if mode == 'min' else val_acc

            ####### store tensor in cpu
            current = current.cpu()

            if monitor_op(current, best):
                print('Epoch %d: %s %sd from %.4f to %.4f, '
                      'saving model to %s'
                      % (epoch + 1, args.monitor, str_op, best, current, checkpoint))
                best = current
                best_epoch = epoch + 1
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best': best,
                    'monitor': args.monitor,
                    'optimizer': optimizer.state_dict(),
                }, checkpoint)
                earlystop_cnt = 0
            else:
                print('Epoch %d: %s did not %s' % (epoch + 1, args.monitor, str_op))
                earlystop_cnt += 1

            scheduler.step()

        print('Best %s: %.4f from epoch-%d' % (args.monitor, best, best_epoch))
        with open(csv_file, 'w') as fw:
            cw = csv.writer(fw)
            cw.writerow(['loss', 'acc', 'val_loss', 'val_acc'])
            cw.writerows(log_res)
        print('Save train and validation log into into %s' % csv_file)

    ### Test
    args.train = 0
    model = SGN(args.num_classes, args.dataset, args.seg, args)
    model = model.cuda()
    test(test_loader, model, checkpoint, lable_path, pred_path)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    acces = AverageMeter()
    model.train()

    for i, (inputs, target) in enumerate(train_loader):

        output = model(inputs.cuda())
        target = target.cuda()
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc = accuracy(output.data, target)
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        # backward
        optimizer.zero_grad()  # clear gradients out before each mini-batch
        loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            print('Epoch-{:<3d} {:3d} batches\t'
                  'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'accu {acc.val:.3f} ({acc.avg:.3f})'.format(
                   epoch + 1, i + 1, loss=losses, acc=acces))

    return losses.avg, acces.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    acces = AverageMeter()
    model.eval()

    for i, (inputs, target) in enumerate(val_loader):
        with torch.no_grad():
            output = model(inputs.cuda())
        target = target.cuda()
        with torch.no_grad():
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc = accuracy(output.data, target)
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))

    return losses.avg, acces.avg


def test(test_loader, model, checkpoint, lable_path, pred_path):
    """Test loop that prints final top-1 accuracy. Also prints/stores top-5 
       accuracy if tag is ar or ri.
    """
    top1_meter = AverageMeter()
    top5_meter = AverageMeter() if args.tag in ['ar', 'ri'] else None

    # Load best model
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    model.eval()

    label_output = []
    pred_output = []

    t_start = time.time()
    for i, (inputs, target) in enumerate(test_loader):
        with torch.no_grad():
            outputs = model(inputs.cuda())
            # For multi-crop or multiple segment inference, if used
            outputs = outputs.view((-1, inputs.size(0)//target.size(0), outputs.size(1)))
            outputs = outputs.mean(1)

        label_output.append(target.numpy())
        pred_output.append(outputs.cpu().numpy())

        # Compute top-1 & top-5
        acc1, acc5 = accuracy_topk(outputs, target.cuda(), topk=(1, 5))
        top1_meter.update(acc1.item(), inputs.size(0))
        if top5_meter is not None:
            top5_meter.update(acc5.item(), inputs.size(0))

    # Write label & pred to disk
    label_output = np.concatenate(label_output, axis=0)
    np.savetxt(lable_path, label_output, fmt='%d')
    pred_output = np.concatenate(pred_output, axis=0)
    np.savetxt(pred_path, pred_output, fmt='%f')

    elapsed = time.time() - t_start

    if top5_meter is not None:
        print('Test: top1 accuracy = {:.3f}, top5 accuracy = {:.3f}, time = {:.2f}s'
              .format(top1_meter.avg, top5_meter.avg, elapsed))
    else:
        print('Test: accuracy = {:.3f}, time = {:.2f}s'
              .format(top1_meter.avg, elapsed))


def accuracy_topk(output, target, topk=(1,)):
    """Compute top-k accuracies. Returns a tuple of 
       accuracies in the same order as topk."""
    maxk = max(topk)
    batch_size = target.size(0)

    # Get indices of top-k predictions
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # shape: [batch_size, maxk]
    pred = pred.t()  # shape: [maxk, batch_size]
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # Flatten first k rows of correct into a single vector
        # and count how many are true
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return tuple(res)


def accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.view(-1).float().sum(0, keepdim=True)

    return correct.mul_(100.0 / batch_size)

def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

if __name__ == '__main__':
    main()
    
