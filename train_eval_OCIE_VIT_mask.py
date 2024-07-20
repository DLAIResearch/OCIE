import numpy as np
import argparse
import os
import random
import shutil
import time
import sys
import warnings
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from modules.layers_ours import *
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dataset.imagefolder_OCIE_mask import ImageFolder
import logging
from sklearn.metrics import recall_score, f1_score,precision_score
from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP_ViT
from baselines.ViT.ViT_explanation_generator import LRP,Baselines
def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r",encoding='UTF-8') as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


model_names = ['ViT']
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',default='',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=True,action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--save_dir', default='checkpoint', type=str, metavar='SV_PATH',
                    help='path to save checkpoints (default: none)')
parser.add_argument('--log_dir', default='logs-VIT-imagenet-mask_O', type=str, metavar='LG_PATH',
                    help='path to write logs (default: logs)')
parser.add_argument('--dataset', type=str, default='imagenet',
                            help='dataset to use: [imagenet, cars,cub,HAM]')

parser.add_argument('--lambda', default=0.01, type=float,
                    metavar='LAM', help='lambda hyperparameter for GCAM loss', dest='lambda_val')
parser.add_argument('--beta', default=0.01, type=float,
                    metavar='LAM', help='lambda hyperparameter for GCAM loss', dest='beta_val')

best_acc1 = 0


def main():
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(logpath=os.path.join(args.log_dir, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')


    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, logger)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, predicted, target):
        """
        Args:
            predicted (torch.Tensor): 模型的预测值
            target (torch.Tensor): 目标值
        Returns:
            torch.Tensor: MSE损失值
        """
        b=predicted.size(0)
        total_loss = 0
        for i in range(b):
          mse_loss =1-F.cosine_similarity(predicted[i], target[i], dim=0)
          total_loss+=mse_loss
        return total_loss
class energy_point_game_Loss(nn.Module):
    def __init__(self, ):
        super(energy_point_game_Loss, self).__init__()

    def forward(self, bbox, saliency_map):
        # saliency_map=saliency_map.cpu().detach().numpy()
        total_loss=0
        b=saliency_map.size(0)
        for i in range (b):
            mask_bbox = saliency_map[i] * bbox[i]
            energy_bbox = torch.sum(mask_bbox)
            energy_whole = torch.sum(saliency_map[i])
            proportion = energy_bbox / energy_whole
            loss = 1 - proportion  # 使用1减去比例作为损失函数
            total_loss+=loss
        return total_loss
def main_worker(gpu, ngpus_per_node, args, logger):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    kwargs = {}
    num_classes = 100
    val_dir_name = 'val'
    if args.dataset == 'imagenet':
        kwargs = {'num_classes': 100}
        num_classes = 100
    elif args.dataset == 'cub':
        kwargs = {'num_classes': 200}
        num_classes = 200
        val_dir_name = 'val'
    elif args.dataset == 'cars':
        kwargs = {'num_classes': 196}
        num_classes = 196
    elif args.dataset == 'HAM':
        kwargs = {'num_classes': 7}
        num_classes = 7

    # create model
    if args.pretrained:
        if  args.arch == 'ViT':
            logger.info("=> using pre-trained model 'ViT-base'")
            model = vit_LRP_ViT(pretrained=True)
        else:
            print('Arch not supported!!')
            exit()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    #

    model.head = Linear(model.head.in_features, num_classes)
    #Parallel training
    #model = torch.nn.DataParallel(model).cuda()
    print(model)
    logger.info(model)

    # define loss function (criterion) and optimizer
    xent_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    align_loss=MSELoss().cuda(args.gpu)
    engry_criterion=energy_point_game_Loss().cuda(args.gpu)
    cudnn.benchmark = True
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, val_dir_name)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = ImageFolder(traindir)   # transforms are handled within the implementation

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_batch_size = args.batch_size
    
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, engry_criterion, xent_criterion, args, logger)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, engry_criterion , xent_criterion,optimizer, epoch, args, logger)

        # evaluate on validation set
        acc1 = validate(val_loader, model, engry_criterion, xent_criterion, args, logger)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.save_dir)

def generate_visualization(attribution_generator,original_image, class_index=None):
    batch_size = original_image.size(0)
    batch_tensor = torch.zeros(batch_size, 224, 224)
    for idx in range(batch_size):
        image = original_image[idx].unsqueeze(0)

        transformer_attribution = attribution_generator.generate_LRP(image.cuda(),
                                                                 method="last_layer_attn",
                                                                 index=class_index).detach()
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
        transformer_attribution = transformer_attribution.reshape(224, 224)
        transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
                transformer_attribution.max() - transformer_attribution.min())
        batch_tensor[idx]=transformer_attribution
    return batch_tensor
def train(train_loader, model, engry_criterion, xent_criterion, optimizer, epoch, args, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    xe_losses = AverageMeter('XE Loss', ':.4e')
    engry_criterion_losses = AverageMeter('engry_criterion_loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, xe_losses, engry_criterion_losses,losses, top1, top5],
        logger,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    train_len = len(train_loader)
    train_iter = iter(train_loader)
    end = time.time()
    for i in range(train_len):
        xe_images,images, targets,pred= train_iter.__next__()
        data_time.update(time.time() - end)
        images = images.cuda(args.gpu, non_blocking=True)

        xe_images = xe_images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)
        pred=pred.cuda(args.gpu, non_blocking=True)

        images_outputs=model(images)
        xe_images_output=model(xe_images)
        xe_loss = xent_criterion(images_outputs, targets.cuda()) + xent_criterion(xe_images_output, targets.cuda())
        attribution_generator = LRP(model)
        aug_gradcam_mask = generate_visualization(attribution_generator, images, class_index=None).cuda()
        engry_criterion_loss2 = engry_criterion(pred.float(), aug_gradcam_mask)
        xe_loss = xe_loss.mean()
        engry_criterion_loss2 = engry_criterion_loss2.mean()

        loss = xe_loss + args.lambda_val * engry_criterion_loss2

        # measure accuracy and record loss
        acc1, acc5 = accuracy(images_outputs, targets, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        xe_losses.update(xe_loss.item(), images.size(0))
        engry_criterion_losses.update(engry_criterion_loss2.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, engry_criterion, criterion, args, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    recalls = AverageMeter('recall', ':.4e')
    f1s = AverageMeter('f1', ':.4e')
    precisions = AverageMeter('precision', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5, recalls, f1s, precisions],
        logger,
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, targets) in enumerate(val_loader):
            if args.gpu is not None:
              images = images.cuda(args.gpu, non_blocking=True)
              targets = targets.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, targets)
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            class_output = np.argmax(output.cpu(), axis=1)
            recall = torch.tensor(recall_score(targets.cpu(), class_output, average='weighted'),
                                  dtype=torch.float32).cuda()
            f1 = torch.tensor(f1_score(targets.cpu(), class_output, average='weighted'),
                              dtype=torch.float32).cuda()  # measure accuracy and record loss
            precision = torch.tensor(precision_score(targets.cpu(), class_output, average='weighted'),
                                     dtype=torch.float32).cuda()
            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            recalls.update(recall.item(), images.size(0))
            f1s.update(f1.item(), images.size(0))
            precisions.update(precision.item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        logger.info(
            ' * Acc@1 {top1.avg:.4f} Acc@5 {top5.avg:.4f} recall {recalls.avg:.4f} f1 {f1s.avg:.4f} precision {precisions.avg:.4f}'
            .format(top1=top1, top5=top5, recalls=recalls, f1s=f1s, precisions=precisions))

    return top1.avg


def save_checkpoint(state, is_best, save_dir):
    epoch = state['epoch']
    filename = 'checkpoint_' + str(epoch).zfill(3) + '.pth.tar'
    save_path = os.path.join(save_dir, filename)
    torch.save(state, save_path)
    if is_best:
        best_filename = 'model_best.pth_mask_ViT_imagenet.tar'
        best_save_path = os.path.join(save_dir, best_filename)
        shutil.copyfile(save_path, best_save_path)


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
    def __init__(self, num_batches, meters,logger, prefix="" ):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]

        self.logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


if __name__ == '__main__':
    main()

