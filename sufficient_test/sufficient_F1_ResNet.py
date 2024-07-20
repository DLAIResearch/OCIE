import numpy as np
import argparse
import os
import random
import time
import gc
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.resnet_test as resnet
import logging

from sklearn.metrics import f1_score

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


model_names = ['resnet18' , 'resnet50']
parser = argparse.ArgumentParser(description='PyTorch ImageNet test')
parser.add_argument('--data', metavar='DIR',default='../dataset/imagenet',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128)')
parser.add_argument('-p', '--print-freq', default=5, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--log_dir', default='logs', type=str, metavar='LG_PATH',
                    help='path to write logs (default: logs)')
parser.add_argument('--dataset', type=str, default='imagenet',
                            help='dataset to use: [imagenet, tiny_imagenet]')



best_acc1 = 0


def main():
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(logpath=os.path.join(args.log_dir, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)
    main_worker(args.gpu, args, logger)

def main_worker(gpu, args, logger):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    num_classes = 100
    val_dir_name = 'val'
    if args.dataset == 'imagenet':
        kwargs = {'num_classes': 100}
        num_classes = 100
    elif args.dataset == 'cub':
        kwargs = {'num_classes': 200}
        num_classes = 200
        val_dir_name = 'val'
    elif args.dataset == 'HAM':
        kwargs = {'num_classes': 7}
        num_classes = 7
    elif args.dataset == 'cars':
        kwargs = {'num_classes': 196}
        num_classes = 196

    # create model

    if args.arch == 'resnet18':
        logger.info("=> using pre-trained model 'resnet18'")
        model = resnet.resnet18(pretrained=False)
    elif args.arch == 'resnet50':
        logger.info("=> using pre-trained model 'resnet50'")
        model = resnet.resnet50(pretrained=False)

    # model = torch.nn.DataParallel(model)
    if args.arch == 'resnet18':
        model.fc = nn.Linear(512, num_classes)
    elif args.arch == 'resnet50':
        model.fc = nn.Linear(2048, num_classes)
    # if args.arch == 'resnet18':
    #     model.module.fc = nn.Linear(512, num_classes)
    # elif args.arch == 'resnet50':
    #     model.module.fc = nn.Linear(2048, num_classes)
    model = model.cuda()
    logger.info(model)
    state_dict = torch.load('../models/model_best.pth_imagenet-baseline-50.tar')
    model.load_state_dict(state_dict["state_dict"])
    model = model.cuda()


    cudnn.benchmark = True

    valdir = os.path.join(args.data, val_dir_name)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
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

    validate(val_loader, model, args, logger)

def validate(val_loader, model, args, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    f1s = AverageMeter('f1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5, f1s],
        logger,
        prefix='Test: ')
    model.eval()
    end = time.time()
    for i, (images, targets) in enumerate(val_loader):

        images = images.cuda(args.gpu, non_blocking=True)
        targets = targets.cuda(args.gpu, non_blocking=True)
        output, feats = model(images.cuda(), vanilla_with_feats=True)
        target = np.argmax(output.cpu().data.numpy(), axis=-1)
        output_gradcam = compute_Layercam(output, feats,torch.from_numpy(target).cuda())


        threshold = 0.2 #or 0.3
        orig_gradcam_mask = (output_gradcam >= threshold).float()
        copped_image = images[torch.from_numpy(target).cuda() == targets]
        copped_orig_gradcam_mask = orig_gradcam_mask[torch.from_numpy(target).cuda() == targets]
        copped_image = copped_image * copped_orig_gradcam_mask.unsqueeze(1)
        target2 = targets[torch.from_numpy(target).cuda() == targets]

        with torch.no_grad():
            output2,feats2 = model(copped_image)
            acc1, acc5 = accuracy(output2, target2, topk=(1, 5))
            class_output = np.argmax(output2.cpu(), axis=1)
            f1 = torch.tensor(f1_score(target2.cpu(), class_output, average='weighted'),
                              dtype=torch.float32).cuda()  # measure accuracy and record loss
            # measure accuracy and record loss
            top1.update(acc1[0], copped_image.size(0))
            top5.update(acc5[0], copped_image.size(0))
            f1s.update(f1.item(), copped_image.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            torch.cuda.empty_cache()
            del images
            # del cropped_image
            gc.collect()
            if i % args.print_freq == 0:
                progress.display(i)

    logger.info(
        ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}  f1 {f1s.avg:.3f} '
            .format(top1=top1, top5=top5, f1s=f1s))

    return top1.avg


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
def compute_gradcam(output, feats, target):
    """
    Compute the gradcam for the top predicted category
    :param output:
    :param feats:
    :return:
    """
    eps = 1e-8
    relu = nn.ReLU(inplace=True)

    target = target.cpu().numpy()
    # target = np.argmax(output.cpu().data.numpy(), axis=-1)
    one_hot = np.zeros((output.shape[0], output.shape[-1]), dtype=np.float32)
    indices_range = np.arange(output.shape[0])
    one_hot[indices_range, target[indices_range]] = 1
    one_hot = torch.from_numpy(one_hot)
    one_hot.requires_grad = True

    # Compute the Grad-CAM for the original image
    one_hot_cuda = torch.sum(one_hot.cuda() * output)
    dy_dz1, = torch.autograd.grad(one_hot_cuda, feats, grad_outputs=torch.ones(one_hot_cuda.size()).cuda(),
                                  retain_graph=True, create_graph=True)
    dy_dz_sum1 = dy_dz1.sum(dim=2).sum(dim=2)
    gcam512_1 = dy_dz_sum1.unsqueeze(-1).unsqueeze(-1) * feats
    gradcam = gcam512_1.sum(dim=1)
    gradcam = relu(gradcam)
    gradcam_mask = gradcam.unsqueeze(1)
    # print(gradcam_mask[1])
    gradcam_mask = F.interpolate(gradcam_mask, size=224, mode='bilinear')
    gradcam_mask = gradcam_mask.squeeze()
    # normalize the gradcam mask to sum to 1
    gradcam_mask_sum = gradcam_mask.sum(dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
    gradcam_mask = (gradcam_mask / (gradcam_mask_sum + eps)) + eps
    gradcam = (gradcam_mask - gradcam_mask.min()) / (
            gradcam_mask.max() - gradcam_mask.min())
    return gradcam

def compute_Layercam(output, feats , target):
    """
    Compute the gradcam for the top predicted category
    :param output: the model output before softmax
    :param feats: the feature output from the desired layer to be used for computing Grad-CAM
    :param target: The target category to be used for computing Grad-CAM
    :return:
    """
    target = target.cpu().numpy()
    eps = 1e-8
    relu = nn.ReLU(inplace=True)
    one_hot = np.zeros((output.shape[0], output.shape[-1]), dtype=np.float32)
    indices_range = np.arange(output.shape[0])
    one_hot[indices_range, target[indices_range]] = 1
    one_hot = torch.from_numpy(one_hot)
    one_hot.requires_grad = True

    # Compute the Grad-CAM for the original image
    one_hot_cuda = torch.sum(one_hot.cuda() * output)
    dy_dz1, = torch.autograd.grad(one_hot_cuda, feats, grad_outputs=torch.ones(one_hot_cuda.size()).cuda(),
                                    retain_graph=True, create_graph=True)
    # Changing to dot product of grad and features to preserve grad spatial locations
    gcam512_1 = relu(dy_dz1) * feats
    gradcam = gcam512_1.sum(dim=1)
    gradcam = relu(gradcam)
    gradcam_mask = gradcam.unsqueeze(1)
    # print(gradcam_mask[1])
    gradcam_mask = F.interpolate(gradcam_mask, size=224, mode='bilinear')
    gradcam_mask = gradcam_mask.squeeze()
    # normalize the gradcam mask to sum to 1
    gradcam_mask_sum = gradcam_mask.sum(dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
    gradcam_mask = (gradcam_mask / (gradcam_mask_sum + eps)) + eps
    gradcam = (gradcam_mask - gradcam_mask.min()) / (
            gradcam_mask.max() - gradcam_mask.min())
    return gradcam


if __name__ == '__main__':
    main()

