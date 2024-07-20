import numpy as np
import argparse
import os
import time
import gc
import warnings
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from modules.layers_ours import *
from baselines.ViT.ViT_LRP import deit_base_patch16_224 as vit_LRP
from baselines.ViT.ViT_explanation_generator import LRP
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


model_names = ['resnet18', 'resnet50']
parser = argparse.ArgumentParser(description='PyTorch ImageNet test')
parser.add_argument('--data', metavar='DIR',default='../dataset/imagenet',
                    help='path to dataset')

parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--log_dir', default='logs-imagenet', type=str, metavar='LG_PATH',
                    help='path to write logs (default: logs)')
parser.add_argument('--dataset', type=str, default='imagenet',
                            help='dataset to use: [imagenet, tiny_imagenet]')



best_acc1 = 0


def main():
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(logpath=os.path.join(args.log_dir, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    main_worker(args.gpu, args, logger)

def generate_visualization(attribution_generator,original_image, targets):
    batch_size = original_image.size(0)
    batch_tensor = torch.zeros(batch_size, 224, 224)
    for idx in range(batch_size):

        image = original_image[idx].unsqueeze(0)
        class_index=int(targets[idx])
        transformer_attribution = attribution_generator.generate_LRP(image.cuda(),
                                                                 method="rollout",
                                                                 index=class_index).detach()
        #last_layer_attn/transformer_attribution/
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
        transformer_attribution = transformer_attribution.reshape(224, 224)
        if transformer_attribution.max()==transformer_attribution.min():
           transformer_attribution=0.01
        else:
         transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
                transformer_attribution.max() - transformer_attribution.min())
        batch_tensor[idx]=transformer_attribution
    return batch_tensor
def main_worker(gpu, args, logger):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))
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
    elif args.dataset == 'HAM':
        kwargs = {'num_classes': 7}
        num_classes = 7
    elif args.dataset == 'cars':
        kwargs = {'num_classes': 196}
        num_classes = 196

    # create model

    # initialize ViT pretrained
    model = vit_LRP(pretrained=False).cuda()
    model.head = Linear(model.head.in_features, num_classes).cuda()

    checkpoint = torch.load('../ViT_model/imagenet/model_best.pth-Imagenet-ViT_nofilter.tar')
    print(checkpoint['best_acc1'])
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()

    model.eval()

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
        [batch_time, top1, top5,  f1s],
        logger,
        prefix='Test: ')

    end = time.time()
    for i, (images, targets) in enumerate(val_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)

        attribution_generator = LRP(model)
        orig_gradcam_mask = generate_visualization(attribution_generator,images , targets).cuda()
        images_outputs=model(images)
        target = np.argmax(images_outputs.cpu().data.numpy(), axis=-1)
        threshold = 0.3
        orig_gradcam_mask= (orig_gradcam_mask >= threshold).float()
        copped_image=images[torch.from_numpy(target).cuda() == targets]
        copped_orig_gradcam_mask=orig_gradcam_mask[torch.from_numpy(target).cuda() == targets]
        copped_image=copped_image*copped_orig_gradcam_mask.unsqueeze(1)
        target2=targets[torch.from_numpy(target).cuda() == targets]
        model.eval()
        with torch.no_grad():
          output=model(copped_image)
          acc1, acc5 = accuracy(output, target2, topk=(1, 5))
          class_output = np.argmax(output.cpu(), axis=1)
          f1 = torch.tensor(f1_score(target2.cpu(), class_output, average='weighted'),
                              dtype=torch.float32).cuda()  # measure accuracy and record loss
          top1.update(acc1[0], copped_image.size(0))
          top5.update(acc5[0], copped_image.size(0))
          f1s.update(f1.item(), copped_image.size(0))

          batch_time.update(time.time() - end)
          end = time.time()
          torch.cuda.empty_cache()
          del images
          gc.collect()
          if i % args.print_freq == 0:
                 progress.display(i)

    logger.info(
            ' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}  f1 {f1s.avg:.3f} '
            .format(top1=top1, top5=top5,  f1s=f1s))

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


if __name__ == '__main__':
    main()

