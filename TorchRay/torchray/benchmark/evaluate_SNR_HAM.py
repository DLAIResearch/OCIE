import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from torchvision import transforms, datasets
import resnet_multigpu_OCIE as resnet
import cv2

import joblib

""" 
    Here, we evaluate the content heatmap (Grad-CAM heatmap within object bounding box) on the fine-grained dataset.
"""

model_names = ['resnet18', 'resnet50']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-data', default='HAM',metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    metavar='N', help='mini-batch size (default: 96)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-g', '--num-gpus', default=1, type=int,
                    metavar='N', help='number of GPUs to match (default: 4)')
parser.add_argument('--resume', default='env/HAM/model_best.pth_HAM_50--0.01-0.01.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--input_resize', default=224, type=int,
                    metavar='N', help='Resize for smallest side of input (default: 224)')
parser.add_argument('--maxpool', dest='maxpool', action='store_true',
                    help='use maxpool version of the model')
parser.add_argument('--dataset', type=str, default='HAM',
                            help='dataset to use: [imagenet2, cub, aircraft, flowers, cars]')


def main():
    global args
    args = parser.parse_args()

    if args.dataset == 'HAM':
        num_classes = 7



    print("=> creating model '{}' for '{}'".format(args.arch, args.dataset))
    if args.arch.startswith('resnet'):
        model = resnet.__dict__[args.arch](num_classes=num_classes)
    else:
        print('Other archs not supported')
        exit()
    model = model.cuda()
    # model = torch.nn.DataParallel(model).cuda()
    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        print(checkpoint['best_acc1'])
        model.load_state_dict(checkpoint['state_dict'])
    # model = model.cuda()
    # if (not args.resume) and (not args.pretrained):
    #     assert False, "Please specify either the pre-trained model or checkpoint for evaluation"
    # model = torch.nn.DataParallel(model).cuda()
    # cudnn.benchmark = True

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # In the version, we will not resize the images. We feed the full image and use AdaptivePooling before FC.
    # We will resize Gradcam heatmap to image size and compare the actual bbox co-ordinates

    val_dataset = datasets.ImageFolder(root=args.data, transform=data_transform)
    # we set batch size=1 since we are loading full resolution images.
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    box = joblib.load('target_pt_file.pt')
    # box = joblib.load('target_pt_file.pt')
    validate_multi(val_loader, box, model)




def validate_multi(val_loader, box, model):
    batch_time = AverageMeter()
    heatmap_inside_bbox = AverageMeter()

    # switch to evaluate mode

    zero_count = 0
    total_count = 0
    end = time.time()
    for i, (images,  targets) in enumerate(val_loader):
        total_count += 1
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        # we assume batch size == 1 and unwrap the first elem of every list in annotation object
        # annotation = unwrap_dict(annotation)
        # image_size = val_dataset.as_image_size(annotation)
        model.eval()
        output, feats = model(images.cuda(), vanilla_with_feats=True)
        output_gradcam = compute_gradcam(output, feats, targets)
        output_gradcam_np = output_gradcam.data.cpu().numpy()[0]    # since we have batch size==1
        resized_output_gradcam = cv2.resize(output_gradcam_np, (224,224))
        spatial_sum = resized_output_gradcam.sum()
        if spatial_sum <= 0:
            zero_count += 1
            continue

        # resized_output_gradcam is now normalized and can be considered as probabilities

        mask = box[i]
        mask=np.squeeze(mask)
        # mask = mask.cpu().data.numpy()
        signal_sum = np.sum(resized_output_gradcam*mask)
        # 计算噪声区域内的像素和
        mask2=1-mask
        noise_sum = np.sum(resized_output_gradcam*(mask2))

        # 计算信号功率和噪声功率
        signal_power = signal_sum / np.count_nonzero(mask)
        noise_power = noise_sum / np.count_nonzero((mask2))

        # 计算 SNR
        snr = 10 * np.log10(signal_power / noise_power)
        print(snr)
        heatmap_inside_bbox.update(snr)

        if i % 100 == 0:
            print('\nResults after {} examples: '.format(i+1))
            print('Curr % of heatmap inside bbox: {:.4f} ({:.4f})'.format(heatmap_inside_bbox.val,
                                                                             heatmap_inside_bbox.avg))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('\nFinal Results - ')
    print('\n\n% of heatmap inside bbox: {:.4f}'.format(heatmap_inside_bbox.avg))
    print('Zero GC found for {}/{} samples'.format(zero_count, total_count))

    return


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
    spatial_sum1 = gradcam.sum(dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
    gradcam = (gradcam / (spatial_sum1 + eps)) + eps

    return gradcam

def compute_Layercam(output, feats , target):
    """
    Compute the gradcam for the top predicted category
    :param output: the model output before softmax
    :param feats: the feature output from the desired layer to be used for computing Grad-CAM
    :param target: The target category to be used for computing Grad-CAM
    :return:
    """
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
    # k, _, _ = gradcam.size()
    # for i in range(k):
    #     gradcam[i] = (gradcam[i] - gradcam[i].min()) / (gradcam[i].max() - gradcam[i].min())
    # gradcam_mask = gradcam.unsqueeze(1)
    # # print(gradcam_mask[1])
    # gradcam_mask = F.interpolate(gradcam_mask, size=224, mode='bilinear')
    # gradcam_mask = gradcam_mask.squeeze()
    # normalize the gradcam mask to sum to 1
    gradcam_mask_sum = gradcam.sum(dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
    gradcam_mask = (gradcam / (gradcam_mask_sum + eps)) + eps
    return gradcam_mask
def unwrap_dict(dict_object):
    new_dict = {}
    for k, v in dict_object.items():
        if k == 'object':
            new_v_list = []
            for elem in v:
                new_v_list.append(unwrap_dict(elem))
            new_dict[k] = new_v_list
            continue
        if isinstance(v, dict):
            new_v = unwrap_dict(v)
        elif isinstance(v, list) and len(v) == 1:
            new_v = v[0]
        else:
            new_v = v
        new_dict[k] = new_v
    return new_dict


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


if __name__ == '__main__':
    main()
