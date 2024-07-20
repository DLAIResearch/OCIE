import argparse
import time

import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from baselines.ViT.ViT_LRP import vit_base_patch16_224 as ViT
from baselines.ViT.ViT_LRP import deit_base_patch16_224 as DeiT
# import sys
# sys.path.append('../TorchRay/torchray/benchmark/baselines')
import datasets as pointing_datasets
from baselines.ViT.ViT_explanation_generator import LRP
from modules.layers_ours import Linear

""" 
    Here, we evaluate the content heatmap (Grad-CAM heatmap within object bounding box) on the fine-grained dataset.
"""

model_names = ['ViT', 'DeiT']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-data', default='env/imagenet/',metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 96)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-g', '--num-gpus', default=1, type=int,
                    metavar='N', help='number of GPUs to match (default: 4)')
parser.add_argument('--resume', default='env/imagenet_ViT/model_best.pth-imagenet-0.2-no.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--input_resize', default=224, type=int,
                    metavar='N', help='Resize for smallest side of input (default: 224)')
parser.add_argument('--maxpool', dest='maxpool', action='store_true',
                    help='use maxpool version of the model')
parser.add_argument('--dataset', type=str, default='imagenet',
                            help='dataset to use: [imagenet2, cub, cars]')


def main():
    global args
    args = parser.parse_args()

    if args.dataset == 'cub':
        num_classes = 200
    elif args.dataset == 'aircraft':
        num_classes = 90
    elif args.dataset == 'imagenet':
        num_classes = 100
    elif args.dataset == 'cars':
        num_classes = 196
    elif args.dataset == 'HAM':
        num_classes = 7


    print("=> creating model '{}' for '{}'".format(args.arch, args.dataset))
    if args.arch.startswith('ViT'):
        model = ViT().cuda()
        model.head = Linear(model.head.in_features, num_classes).cuda()
    else:
        model = DeiT().cuda()
        model.head = Linear(model.head.in_features, num_classes).cuda()
    model = model.cuda()
    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        print(checkpoint['best_acc1'])
        model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    if (not args.resume) and (not args.pretrained):
        assert False, "Please specify either the pre-trained model or checkpoint for evaluation"

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    # In the version, we will not resize the images. We feed the full image and use AdaptivePooling before FC.
    # We will resize Gradcam heatmap to image size and compare the actual bbox co-ordinates
    val_dataset = pointing_datasets.ImageNetDetection(args.data,
                                                      transform=transforms.Compose([
                                                          transforms.Resize([224,224]),
                                                          transforms.ToTensor(),
                                                          normalize,
                                                        ]))

    # we set batch size=1 since we are loading full resolution images.
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    attribution_generator = LRP(model)
    validate_multi(val_loader, val_dataset, model,attribution_generator)


def validate_multi(val_loader, val_dataset, model,attribution_generator):
    batch_time = AverageMeter()
    heatmap_inside_bbox = AverageMeter()

    # switch to evaluate mode
    zero_count = 0
    total_count = 0
    end = time.time()
    for i, (images, annotation, targets) in enumerate(val_loader):
        total_count += 1
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        # targets=int(targets)
        # we assume batch size == 1 and unwrap the first elem of every list in annotation object
        annotation = unwrap_dict(annotation)
        image_size = val_dataset.as_image_size(annotation)
        model.eval()
        transformer_attribution = attribution_generator.generate_LRP(images.cuda(),
                                                                     method="rollout",
                                                                     ##method:last_layer_attn/attn_cam/transformer_attribution/rollout
                                                                     index=targets).detach()
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16,
                                                                  mode='bilinear')
        transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
        resized_output_gradcam = (transformer_attribution - transformer_attribution.min()) / (
                transformer_attribution.max() - transformer_attribution.min())
        resized_output_gradcam = cv2.resize(resized_output_gradcam, image_size)
        spatial_sum = resized_output_gradcam.sum()
        if spatial_sum <= 0:
            zero_count += 1
            continue

        # resized_output_gradcam is now normalized and can be considered as probabilities
        resized_output_gradcam = resized_output_gradcam / spatial_sum

        mask = pointing_datasets.imagenet_as_mask(annotation, targets[0].item())

        mask = mask.type(torch.ByteTensor)
        mask = mask.cpu().data.numpy()

        gcam_inside_gt_mask = mask * resized_output_gradcam
        # Now we sum the heatmap inside the object bounding box
        total_gcam_inside_gt_mask = gcam_inside_gt_mask.sum()
        heatmap_inside_bbox.update(total_gcam_inside_gt_mask)

        if i % 1000 == 0:
            print('\nResults after {} examples: '.format(i+1))
            print('Curr % of heatmap inside bbox: {:.4f} ({:.4f})'.format(heatmap_inside_bbox.val * 100,
                                                                             heatmap_inside_bbox.avg * 100))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('\nFinal Results - ')
    print('\n\n% of heatmap inside bbox: {:.4f}'.format(heatmap_inside_bbox.avg * 100))
    print('Zero GC found for {}/{} samples'.format(zero_count, total_count))

    return


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
