import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from baselines.ViT.ViT_LRP import vit_base_patch16_224 as ViT
from baselines.ViT.ViT_LRP import deit_base_patch16_224 as DeiT
import datasets as pointing_datasets
from baselines.ViT.ViT_explanation_generator import LRP
from TorchRay.torchray.benchmark.modules.layers_ours import Linear
import joblib
from torchvision import transforms, datasets
""" 
    Here, we evaluate using the stochastic pointing game metriic on imagenet dataset 
    by using the bbox annotations on the val dataset.
"""

model_names = ['ViT', 'DeiT']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-data', metavar='DIR',default='HAM', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 96)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-g', '--num-gpus', default=1, type=int,
                    metavar='N', help='number of GPUs to match (default: 4)')
parser.add_argument('--resume',default='env/HAM_ViT/model_best.pth_mask_ViT_HAM.tar',  type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--input_resize', default=224, type=int,
                    metavar='N', help='Resize for smallest side of input (default: 224)')
parser.add_argument('--dataset', type=str, default='HAM',
                            help='dataset to use: [HAM]')

def main():
    global args
    args = parser.parse_args()
    if args.dataset == 'HAM':
        num_classes = 7

    print("=> creating model '{}' for '{}'".format(args.arch, args.dataset))
    if args.arch.startswith('ViT'):
        model = ViT().cuda()
        model.head = Linear(model.head.in_features, num_classes).cuda()
    else:
        model = DeiT().cuda()
        model.head = Linear(model.head.in_features, num_classes).cuda()

    model = model.cuda()

    # model = torch.nn.DataParallel(model).cuda()
    # model.fc = nn.Linear(512, 10).cuda()
    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        print(checkpoint['best_acc1'])
        model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    if (not args.resume) and (not args.pretrained):
        assert False, "Please specify either the pre-trained model or checkpoint for evaluation"

    cudnn.benchmark = True
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

    attribution_generator = LRP(model)
    validate_multi(val_loader, box, model,attribution_generator)


def validate_multi(val_loader, box, model,attribution_generator):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    expected_value_list = []
    # We take 100 stochastic samples for our evaluation.
    NUM_SAMPLES = 100
    TOLERANCE = 15
    end = time.time()
    for i, (images, targets) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        # index=int(targets)
        # we assume batch size == 1 and unwrap the first elem of every list in annotation object
        # annotation = unwrap_dict(annotation)
        # w, h = val_dataset.as_image_size(annotation)
        class_id = targets[0].item()

        # compute output
        # we have added return_feats=True to get the output as well as layer4 conv feats
        transformer_attribution = attribution_generator.generate_LRP(images.cuda(),
                                                                     method="last_layer_attn",
                                                                     ##method:last_layer_attn/attn_cam/transformer_attribution/rollout
                                                                     index=targets).detach()
        transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
        transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16,
                                                                  mode='bilinear')
        transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
        resized_output_gradcam = (transformer_attribution - transformer_attribution.min()) / (
                transformer_attribution.max() - transformer_attribution.min())
        resized_output_gradcam = cv2.resize(resized_output_gradcam, (224, 224))
        spatial_sum = resized_output_gradcam.sum()
        if spatial_sum <= 0:
            # We ignore images with zero Grad-CAM
            continue
        resized_output_gradcam = resized_output_gradcam / spatial_sum
        mask = box[i]
        mask=np.squeeze(mask)
        gt_mask=torch.tensor(mask.astype(np.bool))
        # Now, we obtain the mask corresponding to the ground truth bounding boxes
        # Skip if all boxes for class_id are marked difficult.

        # output_gradcam is now normalized and can be considered as probabilities
        sample_indices = np.random.choice(np.arange(224*224), NUM_SAMPLES, p=resized_output_gradcam.ravel())
        curr_image_hits = []
        for sample_index in sample_indices:
            sample_x, sample_y = np.unravel_index(sample_index, (224, 224))
            v, u = torch.meshgrid((
                (torch.arange(gt_mask.shape[0],
                              dtype=torch.float32) - sample_x) ** 2,
                (torch.arange(gt_mask.shape[1],
                              dtype=torch.float32) - sample_y) ** 2,
            ))
            accept = (v + u) < TOLERANCE ** 2
            hit = (gt_mask & accept).view(-1).any()
            if hit:
                hit = +1
            else:
                hit = -1
            curr_image_hits.append((hit+1)/2)
        curr_image_hits_arr = np.array(curr_image_hits)
        # we have a bernoulli distribution for the hits, so we compute mean and variance
        pos_prob = float(curr_image_hits_arr.sum())/float(NUM_SAMPLES)
        expected_value_list.append(pos_prob*100)

        if i % 1000 == 0:
            print('\n{} val images:'.format(i+1))
            expected_value_arr = np.array(expected_value_list)
            mean_expectation = expected_value_arr.mean()
            print('Mean - Expected value for 100 stochastic samples/image for hits(1) and misses(0): {}'.format(
                mean_expectation))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    expected_value_arr = np.array(expected_value_list)
    mean_expectation = expected_value_arr.mean()
    print('Mean - Expected value for 100 stochastic samples/image for hits(1) and misses(0): {}'.format(
        mean_expectation))

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
