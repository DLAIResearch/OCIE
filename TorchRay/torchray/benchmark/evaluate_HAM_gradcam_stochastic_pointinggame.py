import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
import resnet_multigpu_cgc_or as resnet
from torchvision import transforms, datasets
import cv2
import joblib


""" 
    Here, we evaluate using the stochastic pointing game metriic on imagenet dataset 
    by using the bbox annotations on the val dataset.
"""

model_names = ['resnet18', 'resnet50', 'alexnet']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-data', metavar='DIR',default='HAM', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
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
parser.add_argument('--resume',default='env/HAM/model_best.pth_HAM_50--0.01-0.01.tar',  type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--input_resize', default=224, type=int,
                    metavar='N', help='Resize for smallest side of input (default: 224)')
parser.add_argument('--dataset', type=str, default='HAM',
                            help='dataset to use: [imagenet2, cub, aircraft, flowers, cars]')

def main():
    global args
    args = parser.parse_args()
    if args.dataset == 'HAM':
        num_classes = 7

    if args.arch.startswith('resnet'):
        model = resnet.__dict__[args.arch](num_classes=num_classes)
    else:
        print('Other archs not supported')
        exit()


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
    val_dataset = datasets.ImageFolder(root=args.data, transform=data_transform)
    # Here, we don't resize the images. Instead, we feed the full image and use AdaptivePooling before FC.
    # We will resize Gradcam heatmap to image size and compare the actual bbox co-ordinates
    box = joblib.load('target_pt_file.pt')
    # we set batch size=1 since we are loading full resolution images.
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    validate_multi(val_loader, box, model)


def validate_multi(val_loader, box, model):
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

        # we assume batch size == 1 and unwrap the first elem of every list in annotation object

        # class_id = targets[0].item()

        # compute output
        # we have added return_feats=True to get the output as well as layer4 conv feats
        output, feats = model(images.cuda(),  vanilla_with_feats=True)
        output_gradcam = compute_gradcam(output, feats, targets)
        output_gradcam_np = output_gradcam.data.cpu().numpy()[0]    # since we have batch size==1
        resized_output_gradcam = cv2.resize(output_gradcam_np, (224, 224))
        spatial_sum = resized_output_gradcam.sum()
        if spatial_sum <= 0:
            # We ignore images with zero Grad-CAM
            continue
        resized_output_gradcam = resized_output_gradcam / spatial_sum

        # Now, we obtain the mask corresponding to the ground truth bounding boxes
        # Skip if all boxes for class_id are marked difficult.
        mask = box[i]
        mask=np.squeeze(mask)
        gt_mask=torch.tensor(mask.astype(np.bool))

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

        if i % 100 == 0:
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


def compute_gradcam(output, feats, target):
    """
    Compute the gradcam for the given target
    :param output:
    :param feats:
    :param: target:
    :return:
    """
    eps = 1e-8
    relu = nn.ReLU(inplace=True)

    target = target.cpu().numpy()
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
    gcam512_1 = dy_dz1 * feats
    gradcam = gcam512_1.sum(dim=1)
    gradcam = relu(gradcam)
    spatial_sum1 = gradcam.sum(dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
    gradcam = (gradcam / (spatial_sum1 + eps)) + eps

    return gradcam


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
