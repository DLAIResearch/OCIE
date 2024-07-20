import torch
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.utils as vutils
def compute_gradcam(output, feats , target, relu):
    """
    Compute the gradcam for the top predicted category
    :param output: the model output before softmax
    :param feats: the feature output from the desired layer to be used for computing Grad-CAM
    :param target: The target category to be used for computing Grad-CAM
    :return:
    """
    # target=target.cpu().numpy()
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
    # k, _, _ = gradcam.size()
    # for i in range(k):
    #     gradcam[i] = (gradcam[i] - gradcam[i].min()) / (gradcam[i].max() - gradcam[i].min())
    return gradcam
def compute_Layercam(output, feats , target, relu):
    """
    Compute the gradcam for the top predicted category
    :param output: the model output before softmax
    :param feats: the feature output from the desired layer to be used for computing Grad-CAM
    :param target: The target category to be used for computing Grad-CAM
    :return:
    """
    eps = 1e-8
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
    gradcam_mask = gradcam.unsqueeze(1)
    # print(gradcam_mask[1])
    gradcam_mask = F.interpolate(gradcam_mask, size=224, mode='bilinear')
    gradcam_mask = gradcam_mask.squeeze()
    # normalize the gradcam mask to sum to 1
    gradcam_mask_sum = gradcam_mask.sum(dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
    gradcam_mask = (gradcam_mask / (gradcam_mask_sum + eps)) + eps
    return gradcam_mask

def compute_gradcam_mask(images_outputs, images_feats , target, relu):
    """
    This function computes the grad-cam, upsamples it to the image size and normalizes the Grad-CAM mask.
    """
    eps = 1e-8
    gradcam_mask = compute_gradcam(images_outputs, images_feats , target, relu)
    gradcam_mask = gradcam_mask.unsqueeze(1)
    # print(gradcam_mask[1])
    gradcam_mask = F.interpolate(gradcam_mask, size=224, mode='bilinear')
    gradcam_mask = gradcam_mask.squeeze()
    # normalize the gradcam mask to sum to 1
    gradcam_mask_sum = gradcam_mask.sum(dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
    gradcam_mask = (gradcam_mask / (gradcam_mask_sum + eps)) + eps
    # gradcam_mask=forward_vanilla(gradcam_mask)
    # for i in range(k):
    #     gradcam_mask[i] = (gradcam_mask[i] - gradcam_mask[i].min()) / (gradcam_mask[i].max() - gradcam_mask[i].min())
    return gradcam_mask


def convert_to_gray(x, percentile=99):
    """
    Args:
        x: torch tensor with shape of (1, 3, H, W)
        percentile: int
    Return:
        result: shape of (1, 1, H, W)
    """
    x_2d = torch.abs(x).sum(dim=1).squeeze(0)
    v_max = np.percentile(x_2d, percentile)
    v_min = torch.min(x_2d)
    torch.clamp_((x_2d - v_min) / (v_max - v_min), 0, 1)
    return x_2d.unsqueeze(0).unsqueeze(0)

def show_cam(img, mask, title=None):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        img (Tensor): shape (1, 3, H, W)
        mask (Tensor): shape (1, 1, H, W)
    Return:
        heatmap (Tensor): shape (3, H, W)
        cam (Tensor): synthesized GradCAM cam of the same shape with heatmap.
        :param title:
    """
    mask = (mask - mask.min()).div(mask.max() - mask.min()).data
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze().float()), cv2.COLORMAP_JET)  # [H, W, 3]
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)

    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # cam = heatmap + img.cpu()
    cam = 1 * (1 - mask ** 0.8) * img.cpu() + (mask** 0.8) * heatmap
    # cam = (cam - cam.min()) / (cam.max() - cam.min())
    if title is not None:
        vutils.save_image(cam, title)

    return cam

