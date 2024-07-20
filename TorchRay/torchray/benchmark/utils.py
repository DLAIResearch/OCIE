import numpy as np
import torch
import torch.nn.functional as F


def compute_gradcam(output, feats , target, relu):
    """
    Compute the gradcam for the top predicted category
    :param output: the model output before softmax
    :param feats: the feature output from the desired layer to be used for computing Grad-CAM
    :param target: The target category to be used for computing Grad-CAM
    :return:
    """

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

    return gradcam


def compute_gradcam_mask(images_outputs, images_feats , target, relu):
    """
    This function computes the grad-cam, upsamples it to the image size and normalizes the Grad-CAM mask.
    """
    eps = 1e-8
    gradcam_mask = compute_gradcam(images_outputs, images_feats , target, relu)
    gradcam_mask = gradcam_mask.unsqueeze(1)
    gradcam_mask = F.interpolate(gradcam_mask, size=224, mode='bilinear')
    gradcam_mask = gradcam_mask.squeeze()
    # normalize the gradcam mask to sum to 1
    gradcam_mask_sum = gradcam_mask.sum(dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
    gradcam_mask = (gradcam_mask / (gradcam_mask_sum + eps)) + eps

    return gradcam_mask
