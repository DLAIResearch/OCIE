import sys
# from unsupervised_saliency_detection.get_saliency import backbone
import os
import torchvision
import joblib
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
sys.path.append('./model')
import dino  # model
import  gc
import object_discovery as tokencut
import argparse
import utils
import bilateral_solver
import os
from torch.utils.data import DataLoader
import PIL.Image as Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
torch.set_num_threads(4)

class CustomDatasetWithInfo(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.class_names = sorted(os.listdir(data_dir))
        self.image_info = []  # 用于存储图像信息，包括图像名称、类名称和标签

        # 枚举类名称并为每个类分配一个标签
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            image_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
            image_paths = [os.path.join(class_dir, f) for f in image_files]

            # 为每张图像添加图像名称、类名称和标签信息
            for image_path in image_paths:
                self.image_info.append({
                    'image_path': image_path,
                    'image_name': os.path.basename(image_path),
                    'class_name': class_name,
                    'label': label
                })

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        image_path = self.image_info[idx]['image_path']
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 获取图像名称、类名称和标签
        image_name = self.image_info[idx]['image_name']
        class_name = self.image_info[idx]['class_name']
        label = self.image_info[idx]['label']

        return image, image_name, class_name, label

# 加载数据集



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

## input / output dir
parser.add_argument('--out-dir', type=str, default='../datasets/CUB/train_cub_1_2_mask', help='output directory')
parser.add_argument('--img_dir', type=str, default='../datasets/DUTS_Test/car/train', help='output directory')

parser.add_argument('--vit-arch', type=str, default='base', choices=['base', 'small'], help='which architecture')

parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')

parser.add_argument('--patch-size', type=int, default=8, choices=[16, 8], help='patch size')

parser.add_argument('--tau', type=float, default=0.15, help='Tau for tresholding graph')

parser.add_argument('--sigma-spatial', type=float, default=16, help='sigma spatial in the bilateral solver')

parser.add_argument('--sigma-luma', type=float, default=8, help='sigma luma in the bilateral solver')

parser.add_argument('--sigma-chroma', type=float, default=8, help='sigma chroma in the bilateral solver')

args = parser.parse_args()
print(args)

## feature net

if args.vit_arch == 'base' and args.patch_size == 16:
    url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    feat_dim = 768
elif args.vit_arch == 'base' and args.patch_size == 8:
    # url = "/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    feat_dim = 768
elif args.vit_arch == 'small' and args.patch_size == 16:
    url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    # url = "./dino/dino_deitsmall16_pretrain.pth"
    feat_dim = 384
elif args.vit_arch == 'base' and args.patch_size == 8:
    url = "/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"



msg = 'Load {} pre-trained feature...'.format(args.vit_arch)
print(msg)

ToTensor = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)), ])


def get_tokencut_binary_map(img_pth, backbone, patch_size, tau):

    I = Image.open(img_pth).convert('RGB')
    I=I.resize((256,256))
    I_resize, w, h, feat_w, feat_h = utils.resize_pil(I, patch_size)
    tensor = ToTensor(I_resize).unsqueeze(0).cuda()
    feat = backbone(tensor)[0]
    seed, bipartition, eigvec = tokencut.ncut(feat, [feat_h, feat_w], [patch_size, patch_size], [h, w], tau)
    return bipartition, eigvec

if args.out_dir is not None and not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

# if args.img_path is not None:

count_vis = 0
mask_lost = []
mask_bfs = []
gt = []
data_dir ='../datasets/CUB/train_cub_1_2'
result_dir='../datasets/CUB/train_cub_1_2_mask'
backbone = dino.ViTFeat(url, feat_dim, args.vit_arch, args.vit_feat, args.patch_size)
backbone.eval()
backbone.cuda()
dataset = CustomDatasetWithInfo(data_dir,transform=ToTensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
image_annotations={}
for image, img_name, class_name,label in dataloader:
  args.img_dir=os.path.join(data_dir, class_name[0])
  args.out_dir=os.path.join(result_dir,class_name[0])
  if args.out_dir is not None and not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)
  with torch.no_grad():
    img_pth=os.path.join(args.img_dir, img_name[0])
    bipartition, eigvec = get_tokencut_binary_map(img_pth, backbone, args.patch_size, args.tau)

    torch.cuda.empty_cache()

    output_solver, binary_solver = bilateral_solver.bilateral_solver_output(img_pth, bipartition,
                                                                            sigma_spatial=args.sigma_spatial,
                                                                            sigma_luma=args.sigma_luma,
                                                                            sigma_chroma=args.sigma_chroma)
    torch.cuda.empty_cache()
    out_lost = os.path.join(args.out_dir, img_name[0])
    org = Image.open(img_pth).convert('RGB').resize((256,256))
    mask1=binary_solver.astype(float)
    image_cut=(org*mask1[:, :, np.newaxis])/255.0
    plt.imsave(fname=out_lost, arr=image_cut, cmap='cividis')
    resize_transform = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])
    mask = np.where(mask1==1)
    print(img_name)
    # binary_solver=None
    # mask_image=(org*mask1[:,:,np.newaxis])/255.0
    # prototype=torch.load('models_archive_imageNet_100.pt')
    # mask_image1=torch.tensor(np.array(mask_image)).permute(2, 0, 1).float()
    # print(int(label))
    # dot_product = torch.sum(mask_image1 * prototype[int(label)])
    # # dot_product1 = torch.dot(mask_image1.flatten(0) ,prototype[int(label)].flatten(0))
    # # # print(int(label))
    # norm_test = torch.norm(mask_image1)
    # norm_prototype = torch.norm(prototype[int(label)])
    # similarity = dot_product / (norm_test * norm_prototype)
    # print(similarity)
    # # back_image=Image.fromarray(np.uint8(mask_image))
    # mask = np.where(mask1[16:240, 16:240]==1)
    #or similarity<0.5
    mask2=resize_transform(Image.fromarray(mask1)).squeeze(0)
    mask2=mask2[16:240, 16:240].numpy()
    if mask[0].size == 0 or mask[1].size == 0 or mask2[1, 1]!= 0:
        mask2 = np.ones((224, 224), dtype=np.uint8)
    image_annotations[img_name]=mask2.astype(np.uint8)
save_file = os.path.join(result_dir, "*.pt")
joblib.dump(image_annotations, save_file)

