# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

r"""
This module provides a number of benchmark dataset:

* ImageNet ILSVCR 12 and other *image folders* dataset (:class:`ImageFolder`).
* PASCAL VOC (:class:`VOCDetection`).
* MS COCO (:class:`CocoDetection`).

The classes in this module extend corresponding classes in
:mod:`torchvision.dataset` with functions for converting labels in various
formats and similar. Some of these functions are also provided as
"stand alone".
"""

__all__ = [
    'coco_as_class_ids',
    'coco_as_image_size',
    'coco_as_mask',
    'voc_as_class_ids',
    'annotation_as_image_size',
    'voc_as_mask',
    'ImageFolder',
    'VOCDetection',
    'CocoDetection',
    'ImageNetDetection',
    'get_dataset'
]

import os
import sys
from packaging import version
import pdb
from PIL import Image
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import collections
import torch
from torch.utils.data.dataloader import default_collate
import torchvision

try:
    import importlib.resources as resources
except ImportError:
    import importlib_resources as resources

from torchray.utils import get_config

with resources.open_text('torchray.benchmark', 'imagenet_classes.txt') as f:
    IMAGENET_CLASSES = f.readlines()
    """List of the 1000 ImageNet ILSVRC class names."""

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train-o', 'tvmonitor',
]
"""List of the 20 PASCAL VOC class names."""

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train-o',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed',
    'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]
"""List of the 80 COCO class names."""

_VOC_CLASS_TO_INDEX = {c: i for i, c in enumerate(VOC_CLASSES)}
_COCO_CLASS_TO_INDEX = {c: i for i, c in enumerate([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
])}
# parh=r'../dataset/imagenet2/train-o'
#修改此处针对不同数据集
_IMAGENET_CLASS_TO_INDEX = {c: i for i, c in enumerate(sorted(os.listdir('train')))}


def voc_as_class_ids(label):
    """Convert a VOC detection label to the list of class IDs.

    Args:
        label (dict): an image label in the VOC detection format.

    Returns:
        list: List of ids of classes in the image.
    """
    objs = label['annotation']['object']
    if isinstance(objs, list):
        class_names = [obj['name'] for obj in objs]
    else:
        class_names = [objs['name']]
    return list({_VOC_CLASS_TO_INDEX[n] for n in class_names})


def imagenet_as_class_ids(label):
    """Convert an ImageNet detection label to the list of class IDs.

    Args:
        label (dict): an image label in the ImageNet detection format.

    Returns:
        list: List of ids of classes in the image.
    """
    objs = label['annotation']['object']
    if isinstance(objs, list):
        class_names = [obj['name'] for obj in objs]
    else:
        class_names = [objs['name']]
    return list({_IMAGENET_CLASS_TO_INDEX[n] if n.startswith('n') else n for n in class_names})


def voc_as_mask(label, class_id):
    """Convert a VOC detection label to a mask.

    Return a boolean mask selecting the region contained in the bounding boxes
    of :attr:`class_id`.

    Args:
        label (dict): an image label in the VOC detection format.
        class_id (int): ID of the requested class.

    Returns:
        :class:`torch.Tensor`: 2D boolean tensor.
    """
    weight, height = annotation_as_image_size(label)
    mask = torch.zeros((height, weight), dtype=torch.uint8)
    objs = label['annotation']['object']
    if not isinstance(objs, list):
        objs = [objs]
    for obj in objs:
        this_class_id = _VOC_CLASS_TO_INDEX[obj['name']]
        if this_class_id != class_id:
            continue
        bbox = obj['bndbox']
        ymin = int(bbox['ymin']) - 1
        ymax = int(bbox['ymax']) - 1
        xmin = int(bbox['xmin']) - 1
        xmax = int(bbox['xmax']) - 1
        mask[ymin:ymax + 1, xmin:xmax + 1] = 1
    if version.parse(torch.__version__) >= version.parse("1.2.0"):
        mask = mask.to(torch.bool)
    return mask


def imagenet_as_mask(label, class_id, tolerance=0):
    """Convert an ImageNet detection label to a mask.

    Return a boolean mask selecting the region contained in the bounding boxes
    of :attr:`class_id`.

    Args:
        label (dict): an image label in the VOC detection format.
        class_id (int): ID of the requested class.
        tolerance: The mask will be dilated by this value

    Returns:
        :class:`torch.Tensor`: 2D boolean tensor.
    """
    width, height = annotation_as_image_size(label)
    mask = torch.zeros((height, width), dtype=torch.uint8)
    objs = label['annotation']['object']
    if not isinstance(objs, list):
        objs = [objs]
    for obj in objs:
        if obj['name'].startswith('n'):
            this_class_id = _IMAGENET_CLASS_TO_INDEX[obj['name']]
        elif obj['name'].split('.')[0] is not None:
            # this_class_id = int(obj['name'].split('.')[0]) - 1
            this_class_id = _IMAGENET_CLASS_TO_INDEX[obj['name']]
        else:
            this_class_id = int(obj['name'])
        if this_class_id != class_id:
            continue
        bbox = obj['bndbox']
        ymin = int(bbox['ymin']) - 1
        ymax = int(bbox['ymax']) - 1
        xmin = int(bbox['xmin']) - 1
        xmax = int(bbox['xmax']) - 1

        if tolerance > 0:
            ymin = min(0, ymin - tolerance)
            ymax = max(height - 1, ymax + tolerance)
            xmin = min(0, xmin - tolerance)
            xmax = max(width - 1, xmax + tolerance)

        mask[ymin:ymax + 1, xmin:xmax + 1] = 1
    if version.parse(torch.__version__) >= version.parse("1.2.0"):
        mask = mask.to(torch.bool)
    return mask


def annotation_as_image_size(label):
    """Convert a VOC detection label to the image size.

    Args:
        label (dict): an image label in the VOC detection format.

    Returns:
        tuple: width, height of image.
    """
    width = int(label['annotation']['size']['width'])
    height = int(label['annotation']['size']['height'])
    return width, height


def annotation_as_image_name(label):
    """Convert a VOC detection label to the image name.

    Args:
        label (dict): an image label in the VOC detection format.

    Returns:
        str: name.
    """
    return os.path.splitext(label['annotation']['filename'])[0]


def coco_as_class_ids(label):
    """Convert a COCO detection label to the list of class IDs.

    Args:
        label (list of dict): an image label in the VOC detection format.

    Returns:
        list: List of ids of classes in the image.
    """
    if len(label) == 0:
        return []
    return list({_COCO_CLASS_TO_INDEX[ann['category_id']] for ann in label})


def coco_as_mask(dataset, label, class_id):
    """Convert a COCO detection label to a mask.

    Return a boolean mask for the regions of :attr:`class_id`.

    If the label is the empty list, because there are no objects at all in the
    image, the function returns ``None``.

    Args:
        label (array of dict): an image label in the VOC detection format.
        class_id (int): ID of the requested class.

    Returns:
        :class:`torch.Tensor`: 2D boolean tensor.
    """
    assert isinstance(class_id, int)
    mask = None
    if not label:
        return mask
    image_id = label[0]['image_id']
    image = dataset.coco.loadImgs(image_id)[0]
    for ann in label:
        this_class_id = _COCO_CLASS_TO_INDEX[ann['category_id']]
        if class_id == this_class_id:
            if mask is None:
                mask = torch.zeros(
                    image['height'],
                    image['width'],
                    dtype=torch.uint8
                )
            this_mask = dataset.coco.annToMask(ann)
            mask.add_(torch.tensor(this_mask))
    if mask is not None:
        mask = mask > 0
    if version.parse(torch.__version__) >= version.parse("1.2.0"):
        mask = mask.to(torch.bool)
    return mask


def coco_as_image_size(dataset, label):
    """Convert a COCO detection label to the image size.

    Args:
        label (list of dict): an image label in the VOC detection format.

    Returns:
        tuple: width, height of image.
    """
    if not label:
        return None
    image_id = label[0]['image_id']
    image = dataset.coco.loadImgs(image_id)[0]
    return image['width'], image['height']


def coco_as_image_name(dataset, label):
    """Convert a COCO detection label to the image name.

    Args:
        label (list of dict): an image label in the COCO detection format.

    Returns:
        str: image name.
    """
    if not label:
        return None
    image_id = label[0]['image_id']
    image = dataset.coco.loadImgs(image_id)[0]
    return os.path.splitext(image['file_name'])[0]
def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
class ImageFolder(torchvision.datasets.ImageFolder):
    """Image folder dataset.

    This class extends :class:`torchvision.dataset.ImageFolder`.
    Its constructor supports the following additional arguments:

    Args:
        limiter (int, optional): limit the dataset to :attr:`limiter` images,
            picking from each class in a round-robin fashion.
            Default: ``None``.
        full_classes (list of str, optional):  list of full class names.
            Default: ``None``.

    Attributes:
        selection (list of int): indices of the active images.
        full_classes (list of str): class names.
    """

    def __init__(self, *args, limiter=None, full_classes=None, **kwargs):
        super(ImageFolder, self).__init__(*args, **kwargs)
        num_images = super(ImageFolder, self).__len__()
        self.selection = range(num_images)
        self.full_classes = full_classes
        if not limiter:
            return
        # Pick one sample per class in a round-robin manner.
        class_indices = [
            [i for i, y in enumerate(self.targets) if y == label]
            for label in range(len(self.classes))
        ]
        triplets = [
            (k, y, i)
            for y, indices in enumerate(class_indices)
            for k, i in enumerate(indices)
        ]
        triplets.sort()
        self.selection = [i for k, y, i in triplets[:min(limiter, num_images)]]
        self.selection = sorted(self.selection)

    def __getitem__(self, index):
        return super().__getitem__(self.selection[index])

    def __len__(self):
        return len(self.selection)

    def get_image_url(self, i):
        """Get the URL of an image.

        Args:
            i (int): image index.

        Returns:
            str: image URL.
        """
        return self.samples[self.selection[i]][0]


class VOCDetection(torchvision.datasets.VOCDetection):
    """PASCAL VOC Detection dataset.

    This class extends :class:`torchvision.dataset.VOCDetection`.
    Its constructor supports the following additional arguments:

    Args:
        limiter (int, optional): limit the dataset to the first :attr:`limiter`
            images. Default: ``None``.

    Attributes:
        selection (list of int): indices of the active images.
        classes (list of str): class names.
    """

    # The default VOCDetection is too strict and raises an error if ones
    # tries to load a test set, although this is often used in the literature.
    def _compatible_init(
            self,
            root,
            year='2012',
            image_set='train',
            download=False,
            transform=None,
            target_transform=None,
            transforms=None):
        super(torchvision.datasets.voc.VOCDetection, self).__init__(
            root, transforms, transform, target_transform)

        self.year = year
        self.url = torchvision.datasets.voc.DATASET_YEAR_DICT[year]['url']
        self.filename = torchvision.datasets.voc.DATASET_YEAR_DICT[year]['filename']
        self.md5 = torchvision.datasets.voc.DATASET_YEAR_DICT[year]['md5']
        self.image_set = torchvision.datasets.utils.verify_str_arg(image_set, "image_set",
                                                                   ("train", "trainval", "val", "test"))

        base_dir = torchvision.datasets.voc.DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        if download:
            torchvision.datasets.utils.download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.'
                               + ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/Main')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(
            annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))

    def __init__(self,
                 *args,
                 limiter=None,
                 **kwargs):

        if version.parse(torchvision.__version__) > version.parse("0.3"):
            self._compatible_init(*args, **kwargs)
        else:
            super(VOCDetection, self).__init__(*args, **kwargs)

        num_images = super(VOCDetection, self).__len__()
        if limiter:
            num_images = min(num_images, limiter)
        self.selection = range(num_images)
        self.classes = VOC_CLASSES

    def __getitem__(self, index):
        return super().__getitem__(self.selection[index])

    def __len__(self):
        return len(self.selection)

    def get_image_url(self, i):
        """Get the URL of an image.

        Args:
            i (int): Image index.

        Returns:
            str: Image URL.
        """
        return self.images[self.selection[i]]

    def as_class_ids(self, label):
        """Convert a label to list of class IDs.

        The same as :func:`voc_as_class_ids`.
        """
        return voc_as_class_ids(label)

    def as_mask(self, label, class_id):
        """Convert a label to a mask.

        The same as :func:`voc_as_mask`.
        """
        return voc_as_mask(label, class_id)

    def as_image_size(self, label):
        """Convert a label to the image size.

        The same as :func:`annotation_as_image_size`.
        """
        return annotation_as_image_size(label)

    def as_image_name(self, label):
        """Convert a label to the image name.

        The same as :func:`annotation_as_image_name`.
        """
        return annotation_as_image_name(label)

    @staticmethod
    def collate(batch):
        """Collate function for use in a data loader."""
        inputs = default_collate([elem[0] for elem in batch])
        labels = [elem[1] for elem in batch]
        return inputs, labels


class ImageNetDetection(torchvision.datasets.VisionDataset):
    """ImageNet Detection dataset (Annotations available for validation set).

    This class extends :class:`torchvision.dataset.VisionDataset`.
    Its constructor supports the following additional arguments:

    Args:
        limiter (int, optional): limit the dataset to the first :attr:`limiter`
            images. Default: ``None``.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Attributes:
        selection (list of int): indices of the active images.
        classes (list of str): class names.
    """

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 paths=False):
        super(ImageNetDetection, self).__init__(root, transforms, transform, target_transform)

        imagenet_root = self.root
        image_dir = os.path.join(imagenet_root, 'val')
        annotation_dir = os.path.join(imagenet_root, 'annotation')

        # pdb.set_trace()
        if not os.path.isdir(imagenet_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        im_list = []
        for file in os.listdir(image_dir):
            if file.endswith('.JPEG') or file.endswith('.jpg'):
                im_list.append(os.path.join(image_dir, file))

        self.images = sorted(im_list)
        self.annotations = []
        for im in self.images:
            self.annotations.append(im.replace('/val', '/annotation/').replace('.JPEG', '.xml').replace('jpg', 'xml'))
        self.paths = paths
        # with open(os.path.join(self.root, 'ILSVRC2012_validation_ground_truth.txt'), 'r') as label_file:
        #     self.labels = label_file.readlines()
        # self.labels = [int(line.strip('\n')) for line in self.labels]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        # pdb.set_trace()
        img_path = self.images[index]
        ano=ET.parse(self.annotations[index])
        b=ano.getroot()
        img = Image.open(self.images[index]).convert('RGB')
        annotation = self.parse_annotation_xml(
            ET.parse(self.annotations[index]).getroot())
        label = annotation['annotation']['object'][0]['name']
        if isinstance(label, list):
            label = label[0]
        if label.startswith('n'):
            target = _IMAGENET_CLASS_TO_INDEX[label]
        elif label.split('.')[0] is not None:
            # target = int(label.split('.')[0]) - 1
            target = _IMAGENET_CLASS_TO_INDEX[label]
        else:
            target = int(label)
        # target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

        # print(annotation)
        if self.paths:
            return img_path, img, annotation, target
        else:
            return img, annotation, target

    def __len__(self):
        return len(self.images)

    def parse_annotation_xml(self, node):
        annotation_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_annotation_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]

            annotation_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                annotation_dict[node.tag] = text
        return annotation_dict

    def get_image_url(self, i):
        """Get the URL of an image.

        Args:
            i (int): Image index.

        Returns:
            str: Image URL.
        """
        return self.images[self.selection[i]]

    def as_class_ids(self, label):
        """Convert a label to list of class IDs.

        The same as :func:`imagenet_as_class_ids`.
        """
        return imagenet_as_class_ids(label)

    def as_mask(self, label, class_id):
        """Convert a label to a mask.

        The same as :func:`imagenet_as_mask`.
        """
        return imagenet_as_mask(label, class_id)

    def as_image_size(self, label):
        """Convert a label to the image size.

        The same as :func:`annotation_as_image_size`.
        """
        return annotation_as_image_size(label)

    def as_image_name(self, label):
        """Convert a label to the image name.

        The same as :func:`annotation_as_image_name`.
        """
        return annotation_as_image_name(label)

    @staticmethod
    def collate(batch):
        """Collate function for use in a data loader."""
        inputs = default_collate([elem[0] for elem in batch])
        labels = [elem[1] for elem in batch]
        return inputs, labels


class CocoDetection(torchvision.datasets.CocoDetection):
    """COCO Detection dataset.

    The data can be downloaded at `<http://cocodataset.org/#download>`__.

    Args:
        limiter (int, optional): limit the dataset to the first :attr:`limiter`
            images. Default: ``None``.

    Attributes:
        classes (list of str): class names.
        selection (list of int): indices of the active images.
    """

    def __init__(self, root, annFile, *args, limiter=None, **kwargs):
        super(CocoDetection, self).__init__(root, annFile, *args, **kwargs)
        self.subset = os.path.splitext(os.path.basename(annFile))[0]
        num_images = super(CocoDetection, self).__len__()
        if limiter:
            num_images = min(num_images, limiter)
        self.selection = range(num_images)
        self.classes = COCO_CLASSES

    def __getitem__(self, index):
        return super().__getitem__(self.selection[index])

    def __len__(self):
        return len(self.selection)

    def get_image_url(self, i):
        """Return image url.

        Args:
            i (int): image index.

        Returns:
            str: path to image.
        """
        i = self.selection[i]
        image_id = self.ids[i]
        return self.coco.loadImgs(image_id)[0]['file_name']

    @property
    def images(self):
        """list of str: paths to images."""
        return [self.coco.loadImgs(i)[0]['file_name'] for i in self.ids]

    def as_class_ids(self, label):
        """Convert a label to list of class IDs.

        The same as :func:`coco_as_class_ids`.
        """
        return coco_as_class_ids(label)

    def as_mask(self, label, class_id):
        """Convert a label to a mask.

        The same as :func:`coco_as_mask`.
        """
        return coco_as_mask(self, label, class_id)

    def as_image_size(self, label):
        """Convert a label to the image size.

        The same as :func:`coco_as_image_size`.
        """
        return coco_as_image_size(self, label)

    def as_image_name(self, label):
        """Convert a label to the image name.

        The same as :func:`coco_as_image_name.`.
        """
        return coco_as_image_name(self, label)

    @staticmethod
    def collate(batch):
        """Collate function for use in a data loader."""
        inputs = default_collate([elem[0] for elem in batch])
        labels = [elem[1] for elem in batch]
        return inputs, labels


def get_dataset(name,
                subset,
                dataset_dir=None,
                annotation_dir=None,
                transform=None,
                limiter=None,
                download=False):
    """Returns a :class:`torch.data.Dataset`.

    Args:
        name (str): name of the dataset; choose from ``"imagenet2"``, ``"voc"``
            or ``"coco"``.
        subset (str): name of the dataset subset or split.
        dataset_dir (str, optional): Path to root directory containing data.
            Default: ``None``.
        annotation_dir (str, optional):
            Path to root directory containing annotations. Required for COCO
            only. Default: ``None``.
        transform (function, optional): input transformation function.
            Default: ``None``.
        limiter (int, optional): limit the dataset to :attr:`limiter`
            images. Default: ``None``.
        download (bool, optional): If True and :attr:`name` is ``"voc"``,
            download the dataset to :attr:`dataset_dir`. Default: ``False``.

    Returns:
        :class:`torch.data.Dataset`: the requested dataset.
    """

    def get(this, default):
        return this if this is not None else default

    if 'imagenet' in name:
        dataset_dir = get(dataset_dir,
                          get_config()['benchmark']['imagenet_dir'])
        dataset = ImageFolder(os.path.join(dataset_dir, subset),
                              transform=transform,
                              limiter=limiter,
                              full_classes=IMAGENET_CLASSES)

    elif 'voc' in name:
        dataset_dir = get(dataset_dir, get_config()['benchmark']['voc_dir'])
        year = name.split('_')[-1]
        dataset = VOCDetection(
            root=dataset_dir,
            year=year,
            image_set=subset,
            transform=transform,
            download=download,
            limiter=limiter,
        )

    elif 'coco' in name:
        dataset_dir = get(dataset_dir, get_config()['benchmark']['coco_dir'])
        annotation_dir = get(annotation_dir,
                             get_config()['benchmark']['coco_anno_dir'])

        im_path = os.path.join(dataset_dir, subset)
        ann_path = os.path.join(annotation_dir,
                                'instances_{}.json'.format(subset))
        dataset = CocoDetection(im_path,
                                ann_path,
                                transform=transform,
                                limiter=limiter)

    else:
        assert False, "Unknown dataset {}".format(name)

    return dataset
