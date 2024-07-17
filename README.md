# OCIE:  Augmenting Model Interpretability via Deconfounded Explanation-Guided Learning

Deep neural networks (DNNs) face significant challenges related to opacity, inherent biases, and shortcut learning, undermining their practical reliability. In this work, we construct a causal graph to model the unbiased DNN learning process, revealing that recurrent background information in training samples acts as a confounder, leading to spurious associations between model inputs and outputs. These spurious associations enable the model to make predictions based on biases. To address these issues and promote unbiased feature learning, we propose the Object-guided Consistency Interpretation Enhancement (OCIE) algorithm. OCIE enhances DNN interpretability by integrating explicit objects and explanations. Initially, OCIE employs a graph-based algorithm to identify explicit objects within self-supervised vision transformer-learned features. Subsequently, it formulates class prototypes to eliminate invalid detected objects. Finally, OCIE aligns explanations with explicit objects, directing the model's focus toward the most distinctive classification features rather than irrelevant backgrounds. We conduct extensive experiments on general (ImageNet), fine-grained (Stanford Cars and CUB-200), and medical (HAM) image classification datasets using two prevailing network architectures. Experimental results demonstrate that OCIE significantly enhances explanation consistency across all datasets. Furthermore, OCIE proves advantageous for fine-grained classification, especially in few-shot scenarios, improving both interpretability and classification performance. Additionally, our findings underscore the impact of centralized explanations on the sufficiency of model decisions.

![Image image](https://github.com/DLAIResearch/OCIE/blob/main/OCIE.jpg)

## Pre-requisites
- Pytorch 1.3 - Please install [PyTorch](https://pytorch.org/get-started/locally/) and CUDA if you don't have it installed.
- ## Datasets
 - [ImageNet - 100](https://www.image-net.org/download.php)
 - [CUB-200](https://vision.cornell.edu/se3/caltech-ucsd-birds-200/)
 - [HAM](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
 - [Stanford Cars-196](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

## Related Resources
* TokenCut-Github [link](https://github.com/YangtaoWANG95/TokenCut)
* TorchRay-Github [link](https://github.com/facebookresearch/TorchRay)
* LRP-based-Github [link](https://github.com/hila-chefer/Transformer-Explainability)

Parameters and datasets need to be set up before running
## Extract  mask.pt file from training set samples
```
 python unsupervised_saliency_detection/get_saliency.py
```
## Training
Parameters and datasets need to be set up before running

#### Train and evaluate Baseline methods (classification)
```
 python baseline_train_eval.py  
 python baseline_train_eval_VIT.py 
```

#### Train and evaluate OCIE (classification)
```
 python train_eval_OCIE.py
 python train_eval_OCIE_mask.py
 python train_eval_OCIE_VIT.py
 python train_eval_OCIE_VIT_mask.py
```
## Evaluate model explanation using CH,SPG,and SNR
We use the evaluation code adapted from the TorchRay framework.

#### Parameter settings
* Change directory to TorchRay and install the library. Please refer to the [TorchRay](https://github.com/facebookresearch/TorchRay) repository for full documentation and instructions.
    * cd TorchRay
    * python setup.py install

* Change directory to TorchRay/torchray/benchmark
    * cd torchray/benchmark
* For the ImageNet-100, CUB-200 and Stanfordcars datasets, this evaluation requires the following structure for validation images and bounding box xml annotations
    * <PATH_TO_FLAT_VAL_IMAGES_BBOX>/val/*.JPEG - Flat list of validation images
    * <PATH_TO_FLAT_VAL_IMAGES_BBOX>/annotation/*.xml - Flat list of annotation xml files
* For the HAM dataset, use the masks provided with the dataset to generate mask.pt.

#### 
```
python evaluate_fined_gradcam_stochastic_pointinggame.py
...
python evaluate_imagenet_gradcam_energy_inside_bbox.py
```
##  Evaluate explanation sufficiency 
Using images from the validation set
#### Evaluate sufficient-F1
```
python  sufficient_F1_ResNet.py
python  sufficient_F1_Vit.py
```
<br/>
