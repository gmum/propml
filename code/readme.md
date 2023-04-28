# ProPML

## Setup
1. Download the dataset from https://cocodataset.org/#download (COCO 2014) or http://host.robots.ox.ac.uk/pascal/VOC/voc2007/ (VOC 2007).
2. Download a pre-trained TResNet-L for image size 224 trained on ImageNet from https://github.com/Alibaba-MIIL/TResNet/blob/master/MODEL_ZOO.md
3. Install pytorch, torchvision, randaugment, pycocotools, inplace_abn
4. Set dataset and model paths in local_settings.py

## Run training
1. Set hyperparametters in main.py (in first lines of the main function).
2. Start training with `python main.py`.


## Acknowledgements
This code is loosely based on https://github.com/Alibaba-MIIL/ASL
