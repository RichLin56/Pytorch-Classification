#!/usr/bin/python
import argparse
import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torchvision import datasets, models, transforms

import models
import utils.builder
import utils.misc as misc

############################################################
######################### Settings #########################
############################################################

parser = argparse.ArgumentParser()
parser.add_argument("-model", "--model_architecture", type=str, help="Classification architecture, one of: {}.".format(models.MODEL_NAMES), required=True)
parser.add_argument("-data", "--data", type=str, help="Path to an image or to a directory where images are stored.", required=True)
parser.add_argument("-n", "--num_classes", type=int ,help="Number of classes for classification.", required=True)
parser.add_argument("-cp", "--checkpoint", type=str, help="[Optional] Path to model ckeckpoint which will evaluated.")
parser.add_argument("-ext", "--extension", type=str, help="[Optional] Extension of input images.", default="")
parser.add_argument("-c", "--config_file", type=str, help="[Optional] Path to the .json config file.", default='predict_config.json')
parser.add_argument("-o", "--output_dir", type=str , help="[Optional] Directory where output (model, log, config) data is stored. Directory will be created.", default=None)
parser.add_argument("-gpu", "--use_gpu", help="[Optional] Usage of CPU/GPU.", default=False, action='store_true')
args = parser.parse_args()

data_dir = args.data
assert os.path.isdir(data_dir) or os.path.isfile(data_dir), '{} does not exist'.format(data_dir)

model_name = args.model_architecture
num_classes = args.num_classes
checkpoint = args.checkpoint
extension = args.extension
config_file = args.config_file
output_dir = args.output_dir
use_gpu = args.use_gpu

assert os.path.isfile(config_file), '{} does not exist'.format(config_file)
cfg = misc.load_config_json(config_file, config_group="prediction")

images = list()
if os.path.isfile(data_dir):
    images.append(data_dir)
else:
    images = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(extension)]  

if output_dir:
    output_dir = os.path.join(output_dir, "Prediction_{}".format(misc.time_now()))
    os.makedirs(output_dir, exist_ok=True)
else:
    output_dir = os.path.join(os.getcwd(), "Prediction_{}".format(misc.time_now()))
    os.makedirs(output_dir, exist_ok=True)

class_to_label = cfg["class_to_label"]

###########################################################
######################### Methods #########################
###########################################################

def pred_model(model, images: list, class_to_label: dict, data_transforms: dict, device, output_dir):
    since = time.time()
    phase = 'pred'
    model.eval()   # Set model to evaluate mode

    trafo = data_transforms[phase]

    for image_path in images:
        image = Image.open(image_path)
        image = image.convert("RGB")
        inputs = trafo(image)
        inputs = inputs.to(device)       
        inputs = inputs.unsqueeze(0)
        outputs = model(inputs)
        outputs_softmax = nn.Softmax(dim=1)(outputs)
        preds_softmax, preds = torch.max(outputs_softmax, 1)

        image.save(os.path.join(output_dir,  "{}_{}_{}".format(class_to_label[str(preds.item())], str(preds_softmax.item()), os.path.basename(image_path))))
    print()
    time_elapsed = time.time() - since
    print('Prediction complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    pass

#########################################################
######################### Model #########################
#########################################################
device = torch.device("cpu")
if use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Setting up device, using {}...'.format(device))

assert os.path.isfile(checkpoint), '{} does not exist'.format(checkpoint)
model_ft = models.initialize_model(model_name, num_classes, feature_extract=True, use_pretrained=True)
model_ft.load_state_dict(torch.load(checkpoint, map_location=device))

model_ft = model_ft.to(device)

################################################################
######################### Augmentation #########################
################################################################

aug_val = [utils.builder.AugmentationBuilder(group_name=key, group_cfg=cfg['augmentation']['val'][key], 
                                             module=transforms).build() 
           for key in cfg['augmentation']['val'].keys()]
if 'Normalize' in cfg['augmentation']['val'].keys():
    assert list(cfg['augmentation']['val'].keys()).index('Normalize') == len(cfg['augmentation']['val'].keys()) - 1, 'Normalize must be last step of augmentations!'  
    aug_val.insert(-1, transforms.ToTensor())
else:
    aug_val.append(transforms.ToTensor())

data_transforms = {'pred': transforms.Compose(aug_val)}

###########################################################
######################### Evalutation #####################
###########################################################

pred_model(model_ft, images, class_to_label, data_transforms, device=device, output_dir=output_dir)
