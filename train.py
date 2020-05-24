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
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter

import models
import utils.builder
import utils.metrics as metrics
import utils.misc as misc

############################################################
######################### Settings #########################
############################################################

parser = argparse.ArgumentParser()
parser.add_argument("-model", "--model_architecture", type=str, help="Classification architecture, one of: {}.".format(models.MODEL_NAMES), required=True)
parser.add_argument("-data", "--data_dir", type=str, help="Directory where trainings data is stored.", required=True)
parser.add_argument("-n", "--num_classes", type=int ,help="Number of classes for classification.", required=True)
parser.add_argument("-b", "--batch_size", type=int , help="Batch size for training.", required=True)
parser.add_argument("-e", "--num_epochs", type=int , help="Number of epochs for training.", required=True)
parser.add_argument("-o", "--output_dir", type=str , help="[Optional] Directory where output (model, log, config) data is stored. Directory will be created.", default=None)
parser.add_argument("-c", "--config_file", type=str, help="[Optional] Path to the .json config file.", default='train_config.json')
parser.add_argument("-pt", "--use_pretrained", type=bool, help="[Optional] Use a model pretrained on ImageNet 1000.", default=True)
parser.add_argument("-cp", "--checkpoint", type=str, help="[Optional] Path to model ckeckpoint to continue training from an existing set of weights.", default=None)
parser.add_argument("-ft", "--feature_extract", type=bool , help="[Optional] Use for feature extraction == freeze layers (True) or finetuning (False).", default=False)
args = parser.parse_args()

data_dir = args.data_dir
assert os.path.isdir(data_dir), '{} does not exist'.format(data_dir)

model_name = args.model_architecture
num_classes = args.num_classes
batch_size = args.batch_size
num_epochs = args.num_epochs
output_dir = args.output_dir
config_file = args.config_file
use_pretrained = args.use_pretrained
checkpoint = args.checkpoint
feature_extract = args.feature_extract  # False: finetune the whole model, True: update the last fc layer params

assert os.path.isfile(config_file), '{} does not exist'.format(config_file)
cfg = misc.load_config_json(config_file, config_group="training")

if output_dir:
    output_dir = os.path.join(output_dir, "Training_{}".format(misc.time_now()))
    os.makedirs(output_dir, exist_ok=True)
else:
    output_dir = os.path.join(os.getcwd(), "Training_{}".format(misc.time_now()))
    os.makedirs(output_dir, exist_ok=True)

###########################################################
######################### Methods #########################
###########################################################

def train_model(model, dataloaders, criterion, optimizer, num_epochs, scheduler, summary_writer, device, output_dir, is_inception=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 100000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            num_classes = len(dataloaders[phase].dataset.classes)
            confusion_matrix = torch.zeros(num_classes, num_classes)

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    outputs_softmax = nn.Softmax(dim=1)(outputs)
                    preds_softmax, preds = torch.max(outputs_softmax, 1)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Batch statistics
                running_loss += loss.item() * inputs.size(0)

                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

            # Epoch statistics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print("phase[{}] Loss:\t {:4f}".format(phase, epoch_loss))

            # Tensorboard logging
            if summary_writer:
                summary_writer.add_scalar('Loss/{}'.format(phase), epoch_loss, epoch)    
                #TODO: Log Prec, Recall, ... to Tensorboard            
                #summary_writer.add_scalar('Accuracy/{}'.format(phase), epoch_acc, epoch)
                #summary_writer.add_scalars('Accuary per Class/{}'.format(phase), class_acc_dict, epoch)
                if phase == 'val':
                    summary_writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

            # Scheduler and Model saving
            if phase == 'val':
                print("phase[{}] Classification Report:".format(phase))
                print(metrics.create_classification_report(confusion_matrix.numpy(), dataloaders[phase].dataset.class_to_idx))
                if scheduler:
                    summary_writer.add_scalar('LR/{}'.format(phase), optimizer.param_groups[0]['lr'], epoch)
                    scheduler.step(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, os.path.join(output_dir, "__best_state_dict.pth.tar"))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

#########################################################
######################### Model #########################
#########################################################

if checkpoint:
    assert os.path.isfile(checkpoint), '{} does not exist'.format(checkpoint)
    model_ft = models.initialize_model(model_name, num_classes, feature_extract, use_pretrained=use_pretrained)
    model_ft.load_state_dict(torch.load(checkpoint))
else:
    model_ft = models.initialize_model(model_name, num_classes, feature_extract, use_pretrained=use_pretrained)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Setting up device, using {}...'.format(device))
model_ft = model_ft.to(device)

################################################################
######################### Augmentation #########################
################################################################

aug_train = [utils.builder.AugmentationBuilder(group_name=key, group_cfg=cfg['augmentation']['train'][key], 
                                               module=transforms).build() 
             for key in cfg['augmentation']['train'].keys()]

if 'Normalize' in cfg['augmentation']['train'].keys():
    assert list(cfg['augmentation']['train'].keys()).index('Normalize') == len(cfg['augmentation']['train'].keys()) - 1, 'Normalize must be last step of augmentations!'  
    aug_train.insert(-1, transforms.ToTensor())
else:
    aug_train.append(transforms.ToTensor())

aug_val = [utils.builder.AugmentationBuilder(group_name=key, group_cfg=cfg['augmentation']['val'][key], 
                                             module=transforms).build() 
           for key in cfg['augmentation']['val'].keys()]
if 'Normalize' in cfg['augmentation']['val'].keys():
    assert list(cfg['augmentation']['val'].keys()).index('Normalize') == len(cfg['augmentation']['val'].keys()) - 1, 'Normalize must be last step of augmentations!'  
    aug_val.insert(-1, transforms.ToTensor())
else:
    aug_val.append(transforms.ToTensor())


data_transforms = {
    'train': transforms.Compose(aug_train),
    'val': transforms.Compose(aug_val)
    }

##############################################################
######################### Dataloader #########################
##############################################################

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}

#############################################################
######################### Optimizer #########################
#############################################################

optimizer = utils.builder.OptimizerBuilder(group_name='optimizer', group_cfg=cfg['optimizer'], package=optim,
                                              model_params_to_update=model_ft.parameters()).build()

########################################################
######################### Loss #########################
########################################################

criterion = nn.CrossEntropyLoss()

#############################################################
######################### Scheduler #########################
#############################################################

lr_scheduler = utils.builder.SchedulerBuilder(group_name='lr_scheduler', group_cfg=cfg['lr_scheduler'], module=optim.lr_scheduler,
                                              optimizer=optimizer).build()

###########################################################
######################### Tensorboard #####################
###########################################################

outdir_tensorboard = os.path.join(output_dir, "Tensorboard")
os.makedirs(outdir_tensorboard, exist_ok=True)

########################################################
######################### Training #####################
########################################################

summary_writer = SummaryWriter(outdir_tensorboard)    
model_ft = train_model(model_ft, dataloaders_dict, criterion, optimizer, num_epochs, 
                       lr_scheduler, summary_writer, device, output_dir, is_inception=(model_name=="inception"))
summary_writer.close()
torch.save(copy.deepcopy(model_ft.state_dict()), os.path.join(output_dir, "{}_state_dict.pth.tar".format(model_name)))
os.remove(os.path.join(output_dir, "__best_state_dict.pth.tar"))
