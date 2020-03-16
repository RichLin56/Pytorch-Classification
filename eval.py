#!/usr/bin/python
import argparse


############################################################
######################### Settings #########################
############################################################

parser = argparse.ArgumentParser()
parser.add_argument("-model", "--model_architecture", type=str, help="Classification architecture, one of: {}.".format(models.MODEL_NAMES), required=True)
parser.add_argument("-data", "--data_dir", type=str, help="Directory where evaluation data is stored.", required=True)
parser.add_argument("-n", "--num_classes", type=int ,help="Number of classes for classification.", required=True)
parser.add_argument("-cp", "--checkpoint", type=str, help="Path to model ckeckpoint which will evaluated.", default=None)
parser.add_argument("-o", "--output_dir", type=str , help="[Optional] Directory where output (model, log, config) data is stored. Directory will be created.", default=None)
parser.add_argument("-c", "--config_file", type=str, help="[Optional] Path to the .json config file.", default='eval_config.json')

args = parser.parse_args()
data_dir = args.data_dir
assert os.path.isdir(data_dir), '{} does not exist'.format(data_dir)

model_name = args.model_architecture
num_classes = args.num_classes
checkpoint = args.checkpoint
output_dir = args.output_dir
config_file = args.config_file

assert os.path.isfile(config_file), '{} does not exist'.format(config_file)
cfg = misc.load_config_json(config_file, config_group="evaluation")

if output_dir:
    output_dir = os.path.join(output_dir, "Evaluation_{}".format(misc.time_now()))
    os.makedirs(output_dir, exist_ok=True)
else:
    output_dir = os.path.join(os.getcwd(), "Evaluation_{}".format(misc.time_now()))
    os.makedirs(output_dir, exist_ok=True)

###########################################################
######################### Methods #########################
###########################################################

def eval_model(model, dataloader, output_dir):
    since = time.time()

    model.eval()   # Set model to evaluate mode

    running_corrects = 0
    num_classes = len(dataloaders[phase].dataset.classes)
    running_corrects_per_class = torch.zeros(num_classes, num_classes)

    # Iterate over data.
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # Batch statistics
        running_corrects += torch.sum(preds == labels.data)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            running_corrects_per_class[t.long(), p.long()] += 1
        
    # Epoch statistics
    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
    epoch_acc_per_class = running_corrects_per_class.diag()/running_corrects_per_class.sum(1)
    print('{} - Acc: {:.4f}'.format(phase, epoch_acc))
    print('{} - Acc per Class:'.format(phase))
    class_acc_dict = {}
    for i, label in enumerate(dataloaders[phase].dataset.class_to_idx):
        class_acc_dict[label] = epoch_acc_per_class[i].item()
        print('{} - {}: {:.2f}%'.format(phase, label, class_acc_dict[label]*100))
    print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    

#########################################################
######################### Model #########################
#########################################################

assert os.path.isfile(checkpoint), '{} does not exist'.format(checkpoint)
model_ft, input_size = models.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
model_ft.load_state_dict(torch.load(checkpoint))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

data_transforms = {
    'val': transforms.Compose(aug_val)
    }

##############################################################
######################### Dataloader #########################
##############################################################

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['test']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=True, num_workers=0) for x in ['test']}

###########################################################
######################### Evalutation #####################
###########################################################

eval_model(model_ft)

