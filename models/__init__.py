import torch.nn as nn
from torchvision import models as modelzoo

from models.efficientnet_pytorch import EfficientNet

__model_names = {"vgg": ["vgg11", "vgg13", "vgg16", "vgg19"],
                 "resnet": ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "wide_resnet50", "wide_resnet101"],
                 "resnext": ["resnext50", "resnext101"],
                 "densenet": ["densenet121", "densenet161", "densenet169", "densenet201"],
                 "mnasnet": ["mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3"],
                 "shufflenet": ["shufflenet_x0_5", "shufflenet_x1_0", "shufflenet_x1_5", "shufflenet_x2_0"],
                 "squeezenet": ["squeezenet"],
                 "inception": ["inception"],
                 "mobilenet": ["mobilenet"],
                 "efficientnet": ["efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3", "efficientnet-b4", "efficientnet-b5", "efficientnet-b6", "efficientnet-b7"]
                }

MODEL_NAMES = [y for x in __model_names.values() for y in x]

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    # VGGs
    if "vgg" in model_name:
        if model_name == "vgg11":
            """ VGG11_bn
            """
            model_ft = modelzoo.vgg11_bn(pretrained=use_pretrained)

        elif model_name == "vgg13":
            """ VGG13_bn
            """
            model_ft = modelzoo.vgg13_bn(pretrained=use_pretrained)
        
        elif model_name == "vgg16":
            """ VGG16_bn
            """
            model_ft = modelzoo.vgg16_bn(pretrained=use_pretrained)

        elif model_name == "vgg19":
            """ VGG19_bn
            """
            model_ft = modelzoo.vgg19_bn(pretrained=use_pretrained)

        else:
            print("Invalid vgg model name, exiting...")
            exit()
        # Adjust last fully connected layer for VGG
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    # Resnets, Wide-Resnets 
    elif "resnet" in model_name:
        if model_name == "resnet18":
            """ Resnet18
            """
            model_ft = modelzoo.resnet18(pretrained=use_pretrained)

        elif model_name == "resnet34":
            """ Resnet34
            """
            model_ft = modelzoo.resnet34(pretrained=use_pretrained)

        elif model_name == "resnet50":
            """ Resnet50
            """
            model_ft = modelzoo.resnet50(pretrained=use_pretrained)

        elif model_name == "resnet101":
            """ Resnet101
            """
            model_ft = modelzoo.resnet101(pretrained=use_pretrained)

        elif model_name == "resnet152":
            """ Resnet152
            """
            model_ft = modelzoo.resnet152(pretrained=use_pretrained) 

        elif model_name == "wide_resnet50":
            """ Wide ResNet-50-2
            """  
            model_ft = modelzoo.wide_resnet50_2(pretrained=use_pretrained)

        elif model_name == "wide_resnet101":
            """ Wide ResNet-101-2
            """  
            model_ft = modelzoo.wide_resnet101_2(pretrained=use_pretrained)

        else:   
            print("Invalid resnet model name, choose one of: {}. Exiting...".format(__model_names["resnet"]))
            exit()
        # Adjust last fully connected layer for Resnet
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    # Resnexts
    elif "resnext" in model_name:
        if model_name == "resnext50":
            """ResNeXt-50 32x4d
            """
            model_ft = modelzoo.resnext50_32x4d(pretrained=use_pretrained)

        elif model_name == "resnext101":
            """ResNeXt-101 32x8d
            """ 
            model_ft = modelzoo.resnext101_32x8d(pretrained=use_pretrained)

        else:   
            print("Invalid resnext model name, choose one of: {}. Exiting...".format(__model_names["resnext"]))
            exit()
        # Adjust last fully connected layer for Resnext
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    # Densenets
    elif "densenet" in model_name:
        if model_name == "densenet121":
            """ Densenet121
            """
            model_ft = modelzoo.densenet121(pretrained=use_pretrained)

        elif model_name == "densenet161":
            """ Densenet161
            """
            model_ft = modelzoo.densenet161(pretrained=use_pretrained)

        elif model_name == "densenet169":
            """ Densenet169
            """
            model_ft = modelzoo.densenet169(pretrained=use_pretrained)

        elif model_name == "densenet201":
            """ Densenet201
            """
            model_ft = modelzoo.densenet201(pretrained=use_pretrained)

        else:   
            print("Invalid densenet model name, choose one of: {}. Exiting...".format(__model_names["densenet"]))
            exit()
    	# Adjust last fully connected layer for Densenet
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        #MNASNet V1.0
    # MNASnets
    elif "mnasnet" in model_name:
        if model_name == "mnasnet0_5":
            """MNASNet with depth multiplier of 0.5
            """
            model_ft = modelzoo.mnasnet0_5(pretrained=use_pretrained)

        elif model_name == "mnasnet0_75":
            """MNASNet with depth multiplier of 0.75
            """
            model_ft = modelzoo.mnasnet0_75(pretrained=use_pretrained)

        elif model_name == "mnasnet1_0":
            """MNASNet with depth multiplier of 1.0
            """
            model_ft = modelzoo.mnasnet1_0(pretrained=use_pretrained)

        elif model_name == "mnasnet1_3":
            """MNASNet with depth multiplier of 1.3
            """
            model_ft = modelzoo.mnasnet1_3(pretrained=use_pretrained)

        else:   
            print("Invalid mnasnet model name, choose one of: {}. Exiting...".format(__model_names["mnasnet"]))
            exit()
        # Adjust last fully connected layer for MNASnet
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs,num_classes)
    # Shufflenets
    elif "shufflenet" in model_name:
        if model_name == "shufflenet_x0_5":
            """ShuffleNetV2 with 0.5x output channels
            """
            model_ft = modelzoo.shufflenet_v2_x0_5(pretrained=use_pretrained)

        elif model_name == "shufflenet_x1_0":
            """ShuffleNetV2 with 1.0x output channels
            """
            model_ft = modelzoo.shufflenet_v2_x1_0(pretrained=use_pretrained)

        elif model_name == "shufflenet_x1_5":
            """ShuffleNetV2 with 1.5x output channels
            """
            model_ft = modelzoo.shufflenet_v2_x1_5(pretrained=use_pretrained)

        elif model_name == "shufflenet_x2_0":
            """ShuffleNetV2 with 2.0x output channels
            """
            model_ft = modelzoo.shufflenet_v2_x2_0(pretrained=use_pretrained)

        else:
            print("Invalid shufflenet model name, choose one of: {}. Exiting...".format(__model_names["shufflenet"]))
            exit()        
        # Adjust last fully connected layer for Shufflenet
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    elif "efficientnet" in model_name:
        if model_name in __model_names["efficientnet"]:
            """EfficientNetb0
            """   
            if use_pretrained: 
                model_ft = EfficientNet.from_pretrained(model_name)
            else:
                model_ft = EfficientNet.from_name(model_name)  
        else:
            print("Invalid efficientnet model name, choose one of: {}. Exiting...".format(__model_names["efficientnet"]))
            exit()
        # Adjust last fully connected layer for Shufflenet
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft._fc.in_features
        model_ft._fc = nn.Linear(num_ftrs, num_classes)
    # Squeezenet V1.1
    elif model_name == "squeezenet":
        """ Squeezenet1.1
        """
        model_ft = modelzoo.squeezenet1_1(pretrained=use_pretrained)
        # Adjust last fully connected layer for Squeezenet
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
    # Inception V3
    elif model_name == "inception":
        """ Inception v3
        """
        model_ft = modelzoo.inception_v3(pretrained=use_pretrained)
        # Adjust last fully connected layer for Inception v3
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
    # Mobilenet V2
    elif model_name == "mobilenet":
        """ MobileNetV2
        """
        model_ft = modelzoo.mobilenet_v2(pretrained=use_pretrained)
        # Adjust last fully connected layer for Mobilenet
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs,num_classes)
    # model_name not in model zoo
    else:
        print("Invalid model name, chose one of: {}.".format(pytorch_modelzoo.MODEL_NAMES))
        print("exiting...")
        exit()
    return model_ft

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
