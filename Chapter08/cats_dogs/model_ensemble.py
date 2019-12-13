import os

import torch
import torch.nn as nn


def transfer_init(model, model_name, num_class):
    param_to_train = None
    if model_name in ['resnet18', 'resnet34', 'shufflenet_v2_x1_0', 'googlenet', 'resnext50_32x4d']:
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_class)
        param_to_train = model.fc.parameters()
    elif model_name in ['mobilenet_v2']:
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_class)
        param_to_train = model.classifier[1].parameters()
    elif model_name in ['squeezenet1_1']:
        num_features = model.classifier[1].in_channels
        model.classifier[1] = nn.Conv2d(num_features, num_class, kernel_size=1)
        param_to_train = model.classifier[1].parameters()
    elif model_name in ['densenet121']:
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_class)
        param_to_train = model.classifier.parameters()
    elif model_name in ['vgg11']:
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_class)
        param_to_train = model.classifier[6].parameters()
    return model, param_to_train


class ModelEnsemble(nn.Module):
    def __init__(self, model_names, num_class, model_path):
        super(ModelEnsemble, self).__init__()
        self.model_names = model_names
        self.num_class = num_class
        models = []
        for m in self.model_names:
            model = torch.load(os.path.join(model_path, '{}.pth'.format(m)))
            for param in model.parameters():
                param.requires_grad = False
            models.append(model)
        self.models = nn.Sequential(*models)
        self.vote_layer = nn.Linear(len(self.model_names)*self.num_class, self.num_class)

    def forward(self, input):
        raw_outputs = []
        for m in self.models:
            _out = m(input)
            raw_outputs.append(_out)
        raw_out = torch.cat(raw_outputs, dim=1)
        output = self.vote_layer(raw_out)
        return output
