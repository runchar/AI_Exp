import torch
import torch.nn as nn
import torchvision.models as models

class Adapter(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1):
        super(Adapter, self).__init__()
        self.linear = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.linear(x)


def construct_model(cfg):
    pretrained = cfg['model']['pretrained']
    num_classes = cfg['model']['num_classes']
    dropout_rate = cfg['model']['dropout_rate']

    adapter = Adapter(1000, num_classes, dropout_rate)

    if cfg['model']['type'] == 'vit':
        model = models.vit_b_16(pretrained=pretrained)
        model = nn.Sequential(model, adapter)
    elif cfg['model']['type'] == 'resnet':
        model = models.resnet50(pretrained=pretrained)
        model = nn.Sequential(model, adapter)
    else:
        raise ValueError("Unsupported model type")
    return model