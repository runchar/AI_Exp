import torch
import torch.nn as nn
import torchvision.models as models

class Adapter(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0):
        super(Adapter, self).__init__()
        self.linear = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.linear(x)


class CNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.1):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.res1 = nn.Conv2d(32, 64, kernel_size=1)
        self.res2 = nn.Conv2d(64, 128, kernel_size=1)
        self.res3 = nn.Conv2d(128, 256, kernel_size=1)

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward_feature(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        out1 = self.conv1(x)
        out1_p = self.pool1(out1)

        out2 = self.conv2(out1_p)
        res1 = self.res1(out1_p)  
        out2 = out2 + res1
        out2_p = self.pool2(out2)

        out3 = self.conv3(out2_p)
        res2 = self.res2(out2_p)  
        out3 = out3 + res2
        out3_p = self.pool3(out3)

        out4 = self.conv4(out3_p)
        res3 = self.res3(out3_p)  
        out4 = out4 + res3
        out4_p = self.pool4(out4)

        out = self.flatten(out4_p)
        out = self.fc(out)
        return out

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        return self.forward_feature(x)

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
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif cfg['model']['type'] == 'cnn':
        model = CNN(num_classes, dropout_rate)
    else:
        raise ValueError("Unsupported model type")
    return model