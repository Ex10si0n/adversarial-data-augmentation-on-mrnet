import torch
import torch.nn as nn
from torchvision import models
import torchviz

# MRNet architecture definition

class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.cat([x, x, x], dim=1)  # convert 1-channel input to 3 channels
        x = torch.squeeze(x, dim=0) # only batch size 1 supported
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = x.view(-1, 256)
        x = self.classifier(x)
        return x

# if __name__ == '__main__':
#     model = MRNet()
#     x = torch.randn(1, 1, 224, 224)
#     torchviz.make_dot(model(x).mean(), params=dict(model.named_parameters())).render("mrnet", format="png")
