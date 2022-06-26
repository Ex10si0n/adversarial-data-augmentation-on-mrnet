import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms


class MRNet(nn.Module):
    def __init__(self):
        super(MRNet, self).__init__()

        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            # CBR
            else:
                layers += [nn.Conv2d(in_channels, v, 3, padding=1), nn.BatchNorm2d(v), nn.ReLU(True)]
                in_channels = v

        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 16)

    def forward(self, x):
        # print(x.size())
        x = self.features(x)

        # print(x.size())
        x = self.gap(x).view(x.size(0), -1)

        # print(x.size())
        # x = torch.max(x, 0, keepdim=True)[0]  # 我没看明白这个在搞什么
        x = self.classifier(x)
        # print(x.size())

        return x


data_path = '/Users/ex10si0n/Desktop/MRNet-v1.0'

# hyper param
epoch_num = 50
batch_size = 32
lr = 1e-3

train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# initialization
mrnet = MRNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mrnet.parameters(), lr=lr, weight_decay=1e-2)

torch.cuda.empty_cache()

# train
for epoch in range(epoch_num):
    for i, (pic, label) in enumerate(train_loader):
        pic = pic.to(device)
        label = label.to(device)

        output = mrnet(pic)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 28 == 0:
            print(f'Epoch: {epoch},batch: {i + 1}, loss: {loss.item():.4f}, lr:{optimizer.param_groups[0]["lr"]:.6f}')


# test
with torch.no_grad():
    correct = 0
    total = 0
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        output = mrnet(imgs)

        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


    print(f'Accruacy: {correct / total}')

torch.save(mrnet.state_dict(), 'mrnet.pth')
