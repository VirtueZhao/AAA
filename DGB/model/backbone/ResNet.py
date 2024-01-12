import torch
import torch.nn as nn
from .build_backbone import BACKBONE_REGISTRY


pretrained_weights_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth"
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=(1, 1), downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.in_channels = 64
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=(2, 2))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.out_features = 512 * block.expansion

    def _make_layer(self, block, channels, block_num, stride=(1, 1)):
        downsample = None
        if stride[0] != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=channels * block.expansion, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion)
            )

        layers = [block(self.in_channels, channels, stride, downsample)]
        self.in_channels = channels * block.expansion
        for i in range(1, block_num):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        return x.view(x.size(0), -1)


@BACKBONE_REGISTRY.register()
def ResNet18(**kwargs):
    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])
    model.load_state_dict(torch.hub.load_state_dict_from_url(pretrained_weights_urls["resnet18"]), strict=False)

    return model
