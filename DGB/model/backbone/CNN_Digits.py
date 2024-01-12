import torch.nn as nn
from DGB.utils import init_network_weights
from .build_backbone import BACKBONE_REGISTRY


class ConvNet(nn.Module):
    def __init__(self, c_hidden=64):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=c_hidden, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=c_hidden, out_channels=c_hidden, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=c_hidden, out_channels=c_hidden, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=c_hidden, out_channels=c_hidden, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.out_features = 2**2 * c_hidden

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x.reshape(x.size(0), -1)

@BACKBONE_REGISTRY.register()
def CNN_Digits(**kwargs):
    """
    This architecture was used for Digits dataset in
    Zhou et al. Deep Domain-Adversarial Image Generation for Domain Generalisation. AAAI 2020.
    """

    model = ConvNet(c_hidden=64)
    init_network_weights(model, init_type="kaiming")
    return model
