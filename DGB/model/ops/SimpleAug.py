import torch.nn as nn
import torchvision.transforms as transforms


class SimpleAugOP(nn.Module):

    def __init__(self, aug_type=None, p=0.5, degrees=45, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        super().__init__()
        self.aug_type = aug_type
        self.p = p
        self.degrees = degrees
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        if aug_type == "RandomErasing":
            self.transform = transforms.RandomErasing(p=self.p)
        elif aug_type == "Flip":
            self.transform = transforms.RandomHorizontalFlip(p=self.p)
        elif aug_type == "RandomRotation":
            self.transform = transforms.RandomRotation(degrees=self.degrees)
        elif aug_type == "ColorJitter":
            self.transform = transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.transform(x)
