import torch.nn as nn
import torchvision.models as models
import torch

class ResNetEncoder(models.resnet.ResNet):
    """Wrapper for TorchVison ResNet Model
    This was needed to remove the final FC Layer from the ResNet Model"""
    def __init__(self, block, layers, cifar_head=False, hparams=None):
        super().__init__(block, layers)
        self.cifar_head = cifar_head
        if cifar_head:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = self._norm_layer(64)
            self.relu = nn.ReLU(inplace=True)
        self.hparams = hparams

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.cifar_head:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class ResNetSimCLR(nn.Module):

    def __init__(self, model_name='resnet50', out_dim=1000, cifar_head=False):
        super(ResNetSimCLR, self).__init__()
        self.backbone = ResNetEncoder(models.resnet.Bottleneck, [3, 4, 6, 3], cifar_head=cifar_head, hparams=None)
        # add mlp projection head
        self.linear = nn.Linear(2048, out_dim)

    def forward(self, x):
        return self.linear(self.backbone(x))


if __name__=='__main__':
    mod = ResNetSimCLR()
    print(mod.state_dict().keys())
        
        