import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url
import logging
import os
from dotenv import find_dotenv, load_dotenv

# region: CONFIG STUB
logger = logging.getLogger(__name__)
logger.info("Loading ENV variables")
load_dotenv(find_dotenv())

# loading env vars
PROJECT_DIR = os.getenv('PROJECT_DIR')
RANDOM_SEED = int(os.getenv('RANDOM_SEED'))
TRAIN_BATCH_SIZE = int(os.getenv('TRAIN_BATCH_SIZE'))
# endregion


class VGG16(nn.Module):

    WEIGHTS_URL = 'https://download.pytorch.org/models/vgg16-397923af.pth'
    MODEL_DIR = f'{PROJECT_DIR}/models'
    ARCH = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

    def __init__(self, num_classes: int, pretrained: bool = False):
        super(VGG16, self).__init__()

        self.features = self._make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )

        if pretrained:
            self.load_state_dict(load_url(self.WEIGHTS_URL, model_dir=self.MODEL_DIR, progress=True))
        else:
            self._initialize_weights()

        self.output_adapter = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.output_adapter(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, batch_norm: bool = False):
        layers = []
        in_channels = 3
        for layer in self.ARCH:
            if layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, layer, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(layer), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = layer
        return nn.Sequential(*layers)


if __name__ == '__main__':
    x = torch.rand((2, 3, 32, 32))
    vggmodel = VGG16(num_classes=10, pretrained=True)
    y = vggmodel(x)