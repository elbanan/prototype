from ..core.utils import *
from .metrics import *
from .inference import *
from .feature_extractor import *


class THREEDATA():
    def__init__(self, root):
    self.class, self.class_to_idx = find_classes(root)


    def __len__(self):
        return None

    def __getitem__(self, x):
        return None



class THREECNN():
    def __init__(self, img_size, num_target):
        self.conv1 = conv_unit(3, 8)
        self.conv2 = conv_unit(8, 16)
        self.conv3 = conv_unit(16, 32)
        self.conv4 = conv_unit(32, 64)
        self.fc = nn.Linear(self.conv_img_size*self.conv_img_size*64, 2)
        self.num_target = num_target
        self.conv_img_size = img_size/(2**4)

    def conv_unit(in_feature, out_feature, kernel=(3,3,3), stride=(3,3,3), batch_norm=True, relu=True, max_pool=True):
        layers = [nn.Conv3d(in_channels = in_feature, out_channels = out_feature, kernel_size = kernel, stride=stride, padding=0)]
        if batch_norm: layers.append(nn.BatchNorm3d(out_feature))
        if relu: layers.append(nn.ReLU())
        if max_pool:layers.append(nn.MaxPool3d(2,2))
        return nn.Sequential(*layers)


    def forward(x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(self.conv_img_size*self.conv_img_size*64 ,-1)
        x = self.fc(x)
        return x
