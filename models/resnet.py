from __future__ import absolute_import

import os
import time
from functools import partial
import torchvision
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

from models import inflate
from models.statistic_attention_block import StatisticAttentionBlock
from models.salient_to_broad_module import Salient2BroadModule
from models.integration_distribution_module import IntegrationDistributionModule
import matplotlib.pyplot as plt
import numpy as np
import cv2

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        # init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class Bottleneck3d(nn.Module):

    def __init__(self, bottleneck2d, inflate_time=False):
        super(Bottleneck3d, self).__init__()

        if inflate_time == True:
            self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=3, time_padding=1, center=True)
        else:
            self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)
        self.conv2 = inflate.inflate_conv(bottleneck2d.conv2, time_dim=1)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)
        self.conv3 = inflate.inflate_conv(bottleneck2d.conv3, time_dim=1)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)
        self.relu = nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = self._inflate_downsample(bottleneck2d.downsample)
        else:
            self.downsample = None

    def _inflate_downsample(self, downsample2d, time_stride=1):
        downsample3d = nn.Sequential(
            inflate.inflate_conv(downsample2d[0], time_dim=1, 
                                 time_stride=time_stride),
            inflate.inflate_batch_norm(downsample2d[1]))
        return downsample3d

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def draw_features(width,height,x,savename):
    tic=time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255  #float在[0，1]之间，转换成0-255
        img=img.astype(np.uint8)  #转成unit8
        img=cv2.applyColorMap(img, cv2.COLORMAP_JET) #生成heat map
        img = img[:, :, ::-1]#注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        print("{}/{}".format(i,width*height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time()-tic))

savepath=r'features_whitegirl'
if not os.path.exists(savepath):
    os.mkdir(savepath)


class _ResNet50(nn.Module):

    def __init__(self, num_classes, losses, plugin_dict):
        super(_ResNet50, self).__init__()
        self.losses = losses

        resnet2d = torchvision.models.resnet50(pretrained=True)
        resnet2d.layer4[0].conv2.stride = (1, 1)
        resnet2d.layer4[0].downsample[0].stride = (1, 1)

        self.conv1 = inflate.inflate_conv(resnet2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(resnet2d.maxpool, time_dim=1)

        self.layer1 = self._inflate_reslayer(resnet2d.layer1, plugin_dict[1])
        self.layer2 = self._inflate_reslayer(resnet2d.layer2, plugin_dict[2])
        self.layer3 = self._inflate_reslayer(resnet2d.layer3, plugin_dict[3])
        self.layer4 = self._inflate_reslayer(resnet2d.layer4, plugin_dict[4])

        self.bn = nn.BatchNorm1d(2048)
        self.bn.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, num_classes)
        self.classifier.apply(weights_init_classifier)
        # self.last_feature = None
        self.features = []

        if 'infonce' in self.losses:
            self.projection = nn.Conv1d(2048, 2048, (1,), (1,))
            self.projection.apply(weights_init_kaiming)

    def _inflate_reslayer(self, reslayer2d, plugin_dict):
        reslayers3d = []
        for i, layer2d in enumerate(reslayer2d):
            layer3d = Bottleneck3d(layer2d)
            reslayers3d.append(layer3d)

            if i in plugin_dict:
                reslayers3d.append(plugin_dict[i](in_dim=layer2d.bn3.num_features))

        return nn.Sequential(*reslayers3d)

    def forward(self, x):
        # print('fdgdfgfd')
        # print(x.size())
        # y = x[:, :, 0, :, :]
        # print(y.size())

        # 将 y 转换为 numpy 数组并调整形状
        # tensor = y.squeeze(0)  # 删除 batch 维
        # tensor = tensor.permute(1, 2, 0)  # 调整通道顺序
        # tensor = tensor.numpy()

        # 将 numpy 数组转换为 OpenCV 图像
        # image = cv2.cvtColor(np.uint8(tensor * 255), cv2.COLOR_RGB2BGR)

        # 保存图像
        # cv2.imwrite(savepath+"/output.jpg", image)


        x = self.conv1(x)

        # y = x[:,:,0,:,:]
        # draw_features(8, 8, y.cpu().numpy(), "{}/f1_conv1.png".format(savepath))

        x = self.bn1(x)

        # y = x[:, :, 0, :, :]
        # draw_features(8, 8, y.cpu().numpy(), "{}/f2_bn1.png".format(savepath))

        x = self.relu(x)

        # y = x[:, :, 0, :, :]
        # draw_features(8, 8, y.cpu().numpy(), "{}/f3_relu.png".format(savepath))

        x = self.maxpool(x)

        # y = x[:, :, 0, :, :]
        # draw_features(8, 8, y.cpu().numpy(), "{}/f4_maxpool.png".format(savepath))

        x = self.layer1(x)

        # y = x[:, :, 0, :, :]
        # draw_features(16, 16, y.cpu().numpy(), "{}/f5_layer1.png".format(savepath))

        x = self.layer2(x)

        # y = x[:, :, 0, :, :]
        # draw_features(16, 32, y.cpu().numpy(), "{}/f6_layer2.png".format(savepath))

        x = self.layer3(x)

        # y = x[:, :, 0, :, :]
        # draw_features(32, 32, y.cpu().numpy(), "{}/f7_layer3.png".format(savepath))

        x = self.layer4(x)
        # y = x[:, :, 0, :, :]
        # draw_features(32, 32, y.cpu().numpy()[:, 0:1024, :, :], "{}/f8_layer4_1.png".format(savepath))
        # draw_features(32, 32, y.cpu().numpy()[:, 1024:2048, :, :], "{}/f8_layer4_2.png".format(savepath))


        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b*t, c, h, w)
        x = F.max_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        x = x.transpose(1, 2)  # (b, c, t)

        if not self.training:
            x = self.bn(x)
            return x

        mid = self.bn(x)
        # self.last_feature = self.bn(x).detach()
        #
        # print(self.last_feature.size())
        # print("hefiduwgifde")

        v = x.mean(-1)
        f = self.bn(v)
        y = self.classifier(f)

        if 'infonce' in self.losses:
            x = self.bn(x)
            x = self.projection(x)
            return y, f, x,mid

        return y, f,mid

    def hook_fn(self, module, input, output):
        self.features.append(output)


def ResNet50(num_classes, losses):
    plugin_dict = {
        1: {},
        2: {},
        3: {},
        4: {}
    }
    return _ResNet50(num_classes, losses, plugin_dict)


def SANet(num_classes, losses, **kwargs):
    plugin_dict = {
        1: {},
        2: {1: StatisticAttentionBlock,
            3: StatisticAttentionBlock},
        3: {},
        4: {}
    }
    return _ResNet50(num_classes, losses, plugin_dict)


def IDNet(num_classes, losses, seq_len, **kwargs):
    plugin_dict = {
        1: {},
        2: {1: partial(IntegrationDistributionModule, t=seq_len),
            3: partial(IntegrationDistributionModule, t=seq_len)},
        3: {},
        4: {}
    }
    return _ResNet50(num_classes, losses, plugin_dict)


def SBNet(num_classes, losses, seq_len, **kwargs):
    plugin_dict = {
        1: {},
        2: {},
        3: {1: partial(Salient2BroadModule, split_pos=0),
            3: partial(Salient2BroadModule, split_pos=1),
            5: partial(Salient2BroadModule, split_pos=2)},
        4: {}
    }
    return _ResNet50(num_classes, losses, plugin_dict)


def SINet(num_classes, losses, seq_len, **kwargs):
    plugin_dict = {
        1: {},
        2: {1: partial(IntegrationDistributionModule, t=seq_len),
            3: partial(IntegrationDistributionModule, t=seq_len)},
        3: {1: partial(Salient2BroadModule, split_pos=0),
            3: partial(Salient2BroadModule, split_pos=1),
            5: partial(Salient2BroadModule, split_pos=2)},
        4: {}
    }
    return _ResNet50(num_classes, losses, plugin_dict)
