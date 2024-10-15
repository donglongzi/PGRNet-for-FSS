import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from .decoder import build_decoder
from .IEM import build_IEM
from .PGR import PGR_Unit_2D


affine_par = True
mse_loss = torch.nn.MSELoss(reduction='mean')

def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.Resblock(block, 64, layers[0], stride=1)
        self.layer2 = self.Resblock(block, 128, layers[1], stride=2)
        self.layer3 = self.Resblock(block, 256, layers[2], stride=1, dilation=2)
        self.channel_compress = nn.Sequential(
            nn.Conv2d(in_channels=1024 + 512, out_channels=256, kernel_size=3, stride=1, padding=2, dilation=2,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.IEM = build_IEM(256)
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels= 256 + 256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.glore_layers1 = PGR_Unit_2D(256, 256, True)
        self.glore_layers2 = PGR_Unit_2D(256, 256, True)
        self.glore_layers3 = PGR_Unit_2D(256, 256, True)

        self.skip1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.skip2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.skip3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.dilation_conv_0 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
        )
        self.dilation_conv_1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
        )
        self.dilation_conv_6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=6, dilation=6, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.dilation_conv_12 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=12, dilation=12, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.dilation_conv_18 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=18, dilation=18, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.layer_out1 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0, bias=True),  # default=1280
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),

        )
        self.layer_out2 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=True)
        self.decoder = build_decoder(num_classes, 256, nn.BatchNorm2d)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def Resblock(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, query_img, support_img, support_mask):
        # important: do not optimize the RESNET backbone
        cosine_eps = 1e-7
        query_img = self.conv1(query_img)
        query_img = self.bn1(query_img)
        query_img = self.relu(query_img)
        query_img = self.maxpool(query_img)
        query_feature1 = self.layer1(query_img)
        query_feature2 = self.layer2(query_feature1)
        query_feature3 = self.layer3(query_feature2)
        query_feature_maps_23 = torch.cat([query_feature2, query_feature3], dim=1)
        query_feature_maps = self.channel_compress(query_feature_maps_23)

        support_img = self.conv1(support_img)
        support_img = self.bn1(support_img)
        support_img = self.relu(support_img)
        support_img = self.maxpool(support_img)
        support_feature1 = self.layer1(support_img)
        support_feature2 = self.layer2(support_feature1)
        support_feature3 = self.layer3(support_feature2)
        support_feature_maps_23 = torch.cat([support_feature2, support_feature3], dim=1)
        support_feature_maps = self.channel_compress(support_feature_maps_23)

        batch, channel, h, w = support_feature_maps.shape[:]
        batch_q, channel_q, qh, qw = query_feature_maps.shape[:]
        support_mask = F.interpolate(support_mask, support_feature_maps.shape[-2:], mode='bilinear', align_corners=True)

        support_feature_maps_masked = support_mask * support_feature_maps
        area = F.avg_pool2d(support_mask, [h, w]) * h * w + 5e-5

        query_feature_maps, support_feature_maps_masked = self.IEM(query_feature_maps, support_feature_maps_masked)

        # ####Mask Affinity Estimation####
        tmp_query = query_feature_maps.contiguous().view(batch_q, channel_q, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = support_feature_maps_masked.contiguous().view(batch, channel, -1)
        tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

        similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
        similarity = similarity.max(1)[0].view(batch, qh * qw)
        similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                    similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
        corr_query = similarity.view(batch_q, 1, qh, qw)

        # select representative regions by correlation matrix
        thresh = 0.7
        representative_regions = corr_query > thresh
        refined_query_feature_maps = representative_regions * query_feature_maps

        sup_conv_1 = F.avg_pool2d(support_feature_maps_masked, [h, w])
        sup_conv_1 = sup_conv_1.expand(-1, -1, query_feature_maps.shape[-2], query_feature_maps.shape[-1])
        correlation_feature_map = query_feature_maps * sup_conv_1

        query_feature_maps = query_feature_maps.view(1, batch * channel, qh, qw)

        # Dynamic Prototype Generation
        sup_conv_17 = F.adaptive_avg_pool2d(support_feature_maps_masked, output_size=[1, 7])
        sup_conv_17 = sup_conv_17.view(batch * channel, 1, 1, 7)

        sup_conv_71 = F.adaptive_avg_pool2d(support_feature_maps_masked, output_size=[7, 1])
        sup_conv_71 = sup_conv_71.view(batch * channel, 1, 7, 1)
        query_feature_maps = query_feature_maps.view(batch, channel, qh, qw)
        feature_maps = self.layer5(torch.cat([query_feature_maps, refined_query_feature_maps], dim=1))

        global_feature_maps = F.avg_pool2d(feature_maps, kernel_size=feature_maps.shape[-2:])
        global_feature_maps = self.dilation_conv_0(global_feature_maps)
        global_feature_maps = global_feature_maps.expand(-1, -1, feature_maps.shape[-2:][0], feature_maps.shape[-2:][1])

        #  Prototype-Guided Graph Reasoning & multi-scale feature fusion

        sup_conv_17 = sup_conv_17.view(batch, channel, 1, 7)
        sup_conv_71 = sup_conv_71.view(batch, channel, 7, 1)
        aspp_feature = torch.cat([self.glore_layers1(global_feature_maps, sup_conv_17, sup_conv_71),
                                         self.glore_layers2(self.dilation_conv_1(feature_maps), sup_conv_17, sup_conv_71),
                                         self.glore_layers3(self.dilation_conv_6(feature_maps), sup_conv_17, sup_conv_71)], dim=1)

        aspp_feature = self.layer_out1(aspp_feature)
        final_mask = self.decoder(aspp_feature, query_feature1)

        if self.training:
            aux_mask = self.layer_out2(aspp_feature)
            query_pred = nn.functional.softmax(final_mask, dim=1)
            query_pred = query_pred.argmax(dim=1, keepdim=True) # C x H x W
            query_pred = F.interpolate(query_pred.float(), query_feature_maps.shape[-2:], mode='bilinear', align_corners=True)

            query_feature_maps_masked = query_pred * query_feature_maps

            query_conv_17 = F.adaptive_avg_pool2d(query_feature_maps_masked, output_size=[1, 7])
            query_conv_17 = query_conv_17.view(batch_q, channel_q, 1, 7)

            query_conv_71 = F.adaptive_avg_pool2d(query_feature_maps_masked, output_size=[7, 1])
            query_conv_71 = query_conv_71.view(batch_q, channel_q, 7, 1)

            mse1 = mse_loss(query_conv_17.float(), sup_conv_17.float())
            mse2 = mse_loss(query_conv_71.float(), sup_conv_71.float())
            mse = mse1 + mse2

            return final_mask, aux_mask, mse
        else:
            return final_mask

def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3], 2)
    return model

def resnet101():
    model = ResNet(Bottleneck, [3, 4, 23, 3], 2)
    return model
