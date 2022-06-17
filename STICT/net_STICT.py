#!/usr/bin/python3
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from spatial_correlation_sampler import spatial_correlation_sample


def weight_init(module):
    for n, m in module.named_children():
        # print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3 * dilation - 1) // 2,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out + x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 3, stride=1, dilation=1)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('./pretrained_model/resnet50-19c8e357.pth'), strict=False)


class FFM(nn.Module):
    def __init__(self):
        super(FFM, self).__init__()
        self.conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1h = nn.BatchNorm2d(64)
        self.conv2h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2h = nn.BatchNorm2d(64)
        self.conv3h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3h = nn.BatchNorm2d(64)
        self.conv4h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4h = nn.BatchNorm2d(64)
        self.conv5h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5h = nn.BatchNorm2d(64)

        self.conv1v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1v = nn.BatchNorm2d(64)
        self.conv2v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2v = nn.BatchNorm2d(64)
        self.conv3v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3v = nn.BatchNorm2d(64)
        self.conv4v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4v = nn.BatchNorm2d(64)
        self.conv5v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5v = nn.BatchNorm2d(64)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        out1h = F.relu(self.bn1h(self.conv1h(left)), inplace=True)
        out2h = F.relu(self.bn2h(self.conv2h(out1h)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down)), inplace=True)
        out2v = F.relu(self.bn2v(self.conv2v(out1v)), inplace=True)
        fuse = out2h + out2v
        out3h = F.relu(self.bn3h(self.conv3h(fuse)), inplace=True) * out1h
        out4h = F.relu(self.bn4h(self.conv4h(out3h)), inplace=True) + out1v
        out5h = F.relu(self.bn5h(self.conv5h(out4h)), inplace=True)
        out3v = F.relu(self.bn3v(self.conv3v(fuse)), inplace=True) * out1v
        out4v = F.relu(self.bn4v(self.conv4v(out3v)), inplace=True) + out1h
        out5v = F.relu(self.bn5v(self.conv5v(out4v)), inplace=True)
        return out5h, out5v

    def initialize(self):
        weight_init(self)


class DAM(nn.Module):
    def __init__(self):
        super(DAM, self).__init__()
        self.conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1h = nn.BatchNorm2d(64)
        self.conv2h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2h = nn.BatchNorm2d(64)
        self.conv3h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3h = nn.BatchNorm2d(64)
        self.conv4h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4h = nn.BatchNorm2d(64)
        self.conv1v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1v = nn.BatchNorm2d(64)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        out1h = F.relu(self.bn1h(self.conv1h(left)), inplace=True)
        out2h = F.relu(self.bn2h(self.conv2h(out1h)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down)), inplace=True)
        fuse = out2h * out1v
        out3h = F.relu(self.bn3h(self.conv3h(fuse)), inplace=True) + out1h
        out4h = F.relu(self.bn4h(self.conv4h(out3h)), inplace=True)
        return out4h

    def initialize(self):
        weight_init(self)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.ffm45 = FFM()
        self.ffm34 = FFM()
        self.ffm23 = FFM()

    def forward(self, out2h, out3h, out4h, out5v, edge=None, fback=None, fa_=None, fmix=None):
        if fback is not None and fmix is not None:
            refine5 = F.interpolate(fback, size=out5v.size()[2:], mode='bilinear')
            refine4 = F.interpolate(fback, size=out4h.size()[2:], mode='bilinear')
            refine3 = F.interpolate(fback, size=out3h.size()[2:], mode='bilinear')
            refine2 = F.interpolate(fback, size=out2h.size()[2:], mode='bilinear')
            edge4 = F.interpolate(edge, size=out4h.size()[2:], mode='bilinear')
            edge3 = F.interpolate(edge, size=out3h.size()[2:], mode='bilinear')
            edge2 = F.interpolate(edge, size=out2h.size()[2:], mode='bilinear')

            out4h, out4v = self.ffm45(out4h + edge4, out5v + refine5)
            out3h, out3v = self.ffm34(out3h + edge3, out4v + refine4)
            out2h, pred = self.ffm23(out2h + edge2, out3v + refine3)

            out4h_mix, out4v_mix = self.ffm45(out4h + edge4, fmix + refine5)
            out3h_mix, out3v_mix = self.ffm34(out3h + edge3, out4v_mix + refine4)
            out2h_mix, pred_mix = self.ffm23(out2h + edge2, out3v_mix + refine3)

            out4h_fa_, out4v_mix = self.ffm45(out4h + edge4, fa_ + refine5)
            out3h_fa_, out3v_mix = self.ffm34(out3h + edge3, out4h_fa_ + refine4)
            out2h_fa_, pred_fa_ = self.ffm23(out2h + edge2, out3h_fa_ + refine3)

            return out2h, out3h, out4h, out5v, pred, pred_mix, pred_fa_

        elif fback is not None and fmix is None:
            refine5 = F.interpolate(fback, size=out5v.size()[2:], mode='bilinear')
            refine4 = F.interpolate(fback, size=out4h.size()[2:], mode='bilinear')
            refine3 = F.interpolate(fback, size=out3h.size()[2:], mode='bilinear')
            refine2 = F.interpolate(fback, size=out2h.size()[2:], mode='bilinear')
            edge4 = F.interpolate(edge, size=out4h.size()[2:], mode='bilinear')
            edge3 = F.interpolate(edge, size=out3h.size()[2:], mode='bilinear')
            edge2 = F.interpolate(edge, size=out2h.size()[2:], mode='bilinear')

            out4h, out4v = self.ffm45(out4h + edge4, out5v + refine5)
            out3h, out3v = self.ffm34(out3h + edge3, out4v + refine4)
            out2h, pred = self.ffm23(out2h + edge2, out3v + refine3)
            return out2h, out3h, out4h, out5v, pred

        else:
            out4h, out4v = self.ffm45(out4h, out5v)
            out3h, out3v = self.ffm34(out3h, out4v)
            out2h, pred = self.ffm23(out2h, out3v)
            return out2h, out3h, out4h, out5v, pred

    def initialize(self):
        weight_init(self)


class SANet(nn.Module):
    def __init__(self, path=None):
        super(SANet, self).__init__()
        self.model_path = path
        self.bkbone = ResNet()
        self.squeeze5 = nn.Sequential(nn.Conv2d(2048, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze4 = nn.Sequential(nn.Conv2d(1024, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d(512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.dam = DAM()

        self.decoder1 = Decoder()
        self.refiner = Decoder()
        self.linearp1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearp2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.lineart2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.lineart3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.lineart4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.lineart5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.initialize()

    def forward(self, x, rate=None, flow=None, warp=None, shape=None):
        out1h, out2h, out3h, out4h, out5v = self.bkbone(x)
        out2h, out3h, out4h, out5v = self.squeeze2(out2h), self.squeeze3(out3h), self.squeeze4(out4h), self.squeeze5(
            out5v)

        if rate is not None:
            d = 7
            b, c, W, H = out5v.shape
            out = spatial_correlation_sample(out5v, out5v, kernel_size=1, patch_size=d, stride=1, padding=0)
            out = out.view(b, d * d, W * H)
            indices = torch.argmin(out, dim=1).unsqueeze(1)
            out = torch.zeros_like(out)
            out = out.scatter(1, indices, 1)
            out = out.repeat(1, c, 1)
            unfold = nn.Unfold([d, d], padding=int(d / 2))
            out5v_unfold = unfold(out5v)
            result = torch.mul(out5v_unfold, out)
            result = result.view(b, d * d, c * H * W)
            result = torch.sum(result, dim=1)
            Fa_ = result.view(b, c, W, H)
            mix = rate * out5v + (1 - rate) * Fa_

            oute = self.dam(out1h, out5v)

            out2h, out3h, out4h, out5v, pred1 = self.decoder1(out2h, out3h, out4h, out5v)
            out2t, out3t, out4t, out5t, pred2, pred2mix, pred2Fa_ = self.refiner(out2h, out3h, out4h, out5v, oute,
                                                                                 pred1, Fa_, mix)

            shape = x.size()[2:] if shape is None else shape
            pred1 = F.interpolate(self.linearp1(pred1), size=shape, mode='bilinear')

            pred2o = F.interpolate(self.linearp2(pred2), size=shape, mode='bilinear')
            pred2Fa_ = F.interpolate(self.linearp2(pred2Fa_), size=shape, mode='bilinear')
            pred2mix = F.interpolate(self.linearp2(pred2mix), size=shape, mode='bilinear')

            out2h = F.interpolate(self.linearr2(out2h), size=shape, mode='bilinear')
            out3h = F.interpolate(self.linearr3(out3h), size=shape, mode='bilinear')
            out4h = F.interpolate(self.linearr4(out4h), size=shape, mode='bilinear')

            out2t = F.interpolate(self.lineart2(out2t), size=shape, mode='bilinear')
            out3t = F.interpolate(self.lineart3(out3t), size=shape, mode='bilinear')
            out4t = F.interpolate(self.lineart4(out4t), size=shape, mode='bilinear')

            return pred1, pred2o, pred2Fa_, pred2mix, out2h, out3h, out4h, out2t, out3t, out4t
        else:
            oute = self.dam(out1h, out5v)
            out2h, out3h, out4h, out5v, pred1 = self.decoder1(out2h, out3h, out4h, out5v)
            out2t, out3t, out4t, out5t, pred2 = self.refiner(out2h, out3h, out4h, out5v, oute, pred1)

            shape = x.size()[2:] if shape is None else shape
            pred1 = F.interpolate(self.linearp1(pred1), size=shape, mode='bilinear')
            if flow is not None:
                pred2 = warp(pred2, flow)
            pred2o = F.interpolate(self.linearp2(pred2), size=shape, mode='bilinear')

            out2h = F.interpolate(self.linearr2(out2h), size=shape, mode='bilinear')
            out3h = F.interpolate(self.linearr3(out3h), size=shape, mode='bilinear')
            out4h = F.interpolate(self.linearr4(out4h), size=shape, mode='bilinear')

            out2t = F.interpolate(self.lineart2(out2t), size=shape, mode='bilinear')
            out3t = F.interpolate(self.lineart3(out3t), size=shape, mode='bilinear')
            out4t = F.interpolate(self.lineart4(out4t), size=shape, mode='bilinear')

            return pred1, pred2o, out2h, out3h, out4h, out2t, out3t, out4t

    def initialize(self):
        if self.model_path:
            self.load_state_dict(torch.load(self.model_path))
        else:
            weight_init(self)
