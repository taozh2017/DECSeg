import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from .Res2Net_v1b import res2net50_v1b_26w_4s


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if in_channels // out_channels == 4:
            self.conv0 = nn.ConvTranspose2d(in_channels, in_channels // 4, kernel_size=2, stride=2)
            self.conv1 = DoubleConv(in_channels // 2, out_channels)
        else:
            self.conv0 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv1 = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.conv0(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv1(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


###  CFA  ###
class Up_CFA(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if in_channels // out_channels == 4:
            self.conv0 = nn.ConvTranspose2d(in_channels, in_channels // 4, kernel_size=2, stride=2)
            self.conv1 = DoubleConv(in_channels // 4 + 64, out_channels)
        else:
            self.conv0 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv1 = DoubleConv(in_channels // 2 + 64, out_channels)

    def forward(self, x1, x2):
        x1 = self.conv0(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv1(x)


class Encoder_Res2Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, CFA=False):
        super(Encoder_Res2Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.CFA = CFA
        self.ft_chns = [64, 256, 512, 1024, 2048]

        # ---- ResNet Backbone ----

        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        # Cross-level Aggregation
        if self.CFA:
            self.CFA1 = Crosslevel_Aggregation(in_channels1=self.ft_chns[0], in_channels2=self.ft_chns[1],
                                               base_channel=32)
            self.CFA2 = Crosslevel_Aggregation(in_channels1=self.ft_chns[1], in_channels2=self.ft_chns[2],
                                               base_channel=32)
            self.CFA3 = Crosslevel_Aggregation(in_channels1=self.ft_chns[2], in_channels2=self.ft_chns[3],
                                               base_channel=32)
            self.CFA4 = Crosslevel_Aggregation(in_channels1=self.ft_chns[3], in_channels2=self.ft_chns[4],
                                               base_channel=32)

    def forward(self, x):
        x1 = self.resnet.conv1(x)
        x1 = self.resnet.bn1(x1)
        x1 = self.resnet.relu(x1)  # bs, 64, 96, 96 (176, 176)
        x2 = self.resnet.maxpool(x1)  # bs, 64, 48, 48 (88, 88)
        x2 = self.resnet.layer1(x2)  # bs, 256, 48, 48 (88, 88)
        x3 = self.resnet.layer2(x2)  # bs, 512, 24, 24 (44, 44)
        x4 = self.resnet.layer3(x3)  # bs, 1024, 12, 12 (22, 22)
        x5 = self.resnet.layer4(x4)  # bs, 2048, 6, 6 (11, 11)

        if self.CFA:
            # x1+x2->x1
            x1 = self.CFA1(x1, x2)  # bs, 64, 48, 48 (88, 88)
            # x2+x3->x2
            x2 = self.CFA2(x2, x3)  # bs, 64, 48, 48 (88, 88)
            # x3+x4->x3
            x3 = self.CFA3(x3, x4)  # bs, 64, 24, 24 (44, 44)
            # x4+x5->x4
            x4 = self.CFA4(x4, x5)  # bs, 64, 12, 12 (22, 22)

        return [x1, x2, x3, x4, x5]


class Decoder_Res2Net(nn.Module):
    def __init__(self, n_class=2, CFA=False):
        super(Decoder_Res2Net, self).__init__()
        self.n_class = n_class
        self.ft_chns = [64, 256, 512, 1024, 2048]
        if CFA:
            up = Up_CFA
        else:
            up = Up
        self.up1 = up(self.ft_chns[4], self.ft_chns[3])
        self.up2 = up(self.ft_chns[3], self.ft_chns[2])
        self.up3 = up(self.ft_chns[2], self.ft_chns[1])
        self.up4 = up(self.ft_chns[1], self.ft_chns[0])
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, x):
        x1 = x[0]  # 4*64*88*88
        x2 = x[1]  # 4*64*88*88
        x3 = x[2]  # 4*64*44*44
        x4 = x[3]  # 4*64*22*22
        x5 = x[4]  # 4*2048*11*11

        x4 = self.up1(x5, x4)  # 4*1024*22*22
        x3 = self.up2(x4, x3)  # 4*512*44*44
        x2 = self.up3(x3, x2)  # 4*256*88*88
        x1 = self.up4(x2, x1)  # 4*64*88*88

        logits = self.out_conv(x1)
        logits = F.interpolate(logits, scale_factor=2, mode='bilinear', align_corners=False)
        return logits, [x5, x4, x3, x2, x1]


class MyRes2Net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(MyRes2Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.Encoder = Encoder_Res2Net(n_channels=self.n_channels, n_classes=self.n_classes)
        self.Decoder = Decoder_Res2Net(n_class=self.n_classes)

    def forward(self, x):
        features = self.Encoder(x)
        out = self.Decoder(features)
        return out


class GAP(nn.Module):
    def __init__(self, channels=64, r=4):
        super(GAP, self).__init__()
        out_channels = int(channels // r)

        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):
        xg = self.global_att(x)
        wei = self.sig(xg)

        return wei


class Crosslevel_Aggregation(nn.Module):
    def __init__(self, in_channels1, in_channels2, base_channel=32):
        super(Crosslevel_Aggregation, self).__init__()
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(in_channels1, base_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(base_channel),
            nn.ReLU(inplace=True)
        )

        # self.up = nn.ConvTranspose2d(in_channels2, in_channels2, kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(in_channels2, base_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(base_channel),
            nn.ReLU(inplace=True)
        )

        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(base_channel * 2, base_channel * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channel * 2),
            nn.ReLU(inplace=True)
        )

        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(base_channel * 2, base_channel * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channel * 2),
            nn.ReLU(inplace=True)
        )

        self.GAP = GAP(channels=base_channel * 2, r=4)

    def forward(self, x1, x2):
        x1 = self.conv1x1_1(x1)
        x2 = self.up(x2)

        x2 = self.conv1x1_2(x2)

        x = torch.cat([x1, x2], dim=1)
        x = self.conv3x3_1(x)

        weight = self.GAP(x)
        x = x + x * weight

        x = self.conv3x3_2(x)

        return x


class CA_Decoder(nn.Module):
    def __init__(self, n_class=1):
        super(CA_Decoder, self).__init__()
        up = Up
        self.n_class = n_class
        self.ft_chns = [64, 256, 512, 1024, 2048]

        self.dcf_out1 = Dualscale_Complementary_Fusion(self.ft_chns[4])
        self.dcf_out2 = Dualscale_Complementary_Fusion(self.ft_chns[3])
        self.dcf_out3 = Dualscale_Complementary_Fusion(self.ft_chns[2])
        self.dcf_out4 = Dualscale_Complementary_Fusion(self.ft_chns[1])
        self.dcf_out5 = Dualscale_Complementary_Fusion(self.ft_chns[0])

        self.conv = DoubleConv(self.ft_chns[4], self.ft_chns[4])

        self.up1 = up(self.ft_chns[4], self.ft_chns[3])
        self.up2 = up(self.ft_chns[3], self.ft_chns[2])
        self.up3 = up(self.ft_chns[2], self.ft_chns[1])
        self.up4 = up(self.ft_chns[1], self.ft_chns[0])

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, x1, x2):
        x15, x14, x13, x12, x11 = x1
        x25, x24, x23, x22, x21 = x2
        # 2048,1024,512,256,64

        x35 = self.dcf_out1(x15, x25)  # bs*2048*11*11
        x34 = self.dcf_out2(x14, x24)
        x33 = self.dcf_out3(x13, x23)
        x32 = self.dcf_out4(x12, x22)
        x31 = self.dcf_out5(x11, x21)

        x = self.conv(x35)
        x = self.up1(x, x34)  # bs*1024*22*22
        x = self.up2(x, x33)  # bs*512*44*44
        x = self.up3(x, x32)  # bs*256*88*88
        x = self.up4(x, x31)  # bs*64*88*88
        logits = self.out_conv(x)
        logits = F.interpolate(logits, scale_factor=2, mode='bilinear', align_corners=False)
        return logits


class Dualscale_Complementary_Fusion(nn.Module):
    def __init__(self, channel):
        super(Dualscale_Complementary_Fusion, self).__init__()
        # self.up = nn.ConvTranspose2d(channel, channel, kernel_size=2, stride=2)
        self.CBR1_1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.CBR2_1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.CBR1_2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True))
        self.CBR2_2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True))

        self.CBR3 = nn.Sequential(
            nn.Conv2d(channel * 2, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True))
        self.sig = nn.Sigmoid()

    def forward(self, x1, x2):
        bs, c, h, w = x1.shape
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear', align_corners=True)
        # x2 = self.up(x2)

        x1 = self.CBR1_1(x1)
        x2 = self.CBR2_1(x2)

        w_x1 = self.sig(x1)
        w_x2 = self.sig(x2)

        a_x1 = x1 + w_x2 * x2
        a_x2 = x2 + w_x1 * x1

        a_x1 = self.CBR1_2(a_x1)
        a_x2 = self.CBR2_2(a_x2)

        x = torch.cat([a_x1, a_x2], dim=1)

        g = self.CBR3(x)
        g = nn.Softmax(dim=1)(g)

        x = a_x1 * g[:, :1] + a_x2 * g[:, 1:]

        return x


class Dualscale_Complementary_Fusion_base(nn.Module):
    def __init__(self, channel):
        super(Dualscale_Complementary_Fusion_base, self).__init__()
        # self.up = nn.ConvTranspose2d(channel, channel, kernel_size=2, stride=2)
        self.CBR1 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True))
        self.CBR2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        bs, c, h, w = x1.shape
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear', align_corners=True)
        # x2 = self.up(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.CBR2(self.CBR1(x))

        return x
