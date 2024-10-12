import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from .UNet_res2net import *
from .GAN import *


def net_factory(num_classes=2, in_chns=3, SC=False, CFA=False, DCF=False):
    model = Net(num_classes=num_classes, in_chns=in_chns, SC=SC, CFA=CFA, DCF=DCF)
    return model


class Net(nn.Module):
    def __init__(self, num_classes, in_chns, SC=False, CFA=False, DCF=False):
        super(Net, self).__init__()
        self.SC = SC
        self.CFA = CFA
        self.DCF = DCF
        self.in_chns = in_chns

        if self.in_chns != 3:
            self.conv0 = nn.Conv2d(in_chns, 3, kernel_size=3, padding=1, bias=False)
        self.encoder = Encoder_Res2Net(n_channels=in_chns, n_classes=num_classes, CFA=CFA)
        if self.SC:
            self.decoder = nn.ModuleList([Decoder_Res2Net(n_class=num_classes, CFA=CFA),
                                          Decoder_Res2Net(n_class=num_classes, CFA=CFA)])
        else:
            self.decoder = Decoder_Res2Net(n_class=num_classes, CFA=CFA)

        if self.DCF:
            self.main_decoder = CA_Decoder(n_class=num_classes)

    def forward(self, x1, x2):
        if self.SC:
            if self.in_chns != 3:
                x1 = self.conv0(x1)
            fea1 = self.encoder(x1)
            out1, df1 = self.decoder[0](fea1)

            if self.in_chns != 3:
                x2 = self.conv0(x2)
            fea2 = self.encoder(x2)
            out2, df2 = self.decoder[1](fea2)

            if self.DCF:
                out3 = self.main_decoder(df1, df2)
            else:
                out3 = None
            return out1, out2, out3
        else:
            out, _ = self.decoder(self.encoder(x1))
            return out, None, None
