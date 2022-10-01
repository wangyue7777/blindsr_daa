import torch.nn as nn
import torch
import model.IMBD_block as B

import model.common as common
from model.drconv_my1 import asign_index
import torch.nn.functional as F


class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.region_number = 4

        self.kernel = nn.Sequential(
            nn.Conv1d(channels_in, self.region_number * self.region_number, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv1d(self.region_number * self.region_number, channels_out * self.region_number * self.kernel_size * self.kernel_size,
                      kernel_size=1, groups=self.region_number, bias=False)
        )
        self.conv = common.default_conv(channels_in, channels_out, 1)
        self.ca = CA_layer(channels_in, channels_out, reduction)
        self.conv_guide1 = nn.Conv2d(channels_in, self.region_number, kernel_size=kernel_size, padding=1)
        self.conv_guide2 = nn.Conv2d(channels_in, self.region_number, kernel_size=kernel_size, padding=1)
        self.asign_index1 = asign_index.apply
        self.asign_index2 = asign_index.apply

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].size()

        # branch 1

        kernel = self.kernel(x[1].unsqueeze(2)).squeeze(2)
        kernel = kernel.view(b, -1, c, self.kernel_size, self.kernel_size).transpose(1, 2).contiguous()
        kernel = kernel.view(-1, 1, self.kernel_size, self.kernel_size)
        out = self.relu(F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))
        # out = out.view(b, -1, h, w).view(b, self.region_number, -1, h, w)
        # out = out.view(-1, self.region_number, h, w).view(b, -1, self.region_number, h, w)
        out = out.view(b, -1, self.region_number, h, w)
        out = out.transpose(1, 2)
        guide_feature1 = self.conv_guide1(x[0])    # b, r, h, w
        out = self.asign_index1(out, guide_feature1)
        out = self.conv(out)

        # branch 2
        out2 = self.ca(x)   # b,r,c,h,w
        guide_feature2 = self.conv_guide2(x[0])    # b, r, h, w
        out2 = self.asign_index2(out2, guide_feature2)
        out = out + out2

        return out


class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.region_number = 4
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, self.region_number * self.region_number, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.region_number * self.region_number, channels_out * self.region_number, 1, 1, 0, 
            groups=self.region_number, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].size()

        att = self.conv_du(x[1].unsqueeze(2).unsqueeze(3))
        att = att.view(b, -1, c, 1, 1)   # b,r,c,1,1
        out = x[0].unsqueeze(1) * att   # b,r,c,h,w

        return out  


class DAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size=3, reduction=8):
        super(DAB, self).__init__()

        self.compress = nn.Sequential(
            nn.Linear(256, n_feat, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feat, n_feat, bias=False),
            nn.LeakyReLU(0.1, True)
        )

        self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.da_conv2 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu =  nn.LeakyReLU(0.1, True)

    def forward(self, x, rep):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        x1 = self.compress(rep)

        out = self.relu(self.da_conv1([x, x1]))
        out = self.relu(self.conv1(out))
        out = self.relu(self.da_conv2([out, x1]))
        out = self.conv2(out)

        return out


class IMDN(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=4):
        super(IMDN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        # IMDBs
        self.adapt0 = DAB(common.default_conv, nf)

        self.IMDB1 = B.IMDModule(in_channels=nf)
        self.adapt1 = DAB(common.default_conv, nf)

        self.IMDB2 = B.IMDModule(in_channels=nf)
        self.adapt2 = DAB(common.default_conv, nf)

        self.IMDB3 = B.IMDModule(in_channels=nf)
        self.adapt3 = DAB(common.default_conv, nf)

        self.IMDB4 = B.IMDModule(in_channels=nf)
        self.adapt4 = DAB(common.default_conv, nf)

        self.IMDB5 = B.IMDModule(in_channels=nf)
        self.adapt5 = DAB(common.default_conv, nf)

        self.IMDB6 = B.IMDModule(in_channels=nf)
        self.adapt6 = DAB(common.default_conv, nf)

        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)


    def forward(self, input, d_r):

        diffs = []

        out_fea = self.fea_conv(input)
        diff = self.adapt0(out_fea, d_r)
        diffs.append(diff)
        out_fea = out_fea + diff
        
        out_B1 = self.IMDB1(out_fea)
        diff = self.adapt1(out_B1, d_r)
        diffs.append(diff)
        out_B1 = out_B1 + diff

        out_B2 = self.IMDB2(out_B1)
        diff = self.adapt2(out_B2, d_r)
        diffs.append(diff)
        out_B2 = out_B2 + diff

        out_B3 = self.IMDB3(out_B2)
        diff = self.adapt3(out_B3, d_r)
        diffs.append(diff)
        out_B3 = out_B3 + diff

        out_B4 = self.IMDB4(out_B3)
        diff = self.adapt4(out_B4, d_r)
        diffs.append(diff)
        out_B4 = out_B4 + diff

        out_B5 = self.IMDB5(out_B4)
        diff = self.adapt5(out_B5, d_r)
        diffs.append(diff)
        out_B5 = out_B5 + diff

        out_B6 = self.IMDB6(out_B5)
        diff = self.adapt6(out_B6, d_r)
        diffs.append(diff)
        out_B6 = out_B6 + diff

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        
        return output, diffs