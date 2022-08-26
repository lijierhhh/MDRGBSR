import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from moco.builder import MoCo
import numpy as np

def make_model(args):
    return BlindSR(args)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class MSRB_Block(nn.Module):
    def __init__(self):
        super(MSRB_Block, self).__init__()

        self.conv_1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_1_2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_3_2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_5_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv_5_2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2, bias=True)
        self.confusion = nn.Conv2d(in_channels=576, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity_data = x
        output_1_1 = self.relu(self.conv_1_1(x))
        output_3_1 = self.relu(self.conv_3_1(x))
        output_5_1 = self.relu(self.conv_5_1(x))
        input_2 = torch.cat([output_1_1, output_3_1, output_5_1], 1)#
        output_1_2 = self.relu(self.conv_1_2(input_2))#
        output_3_2 = self.relu(self.conv_3_2(input_2))#
        output_5_2 = self.relu(self.conv_5_2(input_2))#
        output = torch.cat([output_1_2, output_3_2, output_5_2], 1)#
        output = self.confusion(output)#
        output = torch.add(output, identity_data)#
        return output

class DGFEM(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DGFEM, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64 * self.kernel_size * self.kernel_size, bias=False)
        )
        self.conv = common.default_conv(channels_in, channels_out, 1)
        self.ca = CA_layer(channels_in, channels_out, reduction)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        b, c, h, w = x[0].size()
        out = self.ca(x)

        kernel = self.kernel(x[1]).view(-1, 1, self.kernel_size, self.kernel_size)
        out = self.relu(F.conv2d(out.view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))#==》torch.Size([1, 512, 256, 256])
        out = self.conv(out.view(b, -1, h, w))

        return out


class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.conv_du(x[1][:, :, None, None])
        return x[0] * att


class DGFEB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(DGFEB, self).__init__()

        self.da_conv1 = DGFEM(n_feat, n_feat, kernel_size, reduction)
        self.da_conv2 = DGFEM(n_feat, n_feat, kernel_size, reduction)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)
        self.MSRB_block = MSRB_Block()

        self.relu =  nn.LeakyReLU(0.1, True)

    def forward(self, x):
        out = self.relu(self.da_conv1(x))
        out = self.relu(self.conv1(out))
        out = self.relu(self.da_conv2([out, x[1]]))
        out = self.conv2(out) + x[0]
        out = self.MSRB_block(out)#


        return out
        # plt.figure();plt.imshow(out.squeeze(0))


class Res_block(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_blocks):
        super(Res_block, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            DGFEB(conv, n_feat, kernel_size, reduction) \
            for _ in range(n_blocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = x[0]
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1]])
        res = self.body[-1](res)
        res = res + x[0]

        return res


class MDRGBSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(MDRGBSR, self).__init__()

        self.n_groups = 5
        n_blocks = 5
        n_feats = 64
        kernel_size = 3
        reduction = 8
        # scale = 4
        scale = int(args.scale[0])

        rgb_mean = (0.39779539, 0.40924516, 0.36850663)
        rgb_std = (1.0, 1.0, 1.0)

        self.sub_mean = common.MeanShift(255.0, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(255.0, rgb_mean, rgb_std, 1)

        modules_head = [conv(3, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)
        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )

        modules_body = [
            Res_block(common.default_conv, n_feats, kernel_size, reduction, n_blocks) \
            for _ in range(self.n_groups)
        ]
        modules_body.append(conv(n_feats*6, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

        modules_tail = [common.Upsampler(conv, scale, n_feats, act=False),
                        conv(n_feats, 3, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, k_v):
        k_v = self.compress(k_v)
        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        out = res
        for i in range(self.n_groups):
            res = self.body[i]([res, k_v])
            res = torch.add(res, x)
            out = torch.cat([out, res], 1)

        res = self.body[-1](out)
        res = res + x
        x = self.tail(res)

        x = self.add_mean(x)

        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out


class BlindSR(nn.Module):
    def __init__(self, args):
        super(BlindSR, self).__init__()

        self.G = MDRGBSR(args)

        self.E = MoCo(base_encoder=Encoder)

    def forward(self, x):
        if self.training:
            x_query = x[:, 0, ...]
            x_key = x[:, 1, ...]
            fea, logits, labels = self.E(x_query, x_key)

            sr = self.G(x_query, fea)

            return sr, logits, labels
        else:
            fea = self.E(x, x)

            sr = self.G(x, fea)

            return sr

def main():
    pass
    input = [torch.Tensor(8, 8, 64, 256, 256), torch.Tensor(8, 64)]
    MoCo_TEST = MoCo(base_encoder=Encoder)
    x_query = input[0][:, 0, ...]
    x_key = input[0][:, 1, ...]
    fea, logits, labels = MoCo_TEST(x_query, x_key)

if __name__ == '__main__':
    main()