import torch
import torch.nn as nn
import numpy as np
import math
from collections import OrderedDict
from .layer import MultiSpectralAttentionLayer

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        with torch.no_grad():
            self.A = torch.from_numpy(A.astype(np.float32))
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1, stride=1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1, stride=1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1, stride=1))

        if in_channels != out_channels:
            self.down = nn.Sequential(

                nn.Conv2d(in_channels, out_channels, 1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())


        y = None

        for i in range (self.num_subset):

                A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
                A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
                A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))
                A1 = A1 + A[i] + self.PA[i]
                A2 = x.view(N,C*T,V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


def conv1x1(in_channels, out_channels, postfix, stride=1, kernel_size=1, padding=0):
    """1x1 convolution"""
    return [
        ('{}/conv'.format(postfix),
         nn.Conv2d(in_channels, out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   # groups=groups,
                   bias=False)),
        ('{}/norm'.format(postfix),
         nn.BatchNorm2d(out_channels)),
        ('{}/relu'.format(postfix),
         nn.ReLU(inplace=True)),
    ]

def conv3x3(in_channels, out_channels, stride, kernel_size, padding, postfix):
    """3x3 convolution with padding"""
    return [
        ('{}/conv'.format(postfix),
         nn.Conv2d(in_channels, out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   # groups=groups,
                   bias=False)),
        ('{}/norm'.format(postfix),
         nn.BatchNorm2d(out_channels)),
        ('{}/relu'.format(postfix),
         nn.ReLU(inplace=True)),
    ]

class PyConv2(nn.Module):

    def __init__(self, inplans, planes,pyconv_kernels=[3, 5], stride=1, pyconv_groups=[1, 4]):
        super(PyConv2, self).__init__()
        self.conv2_1 = nn.Conv2d(inplans, planes // 2, kernel_size=(pyconv_kernels[0],1), padding=(pyconv_kernels[0] // 2,0),
                            groups=pyconv_groups[0])
        self.conv2_2 = nn.Conv2d(inplans, planes // 2, kernel_size=(pyconv_kernels[1],1), padding=(pyconv_kernels[1] // 2,0),
                            groups=pyconv_groups[1])
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(torch.cat((self.conv2_1(x), self.conv2_2(x)), dim=1)))
class tcn_vov(nn.Module):
    def __init__(self, out_channels, stride=1):
        super(tcn_vov, self).__init__()
        self.layers = nn.ModuleList()
        in_channel = out_channels
        stage_ch = out_channels // 2
        kernel_size = [1, 5, 9]
        # stride = (1, 1)
        padding_s = (kernel_size[0] - 1) // 2
        padding_m = (kernel_size[1] - 1) // 2
        padding_l = (kernel_size[2] - 1) // 2
        padding = [padding_s, padding_m, padding_l]
        self.layers.append(nn.Sequential(
            OrderedDict(conv3x3(in_channels=in_channel, out_channels=stage_ch, stride=(1, 1),
                                kernel_size=(kernel_size[0], 1), padding=(padding[0], 0), postfix='layer'))))
        self.layers.append(nn.Sequential(PyConv2(stage_ch, stage_ch,pyconv_kernels=[3,5])))
        self.layers.append(nn.Sequential(PyConv2(stage_ch, stage_ch, pyconv_kernels=[7,9])))

        in_channel = out_channels + 3 * stage_ch
        self.concat = nn.Sequential(
            OrderedDict(conv1x1(in_channel, out_channels, stride=(stride, 1), postfix='concat')))
        c2wh = dict([(64, 300), (128, 150), (256, 75)])
        self.ese = MultiSpectralAttentionLayer(out_channels, c2wh[out_channels], 25, reduction=4,
                                               freq_sel_method='top16')


    def forward(self, x):
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        x = self.concat(x)
        x = self.ese(x)
        return x
class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        inter_channels = out_channels
        self.gcn1 = unit_gcn(in_channels, inter_channels, A)
        self.tcn_vov = tcn_vov(out_channels, stride=stride)
        self.relu = nn.ReLU()
        if stride == 2:
            self.down = unit_tcn(inter_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.down = lambda x: x

    def forward(self, x):
        x1 = self.gcn1(x)
        identity_feat = x1
        xt = self.tcn_vov(x1)
        x = xt + self.down(identity_feat)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)
        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)
