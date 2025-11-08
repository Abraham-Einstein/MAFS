import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from INN_block import FeatureExtract
from Fusion_block import ARM, ARM2, CAM, ASPP
from util.EAEF import EAEF
from util.CAI_ARM import ARM_I, ARM_II

up_kwargs = {'mode': 'bilinear', 'align_corners': True}


def conv1(in_chsnnels, out_channels):
    "1x1 convolution with padding"
    return nn.Conv2d(in_chsnnels, out_channels, kernel_size=1, stride=1, bias=False)


def conv3(in_chsnnels, out_channels):
    "3x3 convolution with padding"
    return nn.Conv2d(in_chsnnels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)


class FReLU(nn.Module):
    r""" FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)
    """

    def __init__(self, in_channels):
        super().__init__()
        self.conv_frelu = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.bn_frelu = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x1 = self.conv_frelu(x)
        x1 = self.bn_frelu(x1)
        x2 = torch.stack([x, x1], dim=0)
        out, _ = torch.max(x2, dim=0)
        return out

# ---------------------- shallow feature extractor ------------------------ #


class Feature_extract(nn.Module):
    '''
    特征提取模块
    '''

    def __init__(self, in_channels, out_channels):
        super(Feature_extract, self).__init__()
        self.SFEB1 = nn.Sequential(
            nn.Conv2d(in_channels, int(out_channels / 2), kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(int(out_channels / 2)),
            FReLU(int(out_channels / 2)),
            nn.Conv2d(int(out_channels / 2), int(out_channels / 2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(out_channels / 2)),
            FReLU(int(out_channels / 2)),
        )
        self.SFEB2 = nn.Sequential(
            nn.Conv2d(int(out_channels / 2), out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            FReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1), )

    def forward(self, x):
        high_x = self.SFEB1(x)  # in_ch---> out_ch / 2
        x = self.SFEB2(high_x)  # out_ch / 2 ----> out_ch
        return high_x, x
# ------------------------------------------------------------- #

class S2M(nn.Module):
    '''
    Scene Specific Mask
    '''

    # def __init__(self, channels, r=4):
    def __init__(self, channels, r=1):
        super(S2M, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_block = nn.BatchNorm2d(channels)

    def forward(self, x):
        # spatial attention
        local_w = self.local_att(x)  ## local attention
        ## channel attention
        global_w = self.global_att(x)
        mask = self.sigmoid(local_w * global_w)
        masked_feature = mask * x
        output = self.conv_block(masked_feature)
        return output


class Prediction_head(nn.Module):
    '''
    自适应特征连接模块, 用于跳变连接的自适应连接 Adaptive_Connection
    '''

    def __init__(self, channels, img=False):
        super(Prediction_head, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        return (self.conv_block(x) + 1) / 2


class SFP(nn.Module):
    '''
    Scene Fidelity Path
    '''

    def __init__(self, channels, img=False):
        super(SFP, self).__init__()
        self.mask = S2M(channels[0])
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.mask(x)
        return (self.conv_block(x) + 1) / 2


class SIM(nn.Module):

    def __init__(self, norm_nc, label_nc, nhidden=64):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)
        # self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        # nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=norm_nc)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)  # torch.Size([4, 32, 128, 128])  torch.Size([4, 32, 256, 256])
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear')
        actv = self.mlp_shared(segmap)
        # actv = segmap
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # apply scale and bias
        out = self.bn(normalized * (1 + gamma)) + beta  # torch.Size([4, 32, 128, 128])  torch.Size([4, 32, 256, 256])

        return out


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.num_resnet_layers = 152
        if self.num_resnet_layers == 18:
            resnet_raw_model1 = models.resnet18(pretrained=True)
            resnet_raw_model2 = models.resnet18(pretrained=True)

        elif self.num_resnet_layers == 34:
            resnet_raw_model1 = models.resnet34(pretrained=True)  # resnet_model
            resnet_raw_model2 = models.resnet34(pretrained=True)

        elif self.num_resnet_layers == 50:
            resnet_raw_model1 = models.resnet50(pretrained=True)
            resnet_raw_model2 = models.resnet50(pretrained=True)

        elif self.num_resnet_layers == 101:
            resnet_raw_model1 = models.resnet101(pretrained=True)
            resnet_raw_model2 = models.resnet101(pretrained=True)

        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = models.resnet152(pretrained=True)
            resnet_raw_model2 = models.resnet152(pretrained=True)

        self.dims = [32, 32, 64, 64, 64, 64]
        self.decoder_dim_rec = 32
        self.decoder_dim_seg = 64

        ########  Thermal ENCODER  ########
        self.encoder_thermal_conv1 = Feature_extract(1, 64)  # one channel ----- thermal
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer3 = resnet_raw_model1.layer1
        self.encoder_thermal_layer4 = resnet_raw_model1.layer2
        self.encoder_thermal_layer5 = resnet_raw_model1.layer3
        self.encoder_thermal_layer6 = resnet_raw_model1.layer4

        ########  RGB ENCODER  ########
        self.encoder_rgb_conv1 = Feature_extract(3, 64)  # three channels --- rgb
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer3 = resnet_raw_model2.layer1
        self.encoder_rgb_layer4 = resnet_raw_model2.layer2
        self.encoder_rgb_layer5 = resnet_raw_model2.layer3
        self.encoder_rgb_layer6 = resnet_raw_model2.layer4

        self.SFEB2 = nn.Sequential(
            nn.Conv2d(int(self.decoder_dim_seg / 2), self.decoder_dim_seg, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.decoder_dim_seg),
            FReLU(self.decoder_dim_seg),
            nn.Conv2d(self.decoder_dim_seg, self.decoder_dim_seg, kernel_size=3, stride=1, padding=1), )

        # ---------------------------------------------- #
        self.high_fuse6 = CAM(2048, 64)
        self.high_fuse5 = CAM(1024, 64)
        # self.high_fuse5 = VisionTransformer(config_vit, img_size)
        self.high_fuse4 = CAM(512, 64)
        self.low_fuse3 = CAM(256, 64)
        # ---------------------------------------------- #

        self.EAEF5 = EAEF(1024)
        self.EAEF4 = EAEF(512)
        self.EAEF3 = EAEF(256)
        self.EAEF2 = EAEF(64)

        self.low_fuse2 = SDFM(64, 32)
        self.low_fuse1 = SDFM(32, 32)


        self.SIM1 = SIM(norm_nc=32, label_nc=64, nhidden=32)
        self.SIM2 = SIM(norm_nc=32, label_nc=32, nhidden=32)


        self.to_fused_seg = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, self.decoder_dim_seg, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2 ** i, mode='bilinear', align_corners=True)  # upsample
        ) for i, dim in enumerate(self.dims[2:])])

        self.SIM3 = SIM(norm_nc=32, label_nc=64, nhidden=32)

        self.seg_decoder = S2PM(4 * self.decoder_dim_seg, self.decoder_dim_seg)
        self.rec_decoder = DSRM(self.decoder_dim_rec, self.decoder_dim_rec)
        self.ARM = ARM()

    def forward(self, rgb, depth):
        rgb = rgb
        thermal = depth[:, :1, ...]

        rgb1, rgb2 = self.encoder_rgb_conv1(rgb)  # (240, 320)
        rgb2 = self.encoder_rgb_bn1(rgb2)  # (240, 320)
        rgb2 = self.encoder_rgb_relu(rgb2)  # (240, 320)

        thermal1, thermal2 = self.encoder_thermal_conv1(thermal)  # (240, 320)
        thermal2 = self.encoder_thermal_bn1(thermal2)  # (240, 320)
        thermal2 = self.encoder_thermal_relu(thermal2)  # (240, 320)

        in_rgbt_2 = torch.cat([rgb2, thermal2], dim=1)
        out_rgbt_2 = self.EAEF2(in_rgbt_2)
        rgb2 = out_rgbt_2[:, 0:64, :, :]
        thermal2 = out_rgbt_2[:, 64:128, :, :]

        ######################################################################
        rgb3 = self.encoder_rgb_maxpool(rgb2)  # (120, 160)
        thermal3 = self.encoder_thermal_maxpool(thermal2)  # (120, 160)
        rgb3 = self.encoder_rgb_layer3(rgb3)  # (120, 160)
        thermal3 = self.encoder_thermal_layer3(thermal3)  # (120, 160)

        in_rgbt_3 = torch.cat([rgb3, thermal3], dim=1)
        out_rgbt_3 = self.EAEF3(in_rgbt_3)
        rgb3 = out_rgbt_3[:, 0:256, :, :]
        thermal3 = out_rgbt_3[:, 256:512, :, :]

        ######################################################################
        rgb4 = self.encoder_rgb_layer4(rgb3)  # (60, 80)
        thermal4 = self.encoder_thermal_layer4(thermal3)  # (60, 80)

        in_rgbt_4 = torch.cat([rgb4, thermal4], dim=1)
        out_rgbt_4 = self.EAEF4(in_rgbt_4)
        rgb4 = out_rgbt_4[:, 0:512, :, :]
        thermal4 = out_rgbt_4[:, 512:1024, :, :]

        ######################################################################
        rgb5 = self.encoder_rgb_layer5(rgb4)  # (30, 40)
        thermal5 = self.encoder_thermal_layer5(thermal4)  # (30, 40)

        in_rgbt_5 = torch.cat([rgb5, thermal5], dim=1)
        out_rgbt_5 = self.EAEF5(in_rgbt_5)
        rgb5 = out_rgbt_5[:, 0:1024, :, :]
        thermal5 = out_rgbt_5[:, 1024:2048, :, :]

        ######################################################################
        rgb6 = self.encoder_rgb_layer6(rgb5)  # (30, 40)
        thermal6 = self.encoder_thermal_layer6(thermal5)  # (30, 40)

        ## fused featrue

        fused_f6 = self.high_fuse6(rgb6, thermal6)  # torch.Size([8, 64, 8, 8])  #  PSFM 8
        fused_f5 = self.high_fuse5(rgb5, thermal5)  # torch.Size([8, 64, 16, 16])  #  4
        fused_f4 = self.high_fuse4(rgb4, thermal4)  # torch.Size([8, 64, 32, 32])  # 2
        fused_f3 = self.low_fuse3(rgb3, thermal3)   # torch.Size([8, 64, 64, 64])

        fused_f2 = self.low_fuse2(rgb2, thermal2)  # torch.Size([8, 32, 128, 128])
        fused_f1 = self.low_fuse1(rgb1, thermal1)  # torch.Size([8, 32, 256, 256])

        encoded_featrues_seg = [fused_f3, fused_f4, fused_f5, fused_f6]  # cat

        # ----------------------------------------------- #
        rec_f1 = self.SIM1(fused_f2, fused_f3)
        rec_f1_ = self.SFEB2(rec_f1)

        rec_f = self.SIM2(fused_f1, rec_f1)  # PSIM--------rec_f  torch.Size([8, 32, 256, 256])

        seg_fused_f = [to_fused(output) for output, to_fused in zip(encoded_featrues_seg, self.to_fused_seg)]  # 64

        # ------------------------------ #
        if seg_fused_f[3].shape == torch.Size([1, 64, 184, 320]):  # pst900
            seg_fused_f[3] = F.interpolate(seg_fused_f[3], size=(180, 320), mode='bilinear', align_corners=False)
        if seg_fused_f[3].shape == torch.Size([1, 64, 152, 200]):  # fmb
            seg_fused_f[3] = F.interpolate(seg_fused_f[3], size=(150, 200), mode='bilinear', align_corners=False)
        if seg_fused_f[2].shape == torch.Size([1, 64, 152, 200]):  # fmb
            seg_fused_f[2] = F.interpolate(seg_fused_f[2], size=(150, 200), mode='bilinear', align_corners=False)
        # print(seg_fused_f[3].shape)
        # ----------------------------- #

        seg_f = torch.cat(seg_fused_f, dim=1)  # torch.Size([4, 256, 64, 64])
        seg_f = torch.cat([rec_f1_, seg_f], dim=1)
        seg_f = self.ARM(seg_f)

        ## image reconstruction
        ## visible image
        rec_f = self.rec_decoder(rec_f)  # DSRM

        return rec_f, rec_f1, seg_f, encoded_featrues_seg


class SDFM(nn.Module):
    def __init__(self, in_C, out_C):
        super(SDFM, self).__init__()
        self.RGBobj1_1 = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.RGBobj1_2 = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.RGBspr = BBasicConv2d(out_C, out_C, 3, 1, 1)

        self.Infobj1_1 = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.Infobj1_2 = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.Infspr = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.obj_fuse = Fusion_module(channels=out_C)

    def forward(self, rgb, depth):
        rgb_sum = self.RGBobj1_2(self.RGBobj1_1(rgb))
        rgb_obj = self.RGBspr(rgb_sum)
        Inf_sum = self.Infobj1_2(self.Infobj1_1(depth))
        Inf_obj = self.Infspr(Inf_sum)
        # print('Inf_obj:', Inf_obj.shape)
        out = self.obj_fuse(rgb_obj, Inf_obj)
        return out


class Fusion_module(nn.Module):
    '''
    基于注意力的自适应特征聚合 Fusion_Module
    '''

    def __init__(self, channels=64, r=4):
        super(Fusion_module, self).__init__()
        inter_channels = int(channels // r)

        self.Recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, 2 * inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * inter_channels, 2 * channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * channels),
            nn.Sigmoid(),
        )

        self.channel_agg = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        _, c, _, _ = x1.shape
        input = torch.cat([x1, x2], dim=1)
        recal_w = self.Recalibrate(input)
        recal_input = recal_w * input  ## 先对特征进行一步自校正
        recal_input = recal_input + input
        x1, x2 = torch.split(recal_input, c, dim=1)
        agg_input = self.channel_agg(recal_input)  ## 进行特征压缩 因为只计算一个特征的权重
        local_w = self.local_att(agg_input)  ## 局部注意力 即spatial attention
        global_w = self.global_att(agg_input)  ## 全局注意力 即channel attention
        w = self.sigmoid(local_w * global_w)  ## 计算特征x1的权重
        xo = w * x1 + (1 - w) * x2  ## fusion results ## 特征聚合
        return xo


class GEFM(nn.Module):
    def __init__(self, in_C, out_C):
        super(GEFM, self).__init__()
        self.RGB_K = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.RGB_V = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.Q = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.INF_K = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.INF_V = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.Second_reduce = BBasicConv2d(in_C, out_C, 3, 1, 1)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        Q = self.Q(torch.cat([x, y], dim=1))
        RGB_K = self.RGB_K(x)
        RGB_V = self.RGB_V(x)
        m_batchsize, C, height, width = RGB_V.size()
        RGB_V = RGB_V.view(m_batchsize, -1, width * height)
        RGB_K = RGB_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        RGB_Q = Q.view(m_batchsize, -1, width * height)
        RGB_mask = torch.bmm(RGB_K, RGB_Q)
        RGB_mask = self.softmax(RGB_mask)
        RGB_refine = torch.bmm(RGB_V, RGB_mask.permute(0, 2, 1))
        RGB_refine = RGB_refine.view(m_batchsize, -1, height, width)
        RGB_refine = self.gamma1 * RGB_refine + y

        INF_K = self.INF_K(y)
        INF_V = self.INF_V(y)
        INF_V = INF_V.view(m_batchsize, -1, width * height)
        INF_K = INF_K.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        INF_Q = Q.view(m_batchsize, -1, width * height)
        INF_mask = torch.bmm(INF_K, INF_Q)
        INF_mask = self.softmax(INF_mask)
        INF_refine = torch.bmm(INF_V, INF_mask.permute(0, 2, 1))
        INF_refine = INF_refine.view(m_batchsize, -1, height, width)
        INF_refine = self.gamma2 * INF_refine + x

        out = self.Second_reduce(torch.cat([RGB_refine, INF_refine], dim=1))
        return out


class PSFM(nn.Module):
    def __init__(self, in_C, out_C, cat_C):
        super(PSFM, self).__init__()
        self.RGBobj = DenseLayer(in_C, out_C)
        self.Infobj = DenseLayer(in_C, out_C)
        self.obj_fuse = GEFM(cat_C, out_C)

    def forward(self, rgb, depth):
        rgb_sum = self.RGBobj(rgb)
        Inf_sum = self.Infobj(depth)
        out = self.obj_fuse(rgb_sum, Inf_sum)
        return out


class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=4):
        """
        更像是DenseNet的Block，从而构造特征内的密集连接
        """
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BBasicConv2d(mid_C * i, mid_C, 3, 1, 1))

        self.fuse = BBasicConv2d(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        # print(down_feats.shape)
        # print(self.denseblock)
        out_feats = []
        for i in self.denseblock:
            # print(self.denseblock)
            feats = i(torch.cat((*out_feats, down_feats), dim=1))
            # print(feats.shape)
            out_feats.append(feats)

        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)


class BBasicConv2d(nn.Module):
    def __init__(
            self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BBasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)


#########################################################################################################    Inception


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU6(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class S2PM(nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(S2PM, self).__init__()
        self.block1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        out = self.block3(x2)
        return out


class DSRM(nn.Module):
    def __init__(self, in_channel=32, out_channel=32):
        super(DSRM, self).__init__()
        self.block1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            BasicConv2d(2 * in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            BasicConv2d(3 * in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.block4 = nn.Sequential(
            BasicConv2d(4 * in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(torch.cat([x, x1], dim=1))
        x3 = self.block3(torch.cat([x, x1, x2], dim=1))
        out = self.block4(torch.cat([x, x1, x2, x3], dim=1))
        return out


# ------------------------- Multi-label Segmentation Heads ----------------------------------- #
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1, bn=True, relu=True):
        padding = ((kernel_size - 1) * dilation + 1) // 2
        # padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                              bias=False if bn else True)
        self.bn = bn
        if bn:
            self.bnop = nn.BatchNorm2d(out_planes)
        self.relu = relu
        if relu:
            self.reluop = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bnop(x)
        if self.relu:
            x = self.reluop(x)
        return x


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


class RecDecoder(nn.Module):
    '''
    Scene Fidelity Path
    '''
    def __init__(self, channels, img=False):
        super(RecDecoder, self).__init__()
        self.decoder_dim_rec = 32
        self.SIM3 = SIM(norm_nc=32, label_nc=64, nhidden=32)
        self.seg_rec_decoder = DSRM(self.decoder_dim_rec, self.decoder_dim_rec)
        self.mask = S2M(channels[0])
        self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1),
        nn.Tanh(),
        )

    def forward(self, rec_f, seg_f):
        ## image reconstruction
        ## visible image
        features_seg_rec = self.SIM3(rec_f, seg_f)  # SIM   torch.Size([4, 32, 256, 256])
        rec_f = self.seg_rec_decoder(features_seg_rec)  # DSRM  torch.Size([4, 32, 256, 256])
        # ------------------------------------------------------------------------ #
        x = self.mask(rec_f)
        return (self.conv_block(x) + 1) / 2


class FusionDecoder(nn.Module):
    def __init__(self):
        super(FusionDecoder, self).__init__()
        self.decoder_dim_rec = 32
        self.decoder_dim_spilt = 16
        self.fuse_decoder = FeatureExtract(channel_in=self.decoder_dim_rec, channel_split_num=self.decoder_dim_spilt)
        channels = [self.decoder_dim_rec, 1]
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, rec_f):
        ## image reconstruction
        ## visible image
        feature = self.fuse_decoder(rec_f)            # dense-connected INN block for fusion----------wly
        return (self.conv_block(feature) + 1) / 2

# ----------------------------------------------------------------------------------------------------- #


class SegHead(nn.Module):
    '''This path plays the role of a classifier and is responsible for predicting the results of semantic segmentation, binary segmentation and edge segmentation'''

    def __init__(self, feature=64, n_classes=15):
        super(SegHead, self).__init__()
        self.binary_conv1 = ConvBNReLU(feature, feature // 4, kernel_size=1)
        self.binary_conv2 = nn.Conv2d(feature // 4, 2, kernel_size=3, padding=1)

        self.semantic_conv1 = ConvBNReLU(feature, feature, kernel_size=1)
        self.semantic_conv2 = nn.Conv2d(feature, n_classes, kernel_size=3, padding=1)

        self.boundary_conv1 = ConvBNReLU(feature * 2, feature, kernel_size=1)
        self.boundary_conv2 = nn.Conv2d(feature, 2, kernel_size=3, padding=1)

        self.boundary_conv = nn.Sequential(
            nn.Conv2d(feature * 2, feature, kernel_size=1),
            nn.BatchNorm2d(feature),
            nn.ReLU6(inplace=True),
            nn.Conv2d(feature, 2, kernel_size=3, padding=1),
        )

        self.up4x = nn.UpsamplingBilinear2d(scale_factor=4)  # 4
        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)  # 2

    def forward(self, feat):
        binary = self.binary_conv2(self.binary_conv1(feat))

        weight = torch.exp(binary)
        weight = weight[:, 1:2, :, :] / torch.sum(weight, dim=1, keepdim=True)

        feat_sematic = feat * weight
        feat_sematic = self.semantic_conv1(feat_sematic)

        semantic_out = self.semantic_conv2(feat_sematic)
        semantic_out = self.up2x(semantic_out)

        return semantic_out


class decoder(nn.Module):
    def __init__(self, channel=64):
        super(decoder, self).__init__()
        self.block1 = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        out = x3 + x
        out = self.up2(out)
        return out
# -------------------------------------------------------------------------------------- #

from pixel_decoder import build_pixel_decoder
from detectron2.config import get_cfg
from maskformer_config import add_mask_former_config
from invpt_AST import InvPT


INTERPOLATE_MODE = 'bilinear'
BATCHNORM = nn.SyncBatchNorm  # nn.BatchNorm2d


class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ConvBlock, self).__init__()
        if norm_layer is None:
            norm_layer = BATCHNORM
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class ShapeInfo:
    def __init__(self, channels, stride):
        self.stride = stride
        self.channels = channels


class SegmentationDecoder(nn.Module):
    def __init__(self):
        super(SegmentationDecoder, self).__init__()
        self.decoder_dim_rec = 32
        self.decoder_dim_seg = 64
        # 示例数据
        input_shape = {
            'res2': ShapeInfo(channels=64, stride=1),
            'res3': ShapeInfo(channels=64, stride=2),
            'res4': ShapeInfo(channels=64, stride=3),
            'res5': ShapeInfo(channels=64, stride=4)
        }
        cfg = get_cfg()
        add_mask_former_config(cfg)

        self.pixel_decoder = build_pixel_decoder(cfg, input_shape)

        self.embed_dim = 64

        self.PRED_OUT_NUM_CONSTANT = 64
        # embed_dim_with_pred = self.embed_dim + self.PRED_OUT_NUM_CONSTANT
        embed_dim_with_pred = self.embed_dim

        spec = {
                    'ori_embed_dim': self.embed_dim,
                    'NUM_STAGES': 3,
                    'PATCH_SIZE': [0, 3, 3],
                    'PATCH_STRIDE': [0, 1, 1],
                    'PATCH_PADDING': [0, 2, 2],
                    # 'DIM_EMBED': [embed_dim_with_pred, embed_dim_with_pred // 2, embed_dim_with_pred // 4],
                    'DIM_EMBED': [embed_dim_with_pred, embed_dim_with_pred, embed_dim_with_pred],
                    'NUM_HEADS': [2, 2, 2],
                    'MLP_RATIO': [4., 4., 4.],
                    'DROP_PATH_RATE': [0.15, 0.15, 0.15],
                    'QKV_BIAS': [True, True, True],
                    'KV_PROJ_METHOD': ['avg', 'avg', 'avg'],
                    'KERNEL_KV': [2, 4, 8],
                    'PADDING_KV': [0, 0, 0],
                    'STRIDE_KV': [2, 4, 8],
                    'Q_PROJ_METHOD': ['dw_bn', 'dw_bn', 'dw_bn'],
                    'KERNEL_Q': [3, 3, 3],
                    'PADDING_Q': [1, 1, 1],
                    'STRIDE_Q': [2, 2, 2],
                }
        self.invpt = InvPT(in_chans=embed_dim_with_pred, spec=spec)
        self.preliminary_decoder = nn.ModuleDict()

        input_channels = 64
        task_channels = 64

        num_outputs = 9

        self.intermediate_head = nn.Conv2d(task_channels, num_outputs, 1)
        self.preliminary_decoder = nn.Sequential(
            ConvBlock(input_channels, input_channels),
            ConvBlock(input_channels, task_channels),
        )
        self.seg_head = SegHead()
        self.ARM = ARM2()
        self.arm3 = ARM_I(64, 64, 64)
        self.arm2 = ARM_II(64, 64)
        self.arm1 = ARM_II(64, 64)
        self.upx2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upx4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upx8 = nn.UpsamplingBilinear2d(scale_factor=8)

        # self.classifier_Z1 = nn.Sequential(
        #     ConvBNReLU(64, 64, kernel_size=3, stride=1),
        #     self.upx2,
        #     nn.Conv2d(64, 9, kernel_size=1),
        #     self.upx2,
        #
        # )
        # self.classifier_Z2 = nn.Sequential(
        #     ConvBNReLU(64, 64, kernel_size=3, stride=1),
        #     self.upx4,
        #     nn.Conv2d(64, 9, kernel_size=1),
        #     self.upx2,
        #
        # )
        # self.classifier_Z3 = nn.Sequential(
        #     ConvBNReLU(64, 64, kernel_size=3, stride=1),
        #     self.upx4,
        #     nn.Conv2d(64, 9, kernel_size=1),
        #     self.upx4,
        #
        # )
        # ------------------------------------------------------- #
        # self.classifier_Z1 = nn.Sequential(
        #     ConvBNReLU(64, 64, kernel_size=3, stride=1),
        #     self.upx2,
        #     nn.Conv2d(64, 5, kernel_size=1),
        #     self.upx2,
        #
        # )
        # self.classifier_Z2 = nn.Sequential(
        #     ConvBNReLU(64, 64, kernel_size=3, stride=1),
        #     self.upx4,
        #     nn.Conv2d(64, 5, kernel_size=1),
        #     self.upx2,
        #
        # )
        # self.classifier_Z3 = nn.Sequential(
        #     ConvBNReLU(64, 64, kernel_size=3, stride=1),
        #     self.upx4,
        #     nn.Conv2d(64, 5, kernel_size=1),
        #     self.upx4,
        #
        # )
        # ------------------------------------------------ #
        self.classifier_Z1 = nn.Sequential(
            ConvBNReLU(64, 64, kernel_size=3, stride=1),
            self.upx2,
            nn.Conv2d(64, 15, kernel_size=1),
            self.upx2,

        )
        self.classifier_Z2 = nn.Sequential(
            ConvBNReLU(64, 64, kernel_size=3, stride=1),
            self.upx4,
            nn.Conv2d(64, 15, kernel_size=1),
            self.upx2,

        )
        self.classifier_Z3 = nn.Sequential(
            ConvBNReLU(64, 64, kernel_size=3, stride=1),
            self.upx4,
            nn.Conv2d(64, 15, kernel_size=1),
            self.upx4,

        )
        # ------------------------------------------------ #
        self.classifier_Z4 = nn.Sequential(
            ConvBNReLU(64, 64, kernel_size=3, stride=1),
            self.upx2,
            nn.Conv2d(64, 2, kernel_size=1),
            self.upx2,

        )
        self.classifier_Z5 = nn.Sequential(
            ConvBNReLU(64, 64, kernel_size=3, stride=1),
            self.upx4,
            nn.Conv2d(64, 2, kernel_size=1),
            self.upx2,

        )
        self.classifier_Z6 = nn.Sequential(
            ConvBNReLU(64, 64, kernel_size=3, stride=1),
            self.upx4,
            nn.Conv2d(64, 2, kernel_size=1),
            self.upx4,

        )
        self.aspp = ASPP(outchannel=64)

    def forward(self, rec_f1, x):

        _, _, multi_scale_features = self.pixel_decoder(x)

        # ---------------------------- #
        if x[3].shape == torch.Size([1, 64, 23, 40]):
            x[3] = F.interpolate(x[3], size=(45, 80), mode='bilinear', align_corners=False)
        if multi_scale_features[0].shape == torch.Size([1, 64, 23, 40]):
            multi_scale_features[0] = F.interpolate(multi_scale_features[0], size=(45, 80), mode='bilinear', align_corners=False)
        if x[1].shape == torch.Size([1, 64, 75, 100]):
            x[1] = F.interpolate(x[1], size=(76, 100), mode='bilinear', align_corners=False)
        # ---------------------------- #

        out1, out2 = self.arm3(x[2], x[3], multi_scale_features[0])

        out3, out4 = self.arm2(x[1], out1, out2)

        # ---------------------------- #
        if out3.shape == torch.Size([1, 64, 76, 100]):
            out3 = F.interpolate(out3, size=(75, 100), mode='bilinear', align_corners=False)
        if out4.shape == torch.Size([1, 64, 76, 100]):
            out4 = F.interpolate(out4, size=(75, 100), mode='bilinear', align_corners=False)
        # ---------------------------- #

        out5, out6 = self.arm1(x[0], out3, out4)


        out1 = self.classifier_Z6(out1)
        out2 = self.classifier_Z3(out2)

        out3 = self.classifier_Z5(out3)
        out4 = self.classifier_Z2(out4)

        out5 = self.classifier_Z4(out5)
        out6 = self.classifier_Z1(out6)

        # ---------------------------- #
        if multi_scale_features[2].shape == torch.Size([1, 64, 75, 100]):
            multi_scale_features[2] = F.interpolate(multi_scale_features[2], size=(76, 100), mode='bilinear', align_corners=False)
        if multi_scale_features[3].shape == torch.Size([1, 64, 150, 200]):
            multi_scale_features[3] = F.interpolate(multi_scale_features[3], size=(152, 200), mode='bilinear', align_corners=False)
        # ---------------------------- #

        for i in range(len(multi_scale_features)):
            multi_scale_features[i] = self.aspp(multi_scale_features[i])

        feature = self.invpt(x[2], multi_scale_features)

        # ---------------------------- #
        if feature.shape == torch.Size([1, 64, 304, 400]):
            feature = F.interpolate(feature, size=(300, 400), mode='bilinear', align_corners=False)
        # ---------------------------- #

        decoder_feature = self.preliminary_decoder(feature)
        feature = torch.cat([rec_f1, decoder_feature], dim=1)
        decoder_feature = self.ARM(feature)
        semantic_out = self.seg_head(decoder_feature)

        return semantic_out, out1, out2, out3, out4, out5, out6


class BoundaryHead(nn.Module):
    '''This path plays the role of a classifier and is responsible for predicting the results of semantic segmentation, binary segmentation and edge segmentation'''

    def __init__(self, feature=64, n_classes=9):
        super(BoundaryHead, self).__init__()
        self.binary_conv1 = ConvBNReLU(feature, feature // 4, kernel_size=1)
        self.binary_conv2 = nn.Conv2d(feature // 4, 2, kernel_size=3, padding=1)

        self.semantic_conv1 = ConvBNReLU(feature, feature, kernel_size=1)
        self.semantic_conv2 = nn.Conv2d(feature, n_classes, kernel_size=3, padding=1)

        self.boundary_conv1 = ConvBNReLU(feature * 2, feature, kernel_size=1)
        self.boundary_conv2 = nn.Conv2d(feature, 2, kernel_size=3, padding=1)

        self.boundary_conv = nn.Sequential(
            nn.Conv2d(feature * 2, feature, kernel_size=1),
            nn.BatchNorm2d(feature),
            nn.ReLU6(inplace=True),
            nn.Conv2d(feature, 2, kernel_size=3, padding=1),
        )

        self.up4x = nn.UpsamplingBilinear2d(scale_factor=4)  # 4
        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)  # 2

    def forward(self, feat):
        binary = self.binary_conv2(self.binary_conv1(feat))

        weight = torch.exp(binary)
        weight = weight[:, 1:2, :, :] / torch.sum(weight, dim=1, keepdim=True)

        feat_sematic = self.up4x(feat * weight)
        feat_sematic = self.semantic_conv1(feat_sematic)

        feat_boundary = self.up2x(torch.cat([feat_sematic, self.up4x(feat)], dim=1))
        boundary_out = self.boundary_conv(feat_boundary)
        boundary_out = self.up4x(boundary_out)

        return boundary_out

class BinaryHead(nn.Module):
    '''This path plays the role of a classifier and is responsible for predicting the results of semantic segmentation, binary segmentation and edge segmentation'''

    def __init__(self, feature=32, n_classes=9):
        super(BinaryHead, self).__init__()
        self.binary_conv1 = ConvBNReLU(feature, feature // 4, kernel_size=1)
        self.binary_conv2 = nn.Conv2d(feature // 4, 2, kernel_size=3, padding=1)

        self.up2x = nn.UpsamplingBilinear2d(scale_factor=2)  # 2

    def forward(self, feat):
        binary = self.binary_conv2(self.binary_conv1(feat))  # 8, 2, 128, 128
        binary_out = self.up2x(binary)  # 8, 2, 256, 256

        return binary_out


if __name__ == "__main__":
    # 设置参数
    # 初始化网络
    # 生成随机输入
    vi = torch.randn(4, 3, 800, 600)
    ir = torch.randn(4, 1, 800, 600)
    encoder = Encoder()
    fuse_decoder = FusionDecoder()
    seg_decoder = SegmentationDecoder()
    rec_f, rec_f1, _, seg_f = encoder(vi, ir)
    semantic_out, out1, out2, out3, out4, out5, out6 = seg_decoder(rec_f1, seg_f)
    fused_img = fuse_decoder(rec_f)
    print(semantic_out.shape, fused_img.shape)



