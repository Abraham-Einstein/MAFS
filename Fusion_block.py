import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from util.dual_self_att import CAM_Module
from timm.models.layers import to_2tuple, trunc_normal_
import math


# =============================================================================
import numbers
from einops import rearrange
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


########### local enhanced feed-forward network #############
class LeFF(nn.Module):
    def __init__(self, dim=64, hidden_dim=64, act_layer=nn.GELU):
        super(LeFF, self).__init__()

        self.linear1 = nn.Sequential(nn.Conv2d(dim, hidden_dim, kernel_size=1, stride=1, padding=0),
                                     act_layer()
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer()
        )
        self.linear2 = nn.Sequential(nn.Conv2d(hidden_dim, dim, kernel_size=1, stride=1, padding=0),
                                     act_layer()
        )
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        origin_x = x
        x = self.linear1(x)
        x = self.dwconv(x)
        x = self.linear2(x) + origin_x
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = LeFF(dim, hidden_dim=dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

# ---------------------------------------------------------- #


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
            # self.reluop = nn.ReLU6(inplace=True)
            self.reluop = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bnop(x)
        if self.relu:
            x = self.reluop(x)
        return x


class PE_TdSAtt(nn.Module):
    """ Image to Patch Embedding  +  Two-dimensional Splicing Attention
    """

    def __init__(self, img_size=104, patch_size=7, stride=4, in_chans=3, embed_dim1=60, embed_dim2=80):
        super(PE_TdSAtt, self).__init__()

        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, 64, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm1_r = nn.LayerNorm(embed_dim2)
        self.norm1_c = nn.LayerNorm(embed_dim1)

        self.norm2_r = nn.LayerNorm(embed_dim2)
        self.norm2_c = nn.LayerNorm(embed_dim1)
        # self.softmax = nn.Softmax(dim=2)
        self.apply(self._init_weights)
        # self.linear = nn.Linear(d_model, S, bias=False)
        self.linear1_r = nn.Linear(embed_dim2, embed_dim2, bias=False)
        self.linear1_c = nn.Linear(embed_dim1, embed_dim1, bias=False)

        self.linear2_r = nn.Linear(embed_dim2, embed_dim2, bias=False)
        self.linear2_c = nn.Linear(embed_dim1, embed_dim1, bias=False)

        self.conv_cat = ConvBNReLU(in_planes=3*64, out_planes=64, kernel_size=3, stride=1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        x = self.proj(x)
        B, C, H, W = x.shape
        x1_r = x.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)  #b*w*c*h - b*w*ch-  b*ch*w
        x1_r = self.norm1_r(x1_r)
        x1_r = self.linear1_r(x1_r)    #b*w*ch

        x1_c = x1_r.view(B, W, C, -1).permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)      #B*W*C*H - B*H*W*C -  B*H*WC - B*WC*H
        x1_c = self.norm1_c(x1_c)
        x1_c = self.linear1_c(x1_c)
        x1 = x1_c.view(B, C, -1, H).permute(0, 1, 3, 2)  #B*C*w*H -- B*C*H*w

        x2_c = x.permute(0, 2, 1, 3).flatten(2).permute(0, 2, 1)  # b*h*c*w - b*h*cw  -b*cw*h
        x2_c = self.norm2_c(x2_c)
        x2_c = self.linear2_c(x2_c)  # b*cw*h

        x2_r = x2_c.view(B, C, -1, H).permute(0, 2, 1, 3).flatten(2).permute(0, 2, 1)  # B*C*W*H - B*W*C*H -  B*W*CH- B*CH*W
        x2_r = self.norm2_r(x2_r)
        x2_r = self.linear2_r(x2_r)
        x2 = x2_r.view(B, C, -1, W)  # B*C*H*W
        x12 = x1 + x2
        out = self.conv_cat(torch.cat((x12, x1, x2), dim=1))

        return out


class PE_Att(nn.Module):
    def __init__(self, img_size=104, patch_size=3, stride=1, in_chans=64, d_model=None, embed_dim=64,
                 embed_dim1=60, embed_dim2=80, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0, sr_ratio=1, drop_rate=0):
        super(PE_Att, self).__init__()
        self.laynorm_r = nn.LayerNorm(embed_dim2)
        self.laynorm_c = nn.LayerNorm(embed_dim1)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.patchembed_add = PE_TdSAtt(img_size=img_size, patch_size=patch_size, stride=stride, in_chans=in_chans,
                                              embed_dim1=embed_dim1, embed_dim2=embed_dim2)
        self.patchembed_mul = PE_TdSAtt(img_size=img_size, patch_size=patch_size, stride=stride,
                                              in_chans=in_chans,embed_dim1=embed_dim2, embed_dim2=embed_dim1)

    def forward(self, x):

        out = self.patchembed_add(x)

        return out

######MMS######
class Fusion_MMS(nn.Module):
    def __init__(self, img_size, patch_size, stride, in_chans, h, w, d_model, embed_dim, embed_dim1, embed_dim2):
        super(Fusion_MMS, self).__init__()
        self.pe_mha_rgb = PE_Att(img_size, patch_size, stride, in_chans, d_model=d_model, embed_dim=embed_dim,
                 embed_dim1=embed_dim1, embed_dim2=embed_dim2, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0, sr_ratio=1, drop_rate=0)
        self.pe_mha_t = PE_Att(img_size, patch_size, stride, in_chans, d_model=d_model, embed_dim=embed_dim,
                                 embed_dim1=embed_dim1, embed_dim2=embed_dim2, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0, sr_ratio=1,
                                 drop_rate=0)
        self.GloablAvgPool_rgb = nn.AdaptiveAvgPool2d(output_size=(h//2, w//2))
        self.GloablAvgPool_t = nn.AdaptiveAvgPool2d(output_size=(h//2, w//2))

        self.GloablAvgPool_rgb_4 = nn.AdaptiveAvgPool2d(output_size=(h , w ))
        self.GloablAvgPool_t_4 = nn.AdaptiveAvgPool2d(output_size=(h , w))

        self.conv1_rgb1 = nn.Conv2d(in_channels=64, out_channels=64//4, kernel_size=1, stride=1, padding=0)
        self.conv1_t1 = nn.Conv2d(in_channels=64, out_channels=64//4, kernel_size=1, stride=1, padding=0)
        self.conv1_rgb2 = nn.Conv2d(in_channels=64//4, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv1_t2 = nn.Conv2d(in_channels=64//4, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.cbl_1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU()
                                      )
        self.cbl_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU()
                                   )
        self.cbl_3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU()
                                   )
        self.cbl_3_4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU()
                                   )
        self.cbl_4 = nn.Sequential(nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU()
                                   )
        self.cbl_4_down = nn.Sequential(nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU()
                                   )
        self.cbl_4_1 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU()
                                   )
        self.cbl_4_2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
                                     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.LeakyReLU()
                                     )
        self.conv3_add = nn.Conv2d(in_channels=in_chans, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3_mul = nn.Conv2d(in_channels=in_chans, out_channels=64, kernel_size=3, stride=2, padding=1)

        self.conv1_add = nn.Conv2d(in_channels=in_chans, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_mul = nn.Conv2d(in_channels=in_chans, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb, t, f, NO):
        rgbt_add = rgb + t  #c*h*w
        rgbt_mul = torch.mul(rgb, t)  #c*h*w

        rgbt_add_former = self.pe_mha_rgb(rgbt_add)
        rgbt_mul_former = self.pe_mha_t(rgbt_mul)

        rgbt_add_2 = self.conv1_add(rgbt_add)
        rgbt_mul_2 = self.conv1_mul(rgbt_mul)

        rgbt_add_former = rgbt_add_2 + rgbt_add_former
        rgbt_mul_former = rgbt_mul_2 + rgbt_mul_former

        rgbt_add_avgpool = self.GloablAvgPool_rgb_4(rgbt_add_2)
        rgbt_add_conv = self.conv1_rgb1(rgbt_add_avgpool)
        rgbt_add_conv = self.conv1_rgb2(rgbt_add_conv)
        rgbt_add_conv = torch.mul(rgbt_add_former, rgbt_add_conv)

        rgbt_mul_avgpool = self.GloablAvgPool_t_4(rgbt_mul_2)
        rgbt_mul_conv = self.conv1_t1(rgbt_mul_avgpool)
        rgbt_mul_conv = self.conv1_t2(rgbt_mul_conv)
        rgbt_mul_conv = torch.mul(rgbt_mul_former, rgbt_mul_conv)

        rgbt_add_conv = self.cbl_1(rgbt_add_conv)
        rgbt_mul_conv = self.cbl_2(rgbt_mul_conv)
        if NO == 1:
            rgbt = torch.cat((rgbt_add_conv, rgbt_mul_conv), dim=1)
            rgbt1 = self.cbl_4_1(rgbt)
            rgbt_down = self.cbl_4_2(rgbt)
            # print('rgbt1, rgbt_down', rgbt1.shape, rgbt_down.shape)
        elif NO == 4:
            f = self.cbl_3_4(f)
            rgbt = torch.cat((rgbt_add_conv, rgbt_mul_conv, f), dim=1)
            rgbt1 = self.cbl_4(rgbt)
            rgbt_down = None
            # print('rgbt1, rgbt_down', rgbt1.shape)
        else:
            f = self.cbl_3_4(f)
            rgbt = torch.cat((rgbt_add_conv, rgbt_mul_conv, f), dim=1)
            rgbt1 = self.cbl_4(rgbt)
            rgbt_down = self.cbl_4_down(rgbt)
            # print('rgbt1, rgbt_down', rgbt1.shape, rgbt_down.shape)

        return rgbt1, rgbt_down


# ----------------- Detail Fusion in CAINet ---------------------- #


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        att = self.sigmoid(out)
        out = torch.mul(x, att)
        return out


class Fusion2(nn.Module):
    def __init__(self, in_dim, kernel_size=7):
        super(Fusion2, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tp = nn.Conv2d(in_dim, 64, kernel_size=1)
        self.ca = ChannelAttention(64)

    def forward(self, x1, x2):
        max_out, _ = torch.max(x2, dim=1, keepdim=True)
        x2 = max_out
        x2 = self.conv1(x2)
        att2 = self.sigmoid(x2+x1)
        out = torch.mul(x1, att2) + x2
        tp = self.tp(out)
        fuseout = self.ca(tp)

        return fuseout

# -------------------- CRRM interaction module in CAINet --------------------------------------- #

class CRRM(nn.Module):
    def __init__(self, in_channels):
        super(CRRM, self).__init__()
        self.C = 128
        self.omg = nn.Conv2d(in_channels, self.C, 1, 1, 0, bias=False)  #C*H*W
        self.theta = nn.Conv2d(in_channels, self.C, 1, 1, 0, bias=False)  #C*H*W
        self.ori = nn.Conv2d(in_channels, self.C, 1, 1, 0, bias=False)   #C*H*W

        self.relu = nn.ReLU()

        self.node_conv = nn.Conv1d(self.C, self.C, 1, 1, 0, bias=False)
        self.channel_conv = nn.Conv1d(self.C, self.C, 1, 1, 0, bias=False)


    def forward(self, x):
        batch, C, H, W = x.size()
        L = H * W
        o = self.omg(x)
        omg = o.view(-1, self.C, L) #C*L
        theta = self.theta(x).view(-1, self.C, L) #C*L
        ori = self.ori(x).view(-1, self.C, L)  #C*L

        B = torch.transpose(ori, 1, 2)       #L*C

        V = torch.bmm(theta, B)  #
        A = torch.nn.functional.softmax(V, dim=-1) #CC
        AV = self.node_conv(A)  #C*C
        IV = A+AV
        IAV = self.relu(self.channel_conv(torch.transpose(IV, 1, 2)))  #C*C

        y = torch.bmm(IAV, omg)
        y = y.view(-1, self.C, H, W)

        return o + y

# ----------------------- ARM in CAINet (three feature maps input) ------------------ #


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1):
        padding = ((kernel_size - 1) * dilation + 1) // 2
        # padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(collections.OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                               bias=False)),
            ('bn', nn.BatchNorm2d(out_planes)),
            ('relu', nn.ReLU6(inplace=True))]))

class ARM(nn.Module):
    def __init__(self):
        super(ARM, self).__init__()

        self.dilation_conv1 = ConvBNReLU(256 + 64, 64, dilation=1)
        self.dilation_conv2 = ConvBNReLU(256 + 64, 64, dilation=2)
        self.dilation_conv3 = ConvBNReLU(256 + 64, 64, dilation=4)
        self.dilation_conv4 = ConvBNReLU(256 + 64, 64, dilation=8)  # ---------original---wly
        # self.dilation_conv1 = ConvBNReLU(256, 64, dilation=1)
        # self.dilation_conv2 = ConvBNReLU(256, 64, dilation=2)
        # self.dilation_conv3 = ConvBNReLU(256, 64, dilation=4)
        # self.dilation_conv4 = ConvBNReLU(256, 64, dilation=8)
        self.conv1 = ConvBNReLU(256, 64, kernel_size=1)

    def forward(self, x):
        x1 = self.dilation_conv1(x)
        x2 = self.dilation_conv2(x)
        x3 = self.dilation_conv3(x)
        x4 = self.dilation_conv4(x)
        seg_feature = self.conv1(torch.cat((x1, x2, x3, x4), dim=1))

        return seg_feature

class ARM2(nn.Module):
    def __init__(self):
        super(ARM2, self).__init__()

        self.dilation_conv1 = ConvBNReLU(32 + 64, 64, dilation=1)
        self.dilation_conv2 = ConvBNReLU(32 + 64, 64, dilation=2)
        self.dilation_conv3 = ConvBNReLU(32 + 64, 64, dilation=4)
        self.dilation_conv4 = ConvBNReLU(32 + 64, 64, dilation=8)
        self.conv1 = ConvBNReLU(256, 64, kernel_size=1)

    def forward(self, x):
        x1 = self.dilation_conv1(x)
        x2 = self.dilation_conv2(x)
        x3 = self.dilation_conv3(x)
        x4 = self.dilation_conv4(x)
        seg_feature = self.conv1(torch.cat((x1, x2, x3, x4), dim=1))

        return seg_feature


# ------------------ glore in CAINet (two feature maps concatenation then input)->channel/2_size_unchange --------------------------- #


class Interv(nn.Module):
    def __init__(self, in_channels):
        super(Interv, self).__init__()
        self.N = in_channels // 8
        self.S = in_channels // 4

        self.theta = nn.Conv2d(in_channels, self.N, 1, 1, 0, bias=False)
        self.phi = nn.Conv2d(in_channels, self.S, 1, 1, 0, bias=False)

        self.relu = nn.ReLU()

        self.node_conv = nn.Conv1d(self.N, self.N, 1, 1, 0, bias=False)
        self.channel_conv = nn.Conv1d(self.S, self.S, 1, 1, 0, bias=False)

        self.conv_2 = nn.Conv2d(self.S, in_channels, 1, 1, 0, bias=False)
        self.conv_3 = nn.Conv2d(in_channels, in_channels // 2, 1, 1, 0, bias=False)

    def forward(self, x):
        batch, C, H, W = x.size()
        L = H * W

        B = self.theta(x).view(-1, self.N, L)  #N*L

        phi = self.phi(x).view(-1, self.S, L)  #S*L
        phi = torch.transpose(phi, 1, 2)       #L*S

        V = torch.bmm(B, phi) / L  #  #N*S

        AV = self.relu(self.node_conv(V))  #N*S
        IV = V+AV
        IAVW = self.relu(self.channel_conv(torch.transpose(IV, 1, 2)))  #S*N

        y = torch.bmm(torch.transpose(B, 1, 2), torch.transpose(IAVW, 1, 2))
        y = y.view(-1, self.S, H, W)
        y = self.conv_2(y)
        feature = self.conv_3(x + y)
        return feature


# ------------------ CAM in LASNet ------------------------------- #


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class CAM(nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(CAM, self).__init__()
        #self.conv1 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.sa = SpatialAttention()
        # self-channel attention
        self.cam = CAM_Module(out_channel)

    def forward(self, x, ir):
        multiplication = x * ir
        summation = self.conv2(x + ir)

        sa = self.sa(multiplication)
        summation_sa = summation.mul(sa)

        sc_feat = self.cam(summation_sa)

        return sc_feat

# ------------------ CLM in LASNet ------------------------- #


class CorrelationModule(nn.Module):
    def __init__(self, in_channel=512, all_channel=64):
        super(CorrelationModule, self).__init__()
        self.linear_e = nn.Linear(in_channel, in_channel, bias=False)
        self.channel = in_channel
        self.fusion = BasicConv2d(in_channel, all_channel, kernel_size=3, padding=1)

    def forward(self, exemplar, query):  # exemplar: middle, query: rgb or T
        fea_size = exemplar.size()[2:]
        all_dim = fea_size[0]*fea_size[1]
        exemplar_flat = exemplar.view(-1, self.channel, all_dim) #N,C,H*W
        query_flat = query.view(-1, self.channel, all_dim)
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  #batchsize x dim x num, N,H*W,C
        exemplar_corr = self.linear_e(exemplar_t)  #
        A = torch.bmm(exemplar_corr, query_flat)
        B = F.softmax(torch.transpose(A, 1, 2), dim=1)
        exemplar_att = torch.bmm(query_flat, B).contiguous()

        exemplar_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])
        exemplar_out = self.fusion(exemplar_att)

        return exemplar_out


class CLM(nn.Module):
    def __init__(self, in_channel=512, all_channel=64):
        super(CLM, self).__init__()
        self.corr_x_2_x_ir = CorrelationModule(in_channel, all_channel)
        self.corr_ir_2_x_ir = CorrelationModule(in_channel, all_channel)
        self.smooth1 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.smooth2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.fusion = BasicConv2d(2*all_channel, all_channel, kernel_size=3, padding=1)
        self.pred = nn.Conv2d(all_channel, 2, kernel_size=3, padding=1, bias=True)

    def forward(self, x, x_ir, ir):  # exemplar: middle, query: rgb or T
        corr_x_2_x_ir = self.corr_x_2_x_ir(x_ir, x)
        corr_ir_2_x_ir = self.corr_ir_2_x_ir(x_ir, ir)

        summation = self.smooth1(corr_x_2_x_ir + corr_ir_2_x_ir)
        multiplication = self.smooth2(corr_x_2_x_ir * corr_ir_2_x_ir)

        fusion = self.fusion(torch.cat([summation, multiplication], 1))
        # sal_pred = self.pred(fusion)

        # return fusion, sal_pred
        return fusion


# ----------------------- ESM in LASNet --------------------------------- #

class ESM(nn.Module):
    def __init__(self, all_channel=64):
        super(ESM, self).__init__()
        self.conv1 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(all_channel, all_channel, kernel_size=3, padding=1)
        self.dconv1 = BasicConv2d(all_channel,int( all_channel/4), kernel_size=3, padding=1)
        self.dconv2 = BasicConv2d(all_channel,int( all_channel/4), kernel_size=3, dilation=3, padding=3)
        self.dconv3 = BasicConv2d(all_channel,int( all_channel/4), kernel_size=3, dilation=5, padding=5)
        self.dconv4 = BasicConv2d(all_channel,int( all_channel/4), kernel_size=3, dilation=7, padding=7)
        self.fuse_dconv = nn.Conv2d(all_channel, all_channel, kernel_size=3,padding=1)
        # self.pred = nn.Conv2d(all_channel, 2, kernel_size=3, padding=1, bias = True)

    def forward(self, x, ir):
        multiplication = self.conv1(x * ir)
        summation = self.conv2(x + ir)
        fusion = (summation + multiplication)
        x1 = self.dconv1(fusion)
        x2 = self.dconv2(fusion)
        x3 = self.dconv3(fusion)
        x4 = self.dconv4(fusion)
        out = self.fuse_dconv(torch.cat((x1, x2, x3, x4), dim=1))
        # edge_pred = self.pred(out)

        # return out, edge_pred
        return out

# ----------- FRM in CMX -------------- #

# Feature Rectify Module


class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1)  # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights


class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FeatureRectifyModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        # print(out_x1.shape, out_x2.shape)
        return out_x1, out_x2

# ----------- FFM in CMX -------------- #


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return x1, x2


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2


class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction,
                                        norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)

        return merge


class FusionModule(nn.Module):
    def __init__(self, dim):
        super(FusionModule, self).__init__()
        self.FRM = FeatureRectifyModule(dim)
        self.FFM = FeatureFusionModule(dim=dim, num_heads=8)

    def forward(self, x1, x2):
        feature1, feature2 = self.FRM(x1, x2)
        # x = self.FFM(feature1, feature2)
        # return x
        return feature1, feature2
# ---------- MFM(multi-modal fusion module) in EGFNet ----------- #


class olm(nn.Module):
    def __init__(self, outchannel, achannel):
        super(olm, self).__init__()
        self.conv1 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=3, padding=3)
        self.conv4 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=4, padding=4)

        self.conv = nn.Conv2d(5*outchannel, outchannel, 3, padding=1)
        self.convs = nn.Sequential(
            nn.Conv2d(outchannel, achannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(achannel),
            nn.ReLU()
        )

        self.convf = nn.Conv2d(2*outchannel, outchannel, kernel_size=1)

        self.rconv = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU()
        )

        self.rrconv = nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1)
        self.rrbn = nn.BatchNorm2d(outchannel)
        self.rrrelu = nn.ReLU()

        self.conv0 = nn.Conv2d(2*outchannel, outchannel, kernel_size=1)


    def  forward(self, x, ir):
        xx1 = x + ir
        xx1x = x * xx1
        xx1ir = ir * xx1
        xx = torch.cat((xx1x, xx1ir), dim=1)
        xx = self.conv0(xx)

        n = self.rrbn(self.rrconv(self.rconv(xx)))
        xx = self.rrrelu(xx + n)

        x1 = self.conv1(xx)
        x2 = self.conv2(xx)
        x3 = self.conv3(xx)
        x4 = self.conv4(xx)

        xp = torch.cat((xx, x1, x2, x3, x4), dim=1)
        xp = self.conv(xp)

        x_s = self.convs(xp)  #

        return x_s, xp

# ------------- SGM(semantic guidance module) in EGFNet ------------------ #


class seman(nn.Module):
    def __init__(self, inchannel):
        super(seman, self).__init__()
        self.rconv = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU()
        )
        self.conv = nn.Conv2d(inchannel, inchannel, kernel_size=1)
        self.convcat = nn.Conv2d(2*inchannel, inchannel, kernel_size=1)

        self.convedge = nn.Conv2d(1, 9, kernel_size=1)

    def forward(self, s1, s2, edge):
        s1 = torch.nn.functional.interpolate(s1, scale_factor=32, mode='bilinear')
        s2 = torch.nn.functional.interpolate(s2, scale_factor=16, mode='bilinear')
        s = torch.cat((s1, s2), dim=1)
        s = self.convcat(s)

        s = s + s1 + s2
        s = self.rconv(s)
        s = s * s1
        s = self.conv(s)
        edge = self.convedge(edge)

        se = s * edge
        s = se + s

        return s

# ------------- GIM(global information) in EGFNet ------------------ #


class ASPP(nn.Module):
    def __init__(self, outchannel):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=3, padding=3)
        self.conv4 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=4, padding=4)
        self.conv0 = nn.Conv2d(outchannel, outchannel, kernel_size=1)

        self.conv = nn.Conv2d(5*outchannel, outchannel, kernel_size=1)

        self.rconv = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU()
        )

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        out = torch.cat((x0, x1, x2, x3, x4), dim=1)
        out = self.conv(out)

        out = out + x
        out = self.rconv(out)


        return out

# --------------- SIM(semantic information) in EGFNet ----------------- #


class EM(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(EM, self).__init__()
        self.conv = nn.Conv2d(2*inchannel, inchannel, kernel_size=1)

        self.rconv = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU()
        )
        self.rconv0 = nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1)
        self.rbn = nn.BatchNorm2d(inchannel)

        self.convfinal = nn.Conv2d(inchannel, outchannel, kernel_size=1)

    def forward(self, laster, current):

        out1 = torch.cat((laster, current), dim=1)
        out1 = self.conv(out1)

        x1 = laster * out1
        ir1 = current * out1
        f = x1 + ir1

        f = self.rbn(self.rconv0(self.rconv(f)))
        f = f + laster

        f = self.convfinal(f)

        return f

# -------------------sum3 in EGFNet --------------------------- #


class EM2(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(EM2, self).__init__()


    def forward(self, laster, current, high):


        f = laster + current + high

        return f

