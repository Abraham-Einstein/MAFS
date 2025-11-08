import torch
import torch.nn as nn

torch.cuda.set_device(1)   # your own choice
# ------------------------- EAEF --------------------------- #
class Atttion_avg_pool(nn.Module):
    def __init__(self, dim, reduction):
        super(Atttion_avg_pool, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Feature_Pool(nn.Module):
    def __init__(self, dim, ratio=2):
        super(Feature_Pool, self).__init__()
        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.down = nn.Linear(dim, dim * ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim * ratio, dim)

    def forward(self, x):
        b, c, _, _ = x.size()
        # print(x.shape)  # 输入的形状
        # print(self.gap_pool(x).shape)
        # y = self.up(self.act(self.down(self.gap_pool(x).permute(0,2,3,1)))).permute(0,3,1,2).view(b,c)
        y = self.gap_pool(x).view(b, c)  # 形状变为 (b, c)

        y = self.up(self.act(self.down(y)))

        return y.view(b, c, 1, 1)

class Channel_Attention(nn.Module):
    def __init__(self, dim, ratio=16):
        super(Channel_Attention, self).__init__()
        self.gmp_pool = nn.AdaptiveMaxPool2d(1)
        self.down = nn.Linear(dim, dim//ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim//ratio, dim)

    def forward(self, x):
        max_out = self.up(self.act(self.down(self.gmp_pool(x).permute(0,2,3,1)))).permute(0,3,1,2)
        return max_out


class Spatial_Attention(nn.Module):
    def __init__(self, dim):
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1,bias=True)

    def forward(self, x):
        x1 = self.conv1(x)
        return x1


class EAEF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp_pool = Feature_Pool(dim)
        self.dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=7,padding=3,groups=dim)
        self.T_c = nn.Parameter(torch.ones([]) * dim)
        self.cse = Channel_Attention(dim*2)
        self.sse_r = Spatial_Attention(dim)
        self.sse_t = Spatial_Attention(dim)

    def forward(self, x):
        ############################################################################
        b, c, h, w = x.size()
        RGB = x[:,0:c//2,:]
        T = x[:,c//2:c,:]
        b, c, h, w = RGB.size()
        rgb_y = self.mlp_pool(RGB)
        t_y = self.mlp_pool(T)
        rgb_y = rgb_y / rgb_y.norm(dim=1, keepdim=True)
        t_y = t_y / t_y.norm(dim=1, keepdim=True)
        rgb_y = rgb_y.view(b, c, 1)
        t_y = t_y.view(b, 1, c)
        logits_per = self.T_c * rgb_y @ t_y
        cross_gate = torch.diagonal(torch.sigmoid(logits_per)).reshape(b,c,1,1)
        add_gate = torch.ones(cross_gate.shape).cuda() - cross_gate
        # add_gate = torch.ones(cross_gate.shape) - cross_gate
        ##########################################################################
        New_RGB_A = RGB * cross_gate
        New_T_A = T * cross_gate
        x_cat = torch.cat((New_RGB_A,New_T_A),dim=1)
        ##########################################################################
        fuse_gate = torch.sigmoid(self.cse(self.dwconv(x_cat)))
        rgb_gate, t_gate = fuse_gate[:, 0:c, :], fuse_gate[:, c:c * 2, :]
        ##########################################################################
        New_RGB =  RGB * add_gate + New_RGB_A * rgb_gate
        New_T =  T * add_gate + New_T_A * t_gate
        ##########################################################################
        New_fuse_RGB = self.sse_r(New_RGB)
        New_fuse_T = self.sse_t(New_T)
        attention_vector = torch.cat([New_fuse_RGB, New_fuse_T], dim=1)
        attention_vector = torch.softmax(attention_vector,dim=1)
        attention_vector_l, attention_vector_r = attention_vector[:, 0:1, :, :], attention_vector[:, 1:2, :, :]
        New_RGB = New_RGB * attention_vector_l + New_T * attention_vector_r
        New_T = New_T * attention_vector_r
        out = torch.cat([New_RGB,New_T],dim=1)
        ##########################################################################
        return out

# ----------------------------------------------------------------------- #


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


#Global Contextual module
class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = BasicConv2d(in_channel, out_channel, 1)
        self.branch1_1 = BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1))
        self.branch1_2 = BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0))

        self.branch2 = BasicConv2d(in_channel, out_channel, 1)
        self.branch2_1 = BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2))
        self.branch2_2 = BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0))

        self.branch3 = BasicConv2d(in_channel, out_channel, 1)
        self.branch3_1 = BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3))
        self.branch3_2 = BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0))

        self.se = Atttion_avg_pool(out_channel,4)
        self.conv_res = nn.Conv2d(in_channel,out_channel,kernel_size=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x0 = self.se(x0)
        x1 = self.branch1_2(self.branch1_1(self.branch1(x)))
        x1 = self.se(x1)
        x2 = self.branch2_2(self.branch2_1(self.branch2(x)))
        x2 = self.se(x2)
        x3 = self.branch3_2(self.branch3_1(self.branch3(x)))
        x3 = self.se(x3)
        x_add = x0 + x1 + x2 + x3
        x = self.relu(x_add + self.conv_res(x))
        return x

# --------------------- miniASPP ------------------------------- #


class mini_Aspp(nn.Module):
    def __init__(self,channel):
        super(mini_Aspp,self).__init__()
        self.conv_6 = nn.Conv2d(channel, channel, kernel_size=3,  stride=1, padding=6,  dilation=6)
        self.conv_12 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=12, dilation=12)
        self.conv_18 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        x1 = self.bn(self.conv_6(x))
        x2 = self.bn(self.conv_12(x))
        x3 = self.bn(self.conv_18(x))
        feature_map = x1 + x2 + x3
        return feature_map