# By Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange as o_rearrange
from einops.layers.torch import Rearrange
from inv_utils.utils import to_2tuple
from timm.models.layers import DropPath, trunc_normal_
from util.AST import Attention, FeedForward, LayerNorm, LeFF
import pdb

BATCHNORM = nn.SyncBatchNorm  # nn.BatchNorm2d

def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()

class UpEmbed(nn.Module):

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, padding=padding, stride=stride, bias=False, dilation=padding),
                    BATCHNORM(embed_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, padding=padding, stride=stride, bias=False, dilation=padding),
                    BATCHNORM(embed_dim),
                    nn.ReLU(inplace=True)
                    )

    def forward(self, x):
        x = self.proj(x)
        return x


class InvPTBlock(nn.Module):

    def __init__(self,
                 task_no,
                 dim_in,
                 num_heads,
                 drop_path=0.,
                 win_size=8,
                 ffn_expansion_factor=2,
                 LayerNorm_type='WithBias',
                 bias=False,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.stride_q = kwargs['stride_q']
        self.embed_dim = dim_in // task_no
        self.task_no = task_no
        self.win_size = win_size
        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()

        self.norm1 = norm_layer(self.embed_dim)
        self.norm2 = norm_layer(self.embed_dim)
        # self.mlp = Mlp(
        #     in_features=self.embed_dim,
        #     hidden_features=dim_mlp_hidden,
        #     act_layer=act_layer,
        #     drop=drop
        # )

        # self.attn = SelfAttention(task_no,
        #     self.embed_dim, self.embed_dim, num_heads, qkv_bias, attn_drop, drop,
        #     **kwargs
        # )
        self.attn = Attention(dim_in, num_heads, bias)
        self.norm1 = LayerNorm(dim_in, LayerNorm_type)
        self.norm2 = LayerNorm(dim_in, LayerNorm_type)
        # self.ffn = FeedForward(dim_in, ffn_expansion_factor, bias)
        self.ffn = LeFF()


    def forward(self, x_list, messages):

        x = x_list + self.attn(self.norm1(x_list))
        x_list = x + self.ffn(self.norm2(x))
        print(x_list.shape)

        return x_list

class InvPTStage(nn.Module):
    def __init__(self,
                 stage_idx,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=1,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 **kwargs):
        super().__init__()
        assert depth == 1
        self.stage_idx = stage_idx
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.task_no = 1

        self.rearrage = None

        if patch_size == 0:
            self.patch_embed = None
        else:
            self.patch_embed = [UpEmbed(
                patch_size=patch_size,
                in_chans=in_chans,
                stride=patch_stride,
                padding=patch_padding,
                embed_dim=embed_dim,
            ) for _ in range(self.task_no)]

            self.patch_embed = nn.ModuleList(self.patch_embed)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule, but we only use depth=1 here.

        blocks = []
        for j in range(depth):
            blocks.append(
                InvPTBlock(
                    dim_in=embed_dim*self.task_no,
                    dim_out=embed_dim*self.task_no,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, BATCHNORM)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, BATCHNORM)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_list, messages, back_fea):
        if self.patch_embed != None:
            for i in range(self.task_no):
                # x = self.patch_embed[i](x_list[i])
                x = self.patch_embed[i](x_list)
                # print('x', x.shape)
                # print('xlist', x_list.shape)
                # backbone skip connection
                # if self.stage_idx == 0:
                #     x = x + back_fea[0]
                # if self.stage_idx == 1:
                #     x = x + back_fea[1]
                # elif self.stage_idx == 2:
                #     x = x + back_fea[2]
                # elif self.stage_idx == 3:
                #     x = x + back_fea[3]
                if self.stage_idx == 0:
                    x = x + back_fea[1]
                if self.stage_idx == 1:
                    x = x + back_fea[2]
                elif self.stage_idx == 2:
                    x = x + back_fea[3]

                x_list = x

        for i, blk in enumerate(self.blocks):
            x_list = blk(x_list, messages)

        return x_list

class InvPT(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        self.num_classes = num_classes

        # self.all_tasks = p.TASKS.NAMES
        # task_no = len(self.all_tasks)
        task_no = 1
        self.task_no = task_no

        self.num_stages = spec['NUM_STAGES']

        embed_dim = in_chans
        self.embed_dim = embed_dim

        mt_in_chans = embed_dim
        self.norm_mts = nn.ModuleList()
        self.mt_embed_dims = []
        target_channel = in_chans
        self.redu_chan = nn.ModuleList()
        self.invpt_stages = nn.ModuleList()
        for i in range(self.num_stages):
            cur_mt_embed_dim = spec['DIM_EMBED'][i]
            kwargs = {
                'task_no': task_no,
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': cur_mt_embed_dim,
                'depth': 1,
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': 0,
                'attn_drop_rate': 0,
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'q_method': spec['Q_PROJ_METHOD'][i],
                'kv_method': spec['KV_PROJ_METHOD'][i],
                'kernel_size_q': spec['KERNEL_Q'][i],
                'kernel_size_kv': spec['KERNEL_KV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
            }
            stage = InvPTStage(
                stage_idx=i,
                in_chans=mt_in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            self.invpt_stages.append(stage)
            mt_in_chans = cur_mt_embed_dim
            self.norm_mts.append(norm_layer(mt_in_chans*task_no))
            self.mt_embed_dims.append(mt_in_chans)
            self.redu_chan = nn.Conv2d(mt_in_chans, target_channel,1)

        self.mt_embed_dim = target_channel
        mt_in_chans = task_no * mt_in_chans
        self.norm_mt = norm_layer(mt_in_chans)

        # Final convs
        # self.mt_proj = nn.ModuleDict()
        self.mt_proj = nn.Sequential(nn.Conv2d(self.mt_embed_dim, self.mt_embed_dim, 3, padding=1), BATCHNORM(self.mt_embed_dim), nn.ReLU(True))
        trunc_normal_(self.mt_proj[0].weight, std=0.02)



    def forward(self, x, back_fea):
        '''
        Input:
        x_dict: dict of feature map lists {task: [torch.tensor([B, H*W, embed_dim]), xxx]}
        '''
        # x_list = []
        #
        # for inp_t in self.all_tasks:
        #     _x = x_dict[inp_t]
        #     _x = torch.cat([_x, inter_pred[inp_t]], dim=1)
        #     _x = self.mix_proj[inp_t](_x)
        #     x_list.append(_x)

        messages = {'attn': None}

        # h, w = self.p.mtt_resolution
        # print(x.shape)
        _, _, h, w = x.shape
        th = h * 2**(self.num_stages-1) * 2
        tw = w * 2**(self.num_stages-1) * 2
        # print(th, tw)
        multi_scale_task_feature = 0

        for i in range(self.num_stages):
            x_list = self.invpt_stages[i](x, messages, back_fea)
            _x_list = rearrange(x_list, 'b c h w -> b (h w) c')
            # x = torch.cat(_x_list, dim=2)
            x = _x_list
            x = self.norm_mts[i](x)

            nh = h * 2 ** (i)
            nw = w * 2 ** (i)

            x = rearrange(x, 'b (h w) c -> b c h w', h=nh, w=nw)
            # print('x', x.shape)
            task_x = x
            # if i > 0:
            #     task_x = self.redu_chan(task_x)
            task_x = F.interpolate(task_x, size=(th, tw), mode='bilinear', align_corners=False)
            # add feature from all the scales
            multi_scale_task_feature += task_x

        # print('1', multi_scale_task_feature.shape)
        x_dict = self.mt_proj(multi_scale_task_feature)
        # print('x_dict', x_dict.shape)

        return x_dict
