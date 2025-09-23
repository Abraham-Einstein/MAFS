
from MayNet_response_final import Encoder, FusionDecoder, SegmentationDecoder
import torch.nn as nn


class NetworkStage1(nn.Module):
    def __init__(self):
        super(NetworkStage1, self).__init__()
        self.decoder_dim_rec = 32
        self.encoder = Encoder()
        self.fuse_decoder = FusionDecoder()

    def forward(self, vi, ir):
        vi = vi  # 3 channels
        ir = ir[:, :1, ...]  # 1 channel
        rec_f, _, seg_f, _ = self.encoder(vi, ir)

        rec_vi = self.fuse_decoder(rec_f)
        rec_ir = self.fuse_decoder(rec_f)

        return rec_vi, rec_ir


class NetworkStage2(nn.Module):
    def __init__(self, n_classes=9):
        super(NetworkStage2, self).__init__()
        self.encoder = Encoder()
        self.fuse_decoder = FusionDecoder()
        self.seg_decoder = SegmentationDecoder()

    def forward(self, vi, ir):
        vi = vi  # 3 channels
        ir = ir[:, :1, ...]  # 1 channel

        rec_f, rec_f1, _, seg_f = self.encoder(vi, ir)
        semantic_out, out1, out2, out3, out4, out5, out6 = self.seg_decoder(rec_f1, seg_f)
        fused_img = self.fuse_decoder(rec_f)

        return semantic_out, out1, out2, out3, out4, out5, out6, fused_img