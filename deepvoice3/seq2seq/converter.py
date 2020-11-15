"""DeepVoice3 converter"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepvoice3.common import expand_speaker_embed
from deepvoice3.modules import Conv1D, Conv1DGLU, ConvTranspose1D


class Converter(nn.Module):
    """Converter to predict a full resolution log-magnitude spectrogram from a coarse mel-spectrogram
    """
    def __init__(self,
                 n_speakers,
                 speaker_embed_dim,
                 in_dim,
                 out_dim,
                 convolutions=((256, 5, 1), ) * 4,
                 time_upsampling=1,
                 dropout=0.1):
        """Initialize the converter
        """
        super().__init__()

        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_speakers = n_speakers

        # Non causual convolution blocks
        in_channels = convolutions[0][0]
        if time_upsampling == 4:
            self.convolutions = nn.ModuleList([
                Conv1D(in_dim,
                       in_channels,
                       kernel_size=1,
                       padding=0,
                       dilation=1,
                       std_mul=1.0),
                ConvTranspose1D(in_channels,
                                in_channels,
                                kernel_size=2,
                                padding=0,
                                stride=2,
                                std_mul=1.0),
                Conv1DGLU(n_speakers,
                          speaker_embed_dim,
                          in_channels,
                          in_channels,
                          kernel_size=3,
                          causal=False,
                          dilation=1,
                          dropout=dropout,
                          std_mul=1.0,
                          residual=True),
                Conv1DGLU(n_speakers,
                          speaker_embed_dim,
                          in_channels,
                          in_channels,
                          kernel_size=3,
                          causal=False,
                          dilation=3,
                          dropout=dropout,
                          std_mul=4.0,
                          residual=True),
                ConvTranspose1D(in_channels,
                                in_channels,
                                kernel_size=2,
                                padding=0,
                                stride=2,
                                std_mul=4.0),
                Conv1DGLU(n_speakers,
                          speaker_embed_dim,
                          in_channels,
                          in_channels,
                          kernel_size=3,
                          causal=False,
                          dilation=1,
                          dropout=dropout,
                          std_mul=1.0,
                          residual=True),
                Conv1DGLU(n_speakers,
                          speaker_embed_dim,
                          in_channels,
                          in_channels,
                          kernel_size=3,
                          causal=False,
                          dilation=3,
                          dropout=dropout,
                          std_mul=4.0,
                          residual=True),
            ])
        elif time_upsampling == 2:
            self.convolutions = nn.ModuleList([
                Conv1D(in_dim,
                       in_channels,
                       kernel_size=1,
                       padding=0,
                       dilation=1,
                       std_mul=1.0),
                ConvTranspose1D(in_channels,
                                in_channels,
                                kernel_size=2,
                                padding=0,
                                stride=2,
                                std_mul=1.0),
                Conv1DGLU(n_speakers,
                          speaker_embed_dim,
                          in_channels,
                          in_channels,
                          kernel_size=3,
                          causal=False,
                          dilation=1,
                          dropout=dropout,
                          std_mul=1.0,
                          residual=True),
                Conv1DGLU(n_speakers,
                          speaker_embed_dim,
                          in_channels,
                          in_channels,
                          kernel_size=3,
                          causal=False,
                          dilation=3,
                          dropout=dropout,
                          std_mul=4.0,
                          residual=True),
            ])
        elif time_upsampling == 1:
            self.convolutions = nn.ModuleList([
                # 1x1 convolution first
                Conv1D(in_dim,
                       in_channels,
                       kernel_size=1,
                       padding=0,
                       dilation=1,
                       std_mul=1.0),
                Conv1DGLU(n_speakers,
                          speaker_embed_dim,
                          in_channels,
                          in_channels,
                          kernel_size=3,
                          causal=False,
                          dilation=3,
                          dropout=dropout,
                          std_mul=4.0,
                          residual=True),
            ])
        else:
            raise ValueError("Not supported")

        std_mul = 4.0
        for (out_channels, kernel_size, dilation) in convolutions:
            if in_channels != out_channels:
                self.convolutions.append(
                    Conv1D(in_channels,
                           out_channels,
                           kernel_size=1,
                           padding=0,
                           dilation=1,
                           std_mul=std_mul))
                self.convolutions.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                std_mul = 2.0
            self.convolutions.append(
                Conv1DGLU(n_speakers,
                          speaker_embed_dim,
                          in_channels,
                          out_channels,
                          kernel_size,
                          causal=False,
                          dilation=dilation,
                          dropout=dropout,
                          std_mul=std_mul,
                          residual=True))
            in_channels = out_channels
            std_mul = 4.0
        # Last 1x1 convolution
        self.convolutions.append(
            Conv1D(in_channels,
                   out_dim,
                   kernel_size=1,
                   padding=0,
                   dilation=1,
                   std_mul=std_mul,
                   dropout=dropout))

    def forward(self, x, speaker_embed=None):
        """Forward pass
        """
        assert self.n_speakers == 1 or speaker_embed is not None

        # expand speaker embedding for all time steps
        speaker_embed_btc = expand_speaker_embed(x, speaker_embed)
        if speaker_embed_btc is not None:
            speaker_embed_btc = F.dropout(speaker_embed_btc,
                                          p=self.dropout,
                                          training=self.training)

        # [B, T_max, channels] -> [B, channels, T_max]
        x = x.transpose(1, 2).contiguous()

        for f in self.convolutions:
            if speaker_embed_btc is not None and speaker_embed_btc.size(
                    1) != x.size(-1):
                speaker_embed_btc = expand_speaker_embed(x,
                                                         speaker_embed,
                                                         tdim=-1)
                speaker_embed_btc = F.dropout(speaker_embed_btc,
                                              p=self.dropout,
                                              training=self.training)
            x = f(x, speaker_embed_btc) if isinstance(f, Conv1DGLU) else f(x)

        x = x.transpose(1, 2).contiguous()

        return torch.sigmoid(x)
