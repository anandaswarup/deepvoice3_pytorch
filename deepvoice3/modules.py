"""DeepVoice3 modules"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from deepvoice3.common import init_position_encoding, sinusoidal_encode
from deepvoice3.conv import IncrementalDilatedConv1D


def Linear(in_features, out_features, dropout=0):
    """Weight normalized Linear layer
    """
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()

    return nn.utils.weight_norm(m)


def Embedding(num_embeddings, embedding_dim, padding_idx, std=0.01):
    """Embedding layer with the embeddings drawn from a normal distribution with mean=0
    """
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, std)

    return m


def Conv1D(in_channels,
           out_channels,
           kernel_size,
           dropout=0,
           std_mul=4.0,
           **kwargs):
    """Weight normalized 1-D convolution layer with incremental dilations
    """
    m = IncrementalDilatedConv1D(in_channels, out_channels, kernel_size,
                                 **kwargs)
    std = math.sqrt(
        (std_mul * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()

    return nn.utils.weight_norm(m)


def ConvTranspose1D(in_channels,
                    out_channels,
                    kernel_size,
                    dropout=0,
                    std_mul=1.0,
                    **kwargs):
    """Weight normalized 1-D transposed convolutions
    """
    m = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, **kwargs)

    std = math.sqrt(
        (std_mul * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()

    return nn.utils.weight_norm(m)


class PositionalEncoding(nn.Embedding):
    """Positional encoding layer
    """
    def __init__(self, num_embeddings, embedding_dim, *args, **kwargs):
        """Initialize the layer
        """
        super().__init__(num_embeddings,
                         embedding_dim,
                         padding_idx=0,
                         *args,
                         **kwargs)

        self.weight.data = init_position_encoding(num_embeddings,
                                                  embedding_dim,
                                                  position_rate=1.0,
                                                  sinusoidal=False)

    def forward(self, x, w=1.0):
        """Forward pass
        """
        isscaler = np.isscalar(w)
        assert self.padding_idx is not None

        if isscaler or w.size(0) == 1:
            weight = sinusoidal_encode(self.weight, w)
            return F.embedding(x, weight, self.padding_idx, self.max_norm,
                               self.norm_type, self.scale_grad_by_freq,
                               self.sparse)
        else:
            pe = []
            for batch_idx, we in enumerate(w):
                weight = sinusoidal_encode(self.weight, we)
                pe.append(
                    F.embedding(x[batch_idx], weight, self.padding_idx,
                                self.max_norm, self.norm_type,
                                self.scale_grad_by_freq, self.sparse))
            pe = torch.stack(pe)
            return pe


class Conv1DGLU(nn.Module):
    """Weight normalized 1-D convolution with incremental dilations + Gated linear unit + speaker embedding (optional)
    """
    def __init__(self,
                 n_speakers,
                 speaker_embed_dim,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout,
                 padding=None,
                 dilation=1,
                 causal=False,
                 residual=False,
                 *args,
                 **kwargs):
        """Initialize the layer
        """
        super().__init__()

        self.dropout = dropout
        self.residual = residual

        if padding is None:
            # no future time stamps available
            if causal:
                padding = (kernel_size - 1) * dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
        self.causal = causal

        self.conv = Conv1D(in_channels,
                           2 * out_channels,
                           kernel_size,
                           dropout=dropout,
                           padding=padding,
                           dilation=dilation,
                           *args,
                           **kwargs)
        if n_speakers > 1:
            self.speaker_proj = Linear(speaker_embed_dim, out_channels)
        else:
            self.speaker_proj = None

    def forward(self, x, speaker_embed=None):
        """Forward pass
        """
        return self._forward(x, speaker_embed, False)

    def incremental_forward(self, x, speaker_embed=None):
        """Incremental forward (inference)
        """
        return self._forward(x, speaker_embed, True)

    def _forward(self, x, speaker_embed, is_incremental):
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        if is_incremental:
            splitdim = -1
            x = self.conv.incremental_forward(x)
        else:
            splitdim = 1
            x = self.conv(x)
            # remove future time steps if causal
            x = x[:, :, :residual.size(-1)] if self.causal else x

        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
        if self.speaker_proj is not None:
            softsign = F.softsign(self.speaker_proj(speaker_embed))
            softsign = softsign if is_incremental else softsign.transpose(1, 2)
            a = a + softsign
        x = a * torch.sigmoid(b)
        return (x + residual) * math.sqrt(0.5) if self.residual else x

    def clear_buffer(self):
        self.conv.clear_buffer()
