"""Attention Layer"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepvoice3.modules import Linear


class AttentionLayer(nn.Module):
    """DeepVoice3 attention layer
    """
    def __init__(self,
                 conv_channels,
                 embed_dim,
                 dropout=0.1,
                 window_ahead=3,
                 window_backward=1,
                 key_projection=True,
                 value_projection=True):
        """Initialize the attention layer
        """
        super().__init__()

        self.query_projection = Linear(conv_channels, embed_dim)
        if key_projection:
            self.key_projection = Linear(embed_dim, embed_dim)
            if conv_channels == embed_dim:
                self.key_projection.weight.data = self.query_projection.weight.data.clone(
                )
        else:
            self.key_projection = None

        if value_projection:
            self.value_projection = Linear(embed_dim, embed_dim)
        else:
            self.value_projection = None

        self.out_projection = Linear(embed_dim, conv_channels)

        self.dropout = dropout
        self.window_ahead = window_ahead
        self.window_backward = window_backward

    def forward(self, query, encoder_out, mask=None, last_attended=None):
        """Forward pass
        """
        keys, values = encoder_out
        residual = query

        if self.value_projection is not None:
            values = self.value_projection(values)

        if self.key_projection is not None:
            keys = self.key_projection(keys.transpose(1, 2)).transpose(1, 2)

        # Compute attention scores
        x = self.query_projection(query)
        x = torch.bmm(x, keys)

        # Mask attention values
        mask_value = -float("inf")
        if mask is not None:
            mask = mask.view(query.size(0), 1, -1)
            x.data.masked_fill_(mask, mask_value)

        if last_attended is not None:
            backward = last_attended - self.window_backward
            if backward > 0:
                x[:, :, :backward] = mask_value
            ahead = last_attended + self.window_ahead
            if ahead < x.size(-1):
                x[:, :, ahead:] = mask_value

        # Normalize attention scores
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = torch.bmm(x, values)

        # scale attention output
        s = values.size(1)
        x = x * (s * math.sqrt(1.0 / s))

        # project back
        x = self.out_projection(x)
        x = (x + residual) * math.sqrt(0.5)

        return x, attn_scores
