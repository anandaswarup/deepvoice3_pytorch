"""DeepVoice3 seq2seq encoder"""

import math

import torch.nn as nn
import torch.nn.functional as F
from deepvoice3.common import GradMultiply, expand_speaker_embed
from deepvoice3.modules import Conv1D, Conv1DGLU, Embedding, Linear


class Encoder(nn.Module):
    """DeepVoice3 seq2seq Encoder
    """
    def __init__(self,
                 n_vocab,
                 embed_dim,
                 n_speakers,
                 speaker_embed_dim,
                 padding_idx=None,
                 embedding_weight_std=0.1,
                 convolutions=((64, 5, .1), ) * 7,
                 dropout=0.1,
                 apply_grad_scaling=False):
        """Initialize the encoder
        """
        super().__init__()

        self.dropout = dropout
        self.num_attention_layers = None
        self.apply_grad_scaling = apply_grad_scaling

        # Text embedding
        self.text_embed = Embedding(n_vocab, embed_dim, padding_idx,
                                    embedding_weight_std)

        # Speaker embedding
        if n_speakers > 1:
            self.speaker_fc1 = Linear(speaker_embed_dim,
                                      embed_dim,
                                      dropout=dropout)
            self.speaker_fc2 = Linear(speaker_embed_dim,
                                      embed_dim,
                                      dropout=dropout)
        self.n_speakers = n_speakers

        # Non causual convolution blocks
        in_channels = embed_dim
        self.convolutions = nn.ModuleList()
        std_mul = 1.0
        for (out_channels, kernel_size, dilation) in convolutions:
            if in_channels != out_channels:
                # Conv1D + ReLU
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
                   embed_dim,
                   kernel_size=1,
                   padding=0,
                   dilation=1,
                   std_mul=std_mul,
                   dropout=dropout))

    def forward(self, text_sequences, speaker_embed=None):
        """Forward pass
        """
        assert self.n_speakers == 1 or speaker_embed is not None

        # Text embedding
        x = self.text_embed(text_sequences.long())
        x = F.dropout(x, p=self.dropout, training=self.training)

        # expand speaker embedding for all time steps
        speaker_embed_btc = expand_speaker_embed(x, speaker_embed)
        if speaker_embed_btc is not None:
            speaker_embed_btc = F.dropout(speaker_embed_btc,
                                          p=self.dropout,
                                          training=self.training)
            x = x + F.softsign(self.speaker_fc1(speaker_embed_btc))

        input_embedding = x

        # [B, T_max, channels] -> [B, channels, T_max]
        x = x.transpose(1, 2).contiguous()

        # １D conv blocks
        for f in self.convolutions:
            x = f(x, speaker_embed_btc) if isinstance(f, Conv1DGLU) else f(x)

        # [B, channels, T_max] -> [B, T_max, channels]
        keys = x.transpose(1, 2).contiguous()

        if speaker_embed_btc is not None:
            keys = keys + F.softsign(self.speaker_fc2(speaker_embed_btc))

        # scale gradients (this only affects backward, not forward)
        if self.apply_grad_scaling and self.num_attention_layers is not None:
            keys = GradMultiply.apply(keys,
                                      1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        values = (keys + input_embedding) * math.sqrt(0.5)

        return keys, values
