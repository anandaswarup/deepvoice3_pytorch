"""DeepVoice3 seq2seq decoder"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepvoice3.common import expand_speaker_embed, get_mask_from_lengths
from deepvoice3.modules import Conv1D, Conv1DGLU, Linear, PositionalEncoding
from deepvoice3.seq2seq.attention import AttentionLayer


class Decoder(nn.Module):
    """DeepVoice3 seq2seq decoder
    """
    def __init__(
        self,
        embed_dim,
        n_speakers,
        speaker_embed_dim,
        in_dim=80,
        r=5,
        max_positions=512,
        preattention=((128, 5, 1), ) * 4,
        convolutions=((128, 5, 1), ) * 4,
        attention=True,
        dropout=0.1,
        use_memory_mask=False,
        force_monotonic_attention=False,
        query_position_rate=1.0,
        key_position_rate=1.29,
        window_ahead=3,
        window_backward=1,
        key_projection=True,
        value_projection=True,
    ):
        """Initialize the decoder
        """
        super().__init__()

        self.dropout = dropout
        self.in_dim = in_dim
        self.r = r
        self.query_position_rate = query_position_rate
        self.key_position_rate = key_position_rate

        in_channels = in_dim * r

        if isinstance(attention, bool):
            # expand True into [True, True, ...] and do the same with False
            attention = [attention] * len(convolutions)

        # Position encodings for query (decoder states) and keys (encoder states)
        self.embed_query_positions = PositionalEncoding(
            max_positions, convolutions[0][0])

        self.embed_keys_positions = PositionalEncoding(max_positions,
                                                       embed_dim)

        # Used for compute multiplier for positional encodings
        if n_speakers > 1:
            self.speaker_proj1 = Linear(speaker_embed_dim, 1, dropout=dropout)
            self.speaker_proj2 = Linear(speaker_embed_dim, 1, dropout=dropout)
        else:
            self.speaker_proj1, self.speaker_proj2 = None, None

        # Prenet: causal convolution blocks
        self.preattention = nn.ModuleList()
        in_channels = in_dim * r
        std_mul = 1.0
        for out_channels, kernel_size, dilation in preattention:
            if in_channels != out_channels:
                # Conv1d + ReLU
                self.preattention.append(
                    Conv1D(in_channels,
                           out_channels,
                           kernel_size=1,
                           padding=0,
                           dilation=1,
                           std_mul=std_mul))
                self.preattention.append(nn.ReLU(inplace=True))
                in_channels = out_channels
                std_mul = 2.0
            self.preattention.append(
                Conv1DGLU(n_speakers,
                          speaker_embed_dim,
                          in_channels,
                          out_channels,
                          kernel_size,
                          causal=True,
                          dilation=dilation,
                          dropout=dropout,
                          std_mul=std_mul,
                          residual=True))
            in_channels = out_channels
            std_mul = 4.0

        # Causal convolution blocks + attention layers
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()

        for i, (out_channels, kernel_size,
                dilation) in enumerate(convolutions):
            assert in_channels == out_channels
            self.convolutions.append(
                Conv1DGLU(n_speakers,
                          speaker_embed_dim,
                          in_channels,
                          out_channels,
                          kernel_size,
                          causal=True,
                          dilation=dilation,
                          dropout=dropout,
                          std_mul=std_mul,
                          residual=False))
            self.attention.append(
                AttentionLayer(out_channels,
                               embed_dim,
                               dropout=dropout,
                               window_ahead=window_ahead,
                               window_backward=window_backward,
                               key_projection=key_projection,
                               value_projection=value_projection
                               ) if attention[i] else None)
            in_channels = out_channels
            std_mul = 4.0
        # Last 1x1 convolution
        self.last_conv = Conv1D(in_channels,
                                in_dim * r,
                                kernel_size=1,
                                padding=0,
                                dilation=1,
                                std_mul=std_mul,
                                dropout=dropout)

        # Mel-spectrogram (before sigmoid) -> Done binary flag
        self.fc = Linear(in_dim * r, 1)

        self.max_decoder_steps = 200
        self.min_decoder_steps = 10
        self.use_memory_mask = use_memory_mask
        if isinstance(force_monotonic_attention, bool):
            self.force_monotonic_attention = [force_monotonic_attention
                                              ] * len(convolutions)
        else:
            self.force_monotonic_attention = force_monotonic_attention

    def forward(self,
                encoder_out,
                inputs=None,
                text_positions=None,
                frame_positions=None,
                speaker_embed=None,
                lengths=None):
        """Forward pass
        """
        if inputs is None:
            assert text_positions is not None
            self.start_fresh_sequence()
            outputs = self.incremental_forward(encoder_out, text_positions,
                                               speaker_embed)
            return outputs

        # Grouping multiple frames if necessary
        if inputs.size(-1) == self.in_dim:
            inputs = inputs.view(inputs.size(0), inputs.size(1) // self.r, -1)
        assert inputs.size(-1) == self.in_dim * self.r

        # expand speaker embedding for all time steps
        speaker_embed_btc = expand_speaker_embed(inputs, speaker_embed)
        if speaker_embed_btc is not None:
            speaker_embed_btc = F.dropout(speaker_embed_btc,
                                          p=self.dropout,
                                          training=self.training)

        keys, values = encoder_out

        if self.use_memory_mask and lengths is not None:
            mask = get_mask_from_lengths(keys, lengths)
        else:
            mask = None

        # position encodings
        if text_positions is not None:
            w = self.key_position_rate
            if self.speaker_proj1 is not None:
                w = w * torch.sigmoid(
                    self.speaker_proj1(speaker_embed)).view(-1)
            text_pos_embed = self.embed_keys_positions(text_positions, w)
            keys = keys + text_pos_embed

        if frame_positions is not None:
            w = self.query_position_rate
            if self.speaker_proj2 is not None:
                w = w * torch.sigmoid(
                    self.speaker_proj2(speaker_embed)).view(-1)
            frame_pos_embed = self.embed_query_positions(frame_positions, w)

        # transpose only once to speed up attention layers
        keys = keys.transpose(1, 2).contiguous()

        x = inputs
        x = F.dropout(x, p=self.dropout, training=self.training)

        # [B, T_max, channels] -> [B, channels, T_max]
        x = x.transpose(1, 2).contiguous()

        # Prenet
        for f in self.preattention:
            x = f(x, speaker_embed_btc) if isinstance(f, Conv1DGLU) else f(x)

        # Casual convolutions + Multi-hop attentions
        alignments = []
        for f, attention in zip(self.convolutions, self.attention):
            residual = x

            x = f(x, speaker_embed_btc) if isinstance(f, Conv1DGLU) else f(x)

            # Feed conv output to attention layer as query
            if attention is not None:
                assert isinstance(f, Conv1DGLU)
                # (B x T x C)
                x = x.transpose(1, 2).contiguous()
                x = x if frame_positions is None else x + frame_pos_embed
                x, alignment = attention(x, (keys, values), mask=mask)
                x = x.transpose(1, 2).contiguous()
                alignments += [alignment]

            if isinstance(f, Conv1DGLU):
                x = (x + residual) * math.sqrt(0.5)

        # decoder state [B, T_max, channels]:
        # internal representation before compressed to output dimention
        decoder_states = x.transpose(1, 2).contiguous()
        x = self.last_conv(x)

        x = x.transpose(1, 2).contiguous()

        # project to mel-spectorgram
        outputs = torch.sigmoid(x)

        # done flag
        done = torch.sigmoid(self.fc(x))

        return outputs, torch.stack(alignments), done, decoder_states

    def incremental_forward(self,
                            encoder_out,
                            text_positions,
                            speaker_embed=None,
                            initial_input=None,
                            test_inputs=None):
        """Incremental forward (Inference)
        """
        keys, values = encoder_out
        B = keys.size(0)

        # position encodings
        w = self.key_position_rate
        if self.speaker_proj1 is not None:
            w = w * torch.sigmoid(self.speaker_proj1(speaker_embed)).view(-1)
        text_pos_embed = self.embed_keys_positions(text_positions, w)
        keys = keys + text_pos_embed

        # transpose only once to speed up attention layers
        keys = keys.transpose(1, 2).contiguous()

        decoder_states = []
        outputs = []
        alignments = []
        dones = []
        # intially set to zeros
        last_attended = [None] * len(self.attention)
        for idx, v in enumerate(self.force_monotonic_attention):
            last_attended[idx] = 0 if v else None

        num_attention_layers = sum(
            [layer is not None for layer in self.attention])
        t = 0
        if initial_input is None:
            initial_input = keys.data.new(B, 1, self.in_dim * self.r).zero_()
        current_input = initial_input
        while True:
            frame_pos = keys.data.new(B, 1).fill_(t + 1).long()
            w = self.query_position_rate
            if self.speaker_proj2 is not None:
                w = w * torch.sigmoid(
                    self.speaker_proj2(speaker_embed)).view(-1)
            frame_pos_embed = self.embed_query_positions(frame_pos, w)

            if test_inputs is not None:
                if t >= test_inputs.size(1):
                    break
                current_input = test_inputs[:, t, :].unsqueeze(1)
            else:
                if t > 0:
                    current_input = outputs[-1]
            x = current_input
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Prenet
            for f in self.preattention:
                if isinstance(f, Conv1DGLU):
                    x = f.incremental_forward(x, speaker_embed)
                else:
                    try:
                        x = f.incremental_forward(x)
                    except AttributeError:
                        x = f(x)

            # Casual convolutions + Multi-hop attentions
            ave_alignment = None
            for idx, (f, attention) in enumerate(
                    zip(self.convolutions, self.attention)):
                residual = x
                if isinstance(f, Conv1DGLU):
                    x = f.incremental_forward(x, speaker_embed)
                else:
                    try:
                        x = f.incremental_forward(x)
                    except AttributeError:
                        x = f(x)

                # attention
                if attention is not None:
                    assert isinstance(f, Conv1DGLU)
                    x = x + frame_pos_embed
                    x, alignment = attention(x, (keys, values),
                                             last_attended=last_attended[idx])
                    if self.force_monotonic_attention[idx]:
                        last_attended[idx] = alignment.max(-1)[1].view(
                            -1).data[0]
                    if ave_alignment is None:
                        ave_alignment = alignment
                    else:
                        ave_alignment = ave_alignment + ave_alignment

                # residual
                if isinstance(f, Conv1DGLU):
                    x = (x + residual) * math.sqrt(0.5)

            decoder_state = x
            x = self.last_conv.incremental_forward(x)
            ave_alignment = ave_alignment.div_(num_attention_layers)

            # Ooutput & done flag predictions
            output = torch.sigmoid(x)
            done = torch.sigmoid(self.fc(x))

            decoder_states += [decoder_state]
            outputs += [output]
            alignments += [ave_alignment]
            dones += [done]

            t += 1
            if test_inputs is None:
                if (done > 0.5).all() and t > self.min_decoder_steps:
                    break
                elif t > self.max_decoder_steps:
                    break

        # Remove 1-element time axis
        alignments = list(map(lambda x: x.squeeze(1), alignments))
        decoder_states = list(map(lambda x: x.squeeze(1), decoder_states))
        outputs = list(map(lambda x: x.squeeze(1), outputs))

        # Combine outputs for all time steps
        alignments = torch.stack(alignments).transpose(0, 1)
        decoder_states = torch.stack(decoder_states).transpose(0,
                                                               1).contiguous()
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()

        return outputs, alignments, dones, decoder_states

    def start_fresh_sequence(self):
        _clear_modules(self.preattention)
        _clear_modules(self.convolutions)
        self.last_conv.clear_buffer()


def _clear_modules(modules):
    for m in modules:
        try:
            m.clear_buffer()
        except AttributeError:
            pass
