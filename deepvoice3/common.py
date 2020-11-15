"""Common utilities"""

import numpy as np
import torch


def init_position_encoding(n_position,
                           d_pos_vec,
                           position_rate=1.0,
                           sinusoidal=True):
    """Initialize the position encoding table
    """
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([[
        position_rate * pos / np.power(10000, 2 * (i // 2) / d_pos_vec)
        for i in range(d_pos_vec)
    ] if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc = torch.from_numpy(position_enc).float()
    if sinusoidal:
        position_enc[1:, 0::2] = torch.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = torch.cos(position_enc[1:, 1::2])  # dim 2i+1

    return position_enc


def sinusoidal_encode(x, w):
    y = w * x
    y[1:, 0::2] = torch.sin(y[1:, 0::2].clone())
    y[1:, 1::2] = torch.cos(y[1:, 1::2].clone())

    return y


def get_mask_from_lengths(memory, memory_lengths):
    """Get mask tensor from list of length
    Args:
        memory: (batch, max_time, dim)
        memory_lengths: array like
    """
    max_len = max(memory_lengths)
    mask = torch.arange(max_len).expand(
        memory.size(0), max_len) < torch.tensor(memory_lengths).unsqueeze(-1)
    mask = mask.to(memory.device)

    return ~mask


def expand_speaker_embed(inputs_btc, speaker_embed=None, tdim=1):
    """Expand speaker embedding for all timesteps of input
    """
    if speaker_embed is None:
        return None
    # expand speaker embedding for all time steps
    # (B, N) -> (B, T, N)
    ss = speaker_embed.size()
    speaker_embed_btc = speaker_embed.unsqueeze(1).expand(
        ss[0], inputs_btc.size(tdim), ss[-1])

    return speaker_embed_btc


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        ctx.mark_shared_storage((x, res))
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None
