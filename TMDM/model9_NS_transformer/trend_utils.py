import torch
import torch.nn.functional as F


def moving_average_trend(x, kernel_size):
    """
    Extract trend component by moving average.

    Args:
        x: [B, L, C]
        kernel_size: int

    Returns:
        trend: [B, L, C]
    """
    if kernel_size <= 1:
        return x

    pad = (kernel_size - 1) // 2

    # [B, L, C] -> [B, C, L]
    x_t = x.permute(0, 2, 1)

    # replicate padding to keep the same length
    front = x_t[:, :, 0:1].repeat(1, 1, pad)
    end = x_t[:, :, -1:].repeat(1, 1, pad)
    x_pad = torch.cat([front, x_t, end], dim=2)

    trend = F.avg_pool1d(x_pad, kernel_size=kernel_size, stride=1)
    trend = trend.permute(0, 2, 1)
    return trend


def series_decomp(x, kernel_size):
    """
    Moving-average decomposition:
        x = trend + residual
    """
    trend = moving_average_trend(x, kernel_size)
    residual = x - trend
    return trend, residual


def build_future_trend_context(x_trend, pred_trend, label_len):
    """
    Build the full trend sequence used in the decoder / residual target.

    Args:
        x_trend:   [B, seq_len, C]
        pred_trend:[B, pred_len, C]
        label_len: int

    Returns:
        trend_full: [B, label_len + pred_len, C]
    """
    trend_label = x_trend[:, -label_len:, :]
    trend_full = torch.cat([trend_label, pred_trend], dim=1)
    return trend_full


def build_residual_decoder_input(x_residual, pred_len, label_len):
    """
    Residual decoder input:
        use the residual context from history for the label part,
        and zeros for the future prediction slots.

    Args:
        x_residual: [B, seq_len, C]

    Returns:
        dec_inp_res: [B, label_len + pred_len, C]
    """
    zeros = torch.zeros(
        x_residual.size(0),
        pred_len,
        x_residual.size(2),
        device=x_residual.device,
        dtype=x_residual.dtype
    )
    dec_inp_res = torch.cat([x_residual[:, -label_len:, :], zeros], dim=1)
    return dec_inp_res
