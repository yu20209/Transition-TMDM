import torch
import torch.nn.functional as F


def moving_average_trend(input_series, kernel_size):
    """
    Extract trend component by moving average.

    Args:
        input_series: [B, L, C]
        kernel_size: int

    Returns:
        trend_series: [B, L, C]
    """
    if kernel_size <= 1:
        return input_series

    pad = (kernel_size - 1) // 2

    # [B, L, C] -> [B, C, L]
    input_series_t = input_series.permute(0, 2, 1)

    front = input_series_t[:, :, 0:1].repeat(1, 1, pad)
    end = input_series_t[:, :, -1:].repeat(1, 1, pad)
    input_series_pad = torch.cat([front, input_series_t, end], dim=2)

    trend_series = F.avg_pool1d(input_series_pad, kernel_size=kernel_size, stride=1)
    trend_series = trend_series.permute(0, 2, 1)
    return trend_series


def series_decomp(input_series, kernel_size):
    """
    Moving-average decomposition:
        input_series = trend_series + residual_series
    """
    trend_series = moving_average_trend(input_series, kernel_size)
    residual_series = input_series - trend_series
    return trend_series, residual_series


def build_future_trend_context(history_trend, future_trend_pred, label_len):
    """
    Build full trend sequence for decoder / residual target.

    Args:
        history_trend:     [B, seq_len, C]
        future_trend_pred: [B, pred_len, C]
        label_len: int

    Returns:
        full_trend_context: [B, label_len + pred_len, C]
    """
    trend_label_context = history_trend[:, -label_len:, :]
    full_trend_context = torch.cat([trend_label_context, future_trend_pred], dim=1)
    return full_trend_context


def build_residual_decoder_input(history_residual, pred_len, label_len):
    """
    Residual decoder input:
    use residual history for label part and zero placeholders for future part.

    Args:
        history_residual: [B, seq_len, C]

    Returns:
        residual_decoder_input: [B, label_len + pred_len, C]
    """
    future_zero_placeholder = torch.zeros(
        history_residual.size(0),
        pred_len,
        history_residual.size(2),
        device=history_residual.device,
        dtype=history_residual.dtype
    )
    residual_decoder_input = torch.cat(
        [history_residual[:, -label_len:, :], future_zero_placeholder],
        dim=1
    )
    return residual_decoder_input
