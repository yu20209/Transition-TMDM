import torch
import torch.nn as nn


class TrendLinear(nn.Module):
    """
    Lightweight trend forecasting head.

    Input:
        history_trend: [B, seq_len, C]
    Output:
        future_trend_pred: [B, pred_len, C]
    """

    def __init__(self, configs):
        super(TrendLinear, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = getattr(configs, "trend_individual", True)

        if self.individual:
            self.linears = nn.ModuleList([
                nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)
            ])
        else:
            self.linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, history_trend):
        # history_trend: [B, seq_len, C]
        batch_size, seq_len, num_channels = history_trend.shape
        assert num_channels == self.channels

        if self.individual:
            future_trend_pred_list = []
            for channel_idx in range(num_channels):
                future_trend_one_channel = self.linears[channel_idx](history_trend[:, :, channel_idx])
                future_trend_pred_list.append(future_trend_one_channel.unsqueeze(-1))
            future_trend_pred = torch.cat(future_trend_pred_list, dim=-1)
        else:
            history_trend_t = history_trend.permute(0, 2, 1)
            future_trend_pred = self.linear(history_trend_t)
            future_trend_pred = future_trend_pred.permute(0, 2, 1)

        return future_trend_pred
