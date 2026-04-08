import torch
import torch.nn as nn


class TrendLinear(nn.Module):
    """
    A very light-weight trend forecasting head.
    Similar in spirit to channel-wise linear forecasting.

    Input:
        x_trend: [B, seq_len, C]
    Output:
        pred_trend: [B, pred_len, C]
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

    def forward(self, x_trend):
        # x_trend: [B, seq_len, C]
        B, L, C = x_trend.shape
        assert C == self.channels

        if self.individual:
            out = []
            for i in range(C):
                # [B, L] -> [B, pred_len]
                out_i = self.linears[i](x_trend[:, :, i])
                out.append(out_i.unsqueeze(-1))
            out = torch.cat(out, dim=-1)  # [B, pred_len, C]
        else:
            # [B, seq_len, C] -> [B, C, seq_len]
            x_t = x_trend.permute(0, 2, 1)
            out = self.linear(x_t)        # [B, C, pred_len]
            out = out.permute(0, 2, 1)    # [B, pred_len, C]

        return out
