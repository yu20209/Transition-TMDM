import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()

        # 过渡版：返回三个模型
        # 1) residual diffusion model
        # 2) residual prior mean model
        # 3) trend forecasting model
        model, residual_prior_model, trend_model = self._build_model()

        self.model = model.to(self.device)
        self.cond_pred_model = residual_prior_model.to(self.device)
        self.trend_model = trend_model.to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None, None, None

    def _acquire_device(self):
        if torch.cuda.is_available():
            if self.args.use_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: cuda:{}'.format(self.args.gpu))
            else:
                device = torch.device('cpu')
                print('Use CPU')
        else:
            try:
                device = torch.device('mps:0')
                print('Use MPS')
            except Exception:
                device = torch.device('cpu')
                print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
