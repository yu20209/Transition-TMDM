import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()

        # 过渡版：现在 _build_model 返回 3 个模型：
        # 1) diffusion model
        # 2) residual condition model (原 TMDM 的 cond model，但现在预测 residual mean)
        # 3) trend model
        model, cond_pred_model, trend_model = self._build_model()

        self.model = model.to(self.device)
        self.cond_pred_model = cond_pred_model.to(self.device)
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
            # 原仓库这里默认走 mps；如果你机器没有 mps，可改成 cpu
            try:
                device_name = 'mps:0'
                device = torch.device(device_name)
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
