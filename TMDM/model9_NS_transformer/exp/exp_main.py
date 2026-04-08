from data_provider.data_factory import data_provider

from utils.tools import EarlyStopping
from utils.metrics import metric

from model9_NS_transformer.ns_models import ns_Transformer
from model9_NS_transformer.exp.exp_basic import Exp_Basic
from model9_NS_transformer.diffusion_models import diffuMTS
from model9_NS_transformer.diffusion_models.diffusion_utils import *
from model9_NS_transformer.trend_models.trend_linear import TrendLinear
from model9_NS_transformer.trend_utils import (
    series_decomp,
    moving_average_trend,
    build_future_trend_context,
    build_residual_decoder_input,
)

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time
import math

from multiprocessing import Pool
import CRPS.CRPS as pscore

import warnings

warnings.filterwarnings('ignore')


def ccc(id, pred, true):
    res_box = np.zeros(len(true))
    for i in range(len(true)):
        res = pscore(pred[i], true[i]).compute()
        res_box[i] = res[0]
    return res_box


def log_normal(x, mu, var):
    """
    Logarithm of normal distribution with mean=mu and variance=var
    """
    eps = 1e-8
    if isinstance(var, float):
        var = torch.tensor(var, device=x.device, dtype=x.dtype)

    if eps > 0.0:
        var = var + eps

    return 0.5 * torch.mean(
        np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var
    )


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        # diffusion model on residual space
        model = diffuMTS.Model(self.args, self.device).float()

        # 原 TMDM 的条件均值模型，现在改成 residual mean model
        cond_pred_model = ns_Transformer.Model(self.args).float()

        # 新增：趋势预测模型
        trend_model = TrendLinear(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            cond_pred_model = nn.DataParallel(cond_pred_model, device_ids=self.args.device_ids)
            trend_model = nn.DataParallel(trend_model, device_ids=self.args.device_ids)

        return model, cond_pred_model, trend_model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # 过渡版：联合优化
        # 1) diffusion model
        # 2) residual mean model
        # 3) trend model
        model_optim = optim.Adam(
            [
                {'params': self.model.parameters()},
                {'params': self.cond_pred_model.parameters()},
                {'params': self.trend_model.parameters()},
            ],
            lr=self.args.learning_rate
        )
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _prepare_transition_batch(self, batch_x, batch_y):
        """
        构造过渡版的核心训练对象：
        1) x -> trend_x + residual_x
        2) trend model predicts future trend
        3) residual target = y - predicted trend context

        Returns:
            x_trend, x_residual,
            pred_trend_future,
            trend_full,
            residual_y,
            trend_target_future
        """
        # 历史序列分解
        x_trend, x_residual = series_decomp(batch_x, self.args.trend_kernel)

        # 未来目标的 trend target，仅用于监督趋势模型
        y_trend_target_full = moving_average_trend(batch_y, self.args.trend_kernel)
        trend_target_future = y_trend_target_full[:, -self.args.pred_len:, :]

        # 趋势模型只预测 future trend
        pred_trend_future = self.trend_model(x_trend)

        # 构造 [label_len + pred_len] 的完整趋势上下文
        trend_full = build_future_trend_context(
            x_trend=x_trend,
            pred_trend=pred_trend_future,
            label_len=self.args.label_len
        )

        # residual target 是“真实未来序列 - 预测趋势”
        residual_y = batch_y - trend_full

        return (
            x_trend,
            x_residual,
            pred_trend_future,
            trend_full,
            residual_y,
            trend_target_future
        )

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        self.cond_pred_model.eval()
        self.trend_model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                (
                    x_trend,
                    x_residual,
                    pred_trend_future,
                    trend_full,
                    residual_y,
                    trend_target_future
                ) = self._prepare_transition_batch(batch_x, batch_y)

                # residual decoder input
                dec_inp_res = build_residual_decoder_input(
                    x_residual=x_residual,
                    pred_len=self.args.pred_len,
                    label_len=self.args.label_len
                )

                n = batch_x.size(0)
                t = torch.randint(
                    low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

                # residual mean model（沿用 TMDM 的 condition model）
                _, residual_mean_full, KL_loss, z_sample = self.cond_pred_model(
                    x_residual, batch_x_mark, dec_inp_res, batch_y_mark
                )

                # VAE-style condition loss（现在监督 residual）
                loss_vae = log_normal(
                    residual_y,
                    residual_mean_full,
                    torch.tensor(1.0, device=residual_y.device, dtype=residual_y.dtype)
                )
                loss_vae_all = loss_vae + self.args.k_z * KL_loss

                # trend supervision
                trend_loss = criterion(pred_trend_future, trend_target_future)

                # TMDM diffusion in residual space:
                # q(r_t | r_0, r_hat)
                residual_T_mean = residual_mean_full
                e = torch.randn_like(residual_y).to(self.device)

                residual_t = q_sample(
                    residual_y,
                    residual_T_mean,
                    self.model.alphas_bar_sqrt,
                    self.model.one_minus_alphas_bar_sqrt,
                    t,
                    noise=e
                )

                output = self.model(
                    x_residual,
                    batch_x_mark,
                    residual_y,
                    residual_t,
                    residual_mean_full,
                    t
                )

                diffusion_loss = (e - output).square().mean()

                loss = diffusion_loss + self.args.k_cond * loss_vae_all + self.args.k_trend * trend_loss
                total_loss.append(loss.detach().cpu().item())

        total_loss = np.average(total_loss)
        self.model.train()
        self.cond_pred_model.train()
        self.trend_model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)

        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()

            iter_count = 0
            train_loss = []

            self.model.train()
            self.cond_pred_model.train()
            self.trend_model.train()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                (
                    x_trend,
                    x_residual,
                    pred_trend_future,
                    trend_full,
                    residual_y,
                    trend_target_future
                ) = self._prepare_transition_batch(batch_x, batch_y)

                dec_inp_res = build_residual_decoder_input(
                    x_residual=x_residual,
                    pred_len=self.args.pred_len,
                    label_len=self.args.label_len
                )

                n = batch_x.size(0)
                t = torch.randint(
                    low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

                _, residual_mean_full, KL_loss, z_sample = self.cond_pred_model(
                    x_residual, batch_x_mark, dec_inp_res, batch_y_mark
                )

                # residual condition loss
                loss_vae = log_normal(
                    residual_y,
                    residual_mean_full,
                    torch.tensor(1.0, device=residual_y.device, dtype=residual_y.dtype)
                )
                loss_vae_all = loss_vae + self.args.k_z * KL_loss

                # trend loss
                trend_loss = criterion(pred_trend_future, trend_target_future)

                residual_T_mean = residual_mean_full
                e = torch.randn_like(residual_y).to(self.device)

                residual_t = q_sample(
                    residual_y,
                    residual_T_mean,
                    self.model.alphas_bar_sqrt,
                    self.model.one_minus_alphas_bar_sqrt,
                    t,
                    noise=e
                )

                output = self.model(
                    x_residual,
                    batch_x_mark,
                    residual_y,
                    residual_t,
                    residual_mean_full,
                    t
                )

                diffusion_loss = (e - output).square().mean()

                # 总损失：
                # 1) trend branch
                # 2) residual mean branch
                # 3) residual diffusion branch
                loss = diffusion_loss + self.args.k_cond * loss_vae_all + self.args.k_trend * trend_loss
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )

            # 保存三个模型
            if vali_loss <= getattr(self, "best_val_loss", 1e18):
                self.best_val_loss = vali_loss
                torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint_diffusion.pth'))
                torch.save(self.cond_pred_model.state_dict(), os.path.join(path, 'checkpoint_cond.pth'))
                torch.save(self.trend_model.state_dict(), os.path.join(path, 'checkpoint_trend.pth'))
                print("Validation improved. Saved diffusion/cond/trend checkpoints.")

            early_stopping(vali_loss, self.model, path)

            if math.isnan(train_loss):
                break

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # 回载最优
        diff_path = os.path.join(path, 'checkpoint_diffusion.pth')
        cond_path = os.path.join(path, 'checkpoint_cond.pth')
        trend_path = os.path.join(path, 'checkpoint_trend.pth')

        if os.path.exists(diff_path):
            self.model.load_state_dict(torch.load(diff_path, map_location=self.device))
        if os.path.exists(cond_path):
            self.cond_pred_model.load_state_dict(torch.load(cond_path, map_location=self.device))
        if os.path.exists(trend_path):
            self.trend_model.load_state_dict(torch.load(trend_path, map_location=self.device))

        return self.model

    def test(self, setting, test=0):
        #####################################################################################################
        ########################## local functions within the class function scope ##########################

        def store_gen_y_at_step_t(config, config_diff, idx, y_tile_seq):
            current_t = self.model.num_timesteps - idx
            gen_y = y_tile_seq[idx].reshape(
                config.test_batch_size,
                int(config_diff.testing.n_z_samples / config_diff.testing.n_z_samples_depart),
                (config.label_len + config.pred_len),
                config.c_out
            ).cpu().numpy()

            if len(gen_y_by_batch_list[current_t]) == 0:
                gen_y_by_batch_list[current_t] = gen_y
            else:
                gen_y_by_batch_list[current_t] = np.concatenate([gen_y_by_batch_list[current_t], gen_y], axis=0)
            return gen_y

        def compute_true_coverage_by_gen_QI(config, dataset_object, all_true_y, all_generated_y):
            n_bins = config.testing.n_bins
            quantile_list = np.arange(n_bins + 1) * (100 / n_bins)
            y_pred_quantiles = np.percentile(all_generated_y.squeeze(), q=quantile_list, axis=1)
            y_true = all_true_y.T
            quantile_membership_array = ((y_true - y_pred_quantiles) > 0).astype(int)
            y_true_quantile_membership = quantile_membership_array.sum(axis=0)
            y_true_quantile_bin_count = np.array(
                [(y_true_quantile_membership == v).sum() for v in np.arange(n_bins + 2)]
            )

            y_true_quantile_bin_count[1] += y_true_quantile_bin_count[0]
            y_true_quantile_bin_count[-2] += y_true_quantile_bin_count[-1]
            y_true_quantile_bin_count_ = y_true_quantile_bin_count[1:-1]
            y_true_ratio_by_bin = y_true_quantile_bin_count_ / dataset_object
            assert np.abs(np.sum(y_true_ratio_by_bin) - 1) < 1e-10, "Sum of quantile coverage ratios shall be 1!"
            qice_coverage_ratio = np.absolute(np.ones(n_bins) / n_bins - y_true_ratio_by_bin).mean()
            return y_true_ratio_by_bin, qice_coverage_ratio, y_true

        def compute_PICP(config, y_true, all_gen_y, return_CI=False):
            low, high = config.testing.PICP_range
            CI_y_pred = np.percentile(all_gen_y.squeeze(), q=[low, high], axis=1)
            y_in_range = (y_true >= CI_y_pred[0]) & (y_true <= CI_y_pred[1])
            coverage = y_in_range.mean()
            if return_CI:
                return coverage, CI_y_pred, low, high
            else:
                return coverage, low, high

        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            path = os.path.join('./checkpoints/', setting)
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint_diffusion.pth'), map_location=self.device))
            self.cond_pred_model.load_state_dict(torch.load(os.path.join(path, 'checkpoint_cond.pth'), map_location=self.device))
            self.trend_model.load_state_dict(torch.load(os.path.join(path, 'checkpoint_trend.pth'), map_location=self.device))

        preds = []
        trues = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        minibatch_sample_start = time.time()

        self.model.eval()
        self.cond_pred_model.eval()
        self.trend_model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                gen_y_by_batch_list = [[] for _ in range(self.model.num_timesteps + 1)]
                y_se_by_batch_list = [[] for _ in range(self.model.num_timesteps + 1)]

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # === 1) trend / residual decomposition ===
                x_trend, x_residual = series_decomp(batch_x, self.args.trend_kernel)

                # === 2) trend prediction ===
                pred_trend_future = self.trend_model(x_trend)  # [B, pred_len, C]
                trend_full = build_future_trend_context(
                    x_trend=x_trend,
                    pred_trend=pred_trend_future,
                    label_len=self.args.label_len
                )

                # === 3) residual mean prediction ===
                dec_inp_res = build_residual_decoder_input(
                    x_residual=x_residual,
                    pred_len=self.args.pred_len,
                    label_len=self.args.label_len
                )

                _, residual_mean_full, _, z_sample = self.cond_pred_model(
                    x_residual, batch_x_mark, dec_inp_res, batch_y_mark
                )

                repeat_n = int(
                    self.model.diffusion_config.testing.n_z_samples / self.model.diffusion_config.testing.n_z_samples_depart
                )

                residual_mean_tile = residual_mean_full.repeat(repeat_n, 1, 1, 1)
                residual_mean_tile = residual_mean_tile.transpose(0, 1).flatten(0, 1).to(self.device)

                residual_T_mean_tile = residual_mean_tile

                x_tile = x_residual.repeat(repeat_n, 1, 1, 1)
                x_tile = x_tile.transpose(0, 1).flatten(0, 1).to(self.device)

                x_mark_tile = batch_x_mark.repeat(repeat_n, 1, 1, 1)
                x_mark_tile = x_mark_tile.transpose(0, 1).flatten(0, 1).to(self.device)

                gen_y_box = []
                for _ in range(self.model.diffusion_config.testing.n_z_samples_depart):
                    for _ in range(self.model.diffusion_config.testing.n_z_samples_depart):
                        y_tile_seq = p_sample_loop(
                            self.model,
                            x_tile,
                            x_mark_tile,
                            residual_mean_tile,
                            residual_T_mean_tile,
                            self.model.num_timesteps,
                            self.model.alphas,
                            self.model.one_minus_alphas_bar_sqrt
                        )

                    gen_residual = store_gen_y_at_step_t(
                        config=self.model.args,
                        config_diff=self.model.diffusion_config,
                        idx=self.model.num_timesteps,
                        y_tile_seq=y_tile_seq
                    )
                    gen_y_box.append(gen_residual)

                # [B, n_samples, label+pred, C]
                outputs_residual = np.concatenate(gen_y_box, axis=1)

                f_dim = -1 if self.args.features == 'MS' else 0

                # crop future residual
                outputs_residual = outputs_residual[:, :, -self.args.pred_len:, f_dim:]

                # add future trend back
                trend_future_np = pred_trend_future[:, :, f_dim:].detach().cpu().numpy()
                outputs = outputs_residual + trend_future_np[:, None, :, :]

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                if i % 5 == 0 and i != 0:
                    print('Testing: %d/%d cost time: %f min' % (
                        i, len(test_loader), (time.time() - minibatch_sample_start) / 60))
                    minibatch_sample_start = time.time()

        preds = np.array(preds)
        trues = np.array(trues)

        preds_save = np.array(preds)
        trues_save = np.array(trues)

        preds_ns = np.array(preds).mean(axis=2)
        print('test shape:', preds_ns.shape, trues.shape)
        preds_ns = preds_ns.reshape(-1, preds_ns.shape[-2], preds_ns.shape[-1])
        trues_ns = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds_ns.shape, trues_ns.shape)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds_ns, trues_ns)
        print('Transition-TMDM metric: mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}'.format(
            mse, mae, rmse, mape, mspe))

        preds = preds.reshape(-1, preds.shape[-3], preds.shape[-2] * preds.shape[-1])
        preds = preds.transpose(0, 2, 1)
        preds = preds.reshape(-1, preds.shape[-1])

        trues = trues.reshape(-1, 1, trues.shape[-2] * trues.shape[-1])
        trues = trues.transpose(0, 2, 1)
        trues = trues.reshape(-1, trues.shape[-1])

        y_true_ratio_by_bin, qice_coverage_ratio, y_true = compute_true_coverage_by_gen_QI(
            config=self.model.diffusion_config,
            dataset_object=preds.shape[0],
            all_true_y=trues,
            all_generated_y=preds,
        )

        coverage, _, _ = compute_PICP(config=self.model.diffusion_config, y_true=y_true, all_gen_y=preds)

        print('CARD/TMDM metric: QICE:{:.4f}%, PICP:{:.4f}%'.format(qice_coverage_ratio * 100, coverage * 100))

        pred = preds_save.reshape(-1, preds_save.shape[-3], preds_save.shape[-2], preds_save.shape[-1])
        true = trues_save.reshape(-1, trues_save.shape[-2], trues_save.shape[-1])

        pool = Pool(processes=32)
        all_res = []
        for i in range(pred.shape[-1]):
            p_in = pred[:, :, :, i]
            p_in = p_in.transpose(0, 2, 1)
            p_in = p_in.reshape(-1, p_in.shape[-1])

            t_in = true[:, :, i]
            t_in = t_in.reshape(-1)
            all_res.append(pool.apply_async(ccc, args=(i, p_in, t_in)))

        p_in = np.sum(pred, axis=-1)
        p_in = p_in.transpose(0, 2, 1)
        p_in = p_in.reshape(-1, p_in.shape[-1])

        t_in = np.sum(true, axis=-1)
        t_in = t_in.reshape(-1)

        CRPS_sum = pool.apply_async(ccc, args=(8, p_in, t_in))

        pool.close()
        pool.join()

        all_res_get = []
        for i in range(len(all_res)):
            all_res_get.append(all_res[i].get())
        all_res_get = np.array(all_res_get)

        CRPS_0 = np.mean(all_res_get, axis=0).mean()
        CRPS_sum = CRPS_sum.get().mean()

        print('CRPS', CRPS_0, 'CRPS_sum', CRPS_sum)

        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy',
                np.array([mse, mae, rmse, mape, mspe, qice_coverage_ratio * 100, coverage * 100, CRPS_0, CRPS_sum]))
        np.save(folder_path + 'pred.npy', preds_save)
        np.save(folder_path + 'true.npy', trues_save)

        np.save("./results/{}.npy".format(self.args.model_id), np.array(mse))
        np.save("./results/{}_Ntimes.npy".format(self.args.model_id),
                np.array([mse, mae, rmse, mape, mspe, qice_coverage_ratio * 100, coverage * 100, CRPS_0, CRPS_sum]))

        return

    def predict(self, setting, load=False):
        """
        简化版 predict：
        使用 trend prediction + residual mean prediction，
        不走 diffusion 采样。
        如果你需要正式预测，建议复用 test() 里的采样流程。
        """
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint_diffusion.pth'), map_location=self.device))
            self.cond_pred_model.load_state_dict(torch.load(os.path.join(path, 'checkpoint_cond.pth'), map_location=self.device))
            self.trend_model.load_state_dict(torch.load(os.path.join(path, 'checkpoint_trend.pth'), map_location=self.device))

        preds = []

        self.model.eval()
        self.cond_pred_model.eval()
        self.trend_model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                x_trend, x_residual = series_decomp(batch_x, self.args.trend_kernel)
                pred_trend_future = self.trend_model(x_trend)

                dec_inp_res = build_residual_decoder_input(
                    x_residual=x_residual,
                    pred_len=self.args.pred_len,
                    label_len=self.args.label_len
                )

                _, residual_mean_full, _, _ = self.cond_pred_model(
                    x_residual, batch_x_mark, dec_inp_res, batch_y_mark
                )

                pred = pred_trend_future + residual_mean_full[:, -self.args.pred_len:, :]
                pred = pred.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        return
