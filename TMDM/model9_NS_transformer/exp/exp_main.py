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


def ccc(worker_id, generated_samples, ground_truth):
    result_box = np.zeros(len(ground_truth))
    for idx in range(len(ground_truth)):
        result = pscore(generated_samples[idx], ground_truth[idx]).compute()
        result_box[idx] = result[0]
    return result_box


def log_normal(target_tensor, mean_tensor, variance_tensor):
    """
    Negative log-likelihood under a diagonal Gaussian.
    """
    eps = 1e-8
    if isinstance(variance_tensor, float):
        variance_tensor = torch.tensor(
            variance_tensor,
            device=target_tensor.device,
            dtype=target_tensor.dtype
        )

    if eps > 0.0:
        variance_tensor = variance_tensor + eps

    return 0.5 * torch.mean(
        np.log(2.0 * np.pi)
        + torch.log(variance_tensor)
        + torch.pow(target_tensor - mean_tensor, 2) / variance_tensor
    )


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        residual_diffusion_model = diffuMTS.Model(self.args, self.device).float()
        residual_prior_model = ns_Transformer.Model(self.args).float()
        trend_model = TrendLinear(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            residual_diffusion_model = nn.DataParallel(residual_diffusion_model, device_ids=self.args.device_ids)
            residual_prior_model = nn.DataParallel(residual_prior_model, device_ids=self.args.device_ids)
            trend_model = nn.DataParallel(trend_model, device_ids=self.args.device_ids)

        return residual_diffusion_model, residual_prior_model, trend_model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
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
        return nn.MSELoss()

    def _prepare_transition_batch(self, history_input, full_target):
        """
        Prepare transition-version training targets.

        Returns:
            history_trend
            history_residual
            future_trend_pred
            full_trend_context
            target_residual
            future_trend_target
        """
        history_trend, history_residual = series_decomp(history_input, self.args.trend_kernel)

        full_target_trend = moving_average_trend(full_target, self.args.trend_kernel)
        future_trend_target = full_target_trend[:, -self.args.pred_len:, :]

        future_trend_pred = self.trend_model(history_trend)

        full_trend_context = build_future_trend_context(
            history_trend=history_trend,
            future_trend_pred=future_trend_pred,
            label_len=self.args.label_len
        )

        target_residual = full_target - full_trend_context

        return (
            history_trend,
            history_residual,
            future_trend_pred,
            full_trend_context,
            target_residual,
            future_trend_target
        )

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        self.cond_pred_model.eval()
        self.trend_model.eval()

        with torch.no_grad():
            for batch_idx, (history_input, full_target, history_mark, target_mark) in enumerate(vali_loader):
                history_input = history_input.float().to(self.device)
                full_target = full_target.float().to(self.device)

                history_mark = history_mark.float().to(self.device)
                target_mark = target_mark.float().to(self.device)

                (
                    history_trend,
                    history_residual,
                    future_trend_pred,
                    full_trend_context,
                    target_residual,
                    future_trend_target
                ) = self._prepare_transition_batch(history_input, full_target)

                residual_decoder_input = build_residual_decoder_input(
                    history_residual=history_residual,
                    pred_len=self.args.pred_len,
                    label_len=self.args.label_len
                )

                batch_size = history_input.size(0)
                step_ids = torch.randint(
                    low=0, high=self.model.num_timesteps, size=(batch_size // 2 + 1,)
                ).to(self.device)
                step_ids = torch.cat([step_ids, self.model.num_timesteps - 1 - step_ids], dim=0)[:batch_size]

                _, residual_prior_mean, kl_loss, latent_sample = self.cond_pred_model(
                    history_residual, history_mark, residual_decoder_input, target_mark
                )

                residual_prior_loss = log_normal(
                    target_residual,
                    residual_prior_mean,
                    torch.tensor(1.0, device=target_residual.device, dtype=target_residual.dtype)
                )
                residual_prior_loss_all = residual_prior_loss + self.args.k_z * kl_loss

                trend_loss = criterion(future_trend_pred, future_trend_target)

                residual_prior_mean_T = residual_prior_mean
                injected_noise = torch.randn_like(target_residual).to(self.device)

                noisy_residual_target = q_sample(
                    clean_target=target_residual,
                    prior_mean=residual_prior_mean_T,
                    alphas_bar_sqrt=self.model.alphas_bar_sqrt,
                    one_minus_alphas_bar_sqrt=self.model.one_minus_alphas_bar_sqrt,
                    step_ids=step_ids,
                    noise=injected_noise
                )

                predicted_noise = self.model(
                    history_residual,
                    history_mark,
                    target_residual,
                    noisy_residual_target,
                    residual_prior_mean,
                    step_ids
                )

                diffusion_loss = (injected_noise - predicted_noise).square().mean()

                total_batch_loss = (
                    diffusion_loss
                    + self.args.k_cond * residual_prior_loss_all
                    + self.args.k_trend * trend_loss
                )
                total_loss.append(total_batch_loss.detach().cpu().item())

        avg_total_loss = np.average(total_loss)
        self.model.train()
        self.cond_pred_model.train()
        self.trend_model.train()
        return avg_total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        checkpoint_dir = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

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
            epoch_train_loss = []

            self.model.train()
            self.cond_pred_model.train()
            self.trend_model.train()

            for batch_idx, (history_input, full_target, history_mark, target_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                history_input = history_input.float().to(self.device)
                full_target = full_target.float().to(self.device)
                history_mark = history_mark.float().to(self.device)
                target_mark = target_mark.float().to(self.device)

                (
                    history_trend,
                    history_residual,
                    future_trend_pred,
                    full_trend_context,
                    target_residual,
                    future_trend_target
                ) = self._prepare_transition_batch(history_input, full_target)

                residual_decoder_input = build_residual_decoder_input(
                    history_residual=history_residual,
                    pred_len=self.args.pred_len,
                    label_len=self.args.label_len
                )

                batch_size = history_input.size(0)
                step_ids = torch.randint(
                    low=0, high=self.model.num_timesteps, size=(batch_size // 2 + 1,)
                ).to(self.device)
                step_ids = torch.cat([step_ids, self.model.num_timesteps - 1 - step_ids], dim=0)[:batch_size]

                _, residual_prior_mean, kl_loss, latent_sample = self.cond_pred_model(
                    history_residual, history_mark, residual_decoder_input, target_mark
                )

                residual_prior_loss = log_normal(
                    target_residual,
                    residual_prior_mean,
                    torch.tensor(1.0, device=target_residual.device, dtype=target_residual.dtype)
                )
                residual_prior_loss_all = residual_prior_loss + self.args.k_z * kl_loss

                trend_loss = criterion(future_trend_pred, future_trend_target)

                residual_prior_mean_T = residual_prior_mean
                injected_noise = torch.randn_like(target_residual).to(self.device)

                noisy_residual_target = q_sample(
                    clean_target=target_residual,
                    prior_mean=residual_prior_mean_T,
                    alphas_bar_sqrt=self.model.alphas_bar_sqrt,
                    one_minus_alphas_bar_sqrt=self.model.one_minus_alphas_bar_sqrt,
                    step_ids=step_ids,
                    noise=injected_noise
                )

                predicted_noise = self.model(
                    history_residual,
                    history_mark,
                    target_residual,
                    noisy_residual_target,
                    residual_prior_mean,
                    step_ids
                )

                diffusion_loss = (injected_noise - predicted_noise).square().mean()

                total_batch_loss = (
                    diffusion_loss
                    + self.args.k_cond * residual_prior_loss_all
                    + self.args.k_trend * trend_loss
                )
                epoch_train_loss.append(total_batch_loss.item())

                if (batch_idx + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        batch_idx + 1, epoch + 1, total_batch_loss.item()
                    ))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - batch_idx)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(total_batch_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    total_batch_loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            avg_train_loss = np.average(epoch_train_loss)
            avg_vali_loss = self.vali(vali_data, vali_loader, criterion)
            avg_test_loss = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, avg_train_loss, avg_vali_loss, avg_test_loss
                )
            )

            if avg_vali_loss <= getattr(self, "best_val_loss", 1e18):
                self.best_val_loss = avg_vali_loss
                torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, 'checkpoint_diffusion.pth'))
                torch.save(self.cond_pred_model.state_dict(), os.path.join(checkpoint_dir, 'checkpoint_cond.pth'))
                torch.save(self.trend_model.state_dict(), os.path.join(checkpoint_dir, 'checkpoint_trend.pth'))
                print("Validation improved. Saved diffusion / residual prior / trend checkpoints.")

            early_stopping(avg_vali_loss, self.model, checkpoint_dir)

            if math.isnan(avg_train_loss):
                break

            if early_stopping.early_stop:
                print("Early stopping")
                break

        diffusion_ckpt = os.path.join(checkpoint_dir, 'checkpoint_diffusion.pth')
        cond_ckpt = os.path.join(checkpoint_dir, 'checkpoint_cond.pth')
        trend_ckpt = os.path.join(checkpoint_dir, 'checkpoint_trend.pth')

        if os.path.exists(diffusion_ckpt):
            self.model.load_state_dict(torch.load(diffusion_ckpt, map_location=self.device))
        if os.path.exists(cond_ckpt):
            self.cond_pred_model.load_state_dict(torch.load(cond_ckpt, map_location=self.device))
        if os.path.exists(trend_ckpt):
            self.trend_model.load_state_dict(torch.load(trend_ckpt, map_location=self.device))

        return self.model

    def test(self, setting, test=0):
        def store_generated_target_at_step(config, diffusion_config, reverse_index, generated_target_sequence):
            current_step = self.model.num_timesteps - reverse_index
            generated_target = generated_target_sequence[reverse_index].reshape(
                config.test_batch_size,
                int(diffusion_config.testing.n_z_samples / diffusion_config.testing.n_z_samples_depart),
                (config.label_len + config.pred_len),
                config.c_out
            ).cpu().numpy()

            if len(generated_target_by_batch[current_step]) == 0:
                generated_target_by_batch[current_step] = generated_target
            else:
                generated_target_by_batch[current_step] = np.concatenate(
                    [generated_target_by_batch[current_step], generated_target], axis=0
                )
            return generated_target

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
            assert np.abs(np.sum(y_true_ratio_by_bin) - 1) < 1e-10
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
            checkpoint_dir = os.path.join('./checkpoints/', setting)
            self.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'checkpoint_diffusion.pth'), map_location=self.device))
            self.cond_pred_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'checkpoint_cond.pth'), map_location=self.device))
            self.trend_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'checkpoint_trend.pth'), map_location=self.device))

        all_preds = []
        all_trues = []

        test_result_dir = './test_results/' + setting + '/'
        if not os.path.exists(test_result_dir):
            os.makedirs(test_result_dir)

        minibatch_sample_start = time.time()

        self.model.eval()
        self.cond_pred_model.eval()
        self.trend_model.eval()

        with torch.no_grad():
            for batch_idx, (history_input, full_target, history_mark, target_mark) in enumerate(test_loader):
                generated_target_by_batch = [[] for _ in range(self.model.num_timesteps + 1)]

                history_input = history_input.float().to(self.device)
                full_target = full_target.float().to(self.device)

                history_mark = history_mark.float().to(self.device)
                target_mark = target_mark.float().to(self.device)

                history_trend, history_residual = series_decomp(history_input, self.args.trend_kernel)

                future_trend_pred = self.trend_model(history_trend)
                full_trend_context = build_future_trend_context(
                    history_trend=history_trend,
                    future_trend_pred=future_trend_pred,
                    label_len=self.args.label_len
                )

                residual_decoder_input = build_residual_decoder_input(
                    history_residual=history_residual,
                    pred_len=self.args.pred_len,
                    label_len=self.args.label_len
                )

                _, residual_prior_mean, _, latent_sample = self.cond_pred_model(
                    history_residual, history_mark, residual_decoder_input, target_mark
                )

                repeat_n = int(
                    self.model.diffusion_config.testing.n_z_samples /
                    self.model.diffusion_config.testing.n_z_samples_depart
                )

                tiled_residual_prior_mean = residual_prior_mean.repeat(repeat_n, 1, 1, 1)
                tiled_residual_prior_mean = tiled_residual_prior_mean.transpose(0, 1).flatten(0, 1).to(self.device)

                tiled_residual_prior_mean_T = tiled_residual_prior_mean

                tiled_history_residual = history_residual.repeat(repeat_n, 1, 1, 1)
                tiled_history_residual = tiled_history_residual.transpose(0, 1).flatten(0, 1).to(self.device)

                tiled_history_mark = history_mark.repeat(repeat_n, 1, 1, 1)
                tiled_history_mark = tiled_history_mark.transpose(0, 1).flatten(0, 1).to(self.device)

                generated_residual_box = []
                for _ in range(self.model.diffusion_config.testing.n_z_samples_depart):
                    for _ in range(self.model.diffusion_config.testing.n_z_samples_depart):
                        generated_residual_sequence = p_sample_loop(
                            self.model,
                            tiled_history_residual,
                            tiled_history_mark,
                            tiled_residual_prior_mean,
                            tiled_residual_prior_mean_T,
                            self.model.num_timesteps,
                            self.model.alphas,
                            self.model.one_minus_alphas_bar_sqrt
                        )

                    generated_residual = store_generated_target_at_step(
                        config=self.model.args,
                        diffusion_config=self.model.diffusion_config,
                        reverse_index=self.model.num_timesteps,
                        generated_target_sequence=generated_residual_sequence
                    )
                    generated_residual_box.append(generated_residual)

                generated_residual = np.concatenate(generated_residual_box, axis=1)

                feature_dim = -1 if self.args.features == 'MS' else 0

                generated_residual = generated_residual[:, :, -self.args.pred_len:, feature_dim:]
                future_trend_np = future_trend_pred[:, :, feature_dim:].detach().cpu().numpy()

                # final prediction = trend prediction + generated residual
                final_prediction = generated_residual + future_trend_np[:, None, :, :]

                full_target = full_target[:, -self.args.pred_len:, feature_dim:].to(self.device)
                full_target = full_target.detach().cpu().numpy()

                all_preds.append(final_prediction)
                all_trues.append(full_target)

                if batch_idx % 5 == 0 and batch_idx != 0:
                    print('Testing: %d/%d cost time: %f min' % (
                        batch_idx, len(test_loader), (time.time() - minibatch_sample_start) / 60))
                    minibatch_sample_start = time.time()

        all_preds = np.array(all_preds)
        all_trues = np.array(all_trues)

        all_preds_save = np.array(all_preds)
        all_trues_save = np.array(all_trues)

        point_prediction = np.array(all_preds).mean(axis=2)
        print('test shape:', point_prediction.shape, all_trues.shape)

        point_prediction = point_prediction.reshape(-1, point_prediction.shape[-2], point_prediction.shape[-1])
        all_trues_point = all_trues.reshape(-1, all_trues.shape[-2], all_trues.shape[-1])
        print('test shape:', point_prediction.shape, all_trues_point.shape)

        result_dir = './results/' + setting + '/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        mae, mse, rmse, mape, mspe = metric(point_prediction, all_trues_point)
        print('Transition-TMDM metric: mse:{:.4f}, mae:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}'.format(
            mse, mae, rmse, mape, mspe))

        flattened_preds = all_preds.reshape(-1, all_preds.shape[-3], all_preds.shape[-2] * all_preds.shape[-1])
        flattened_preds = flattened_preds.transpose(0, 2, 1)
        flattened_preds = flattened_preds.reshape(-1, flattened_preds.shape[-1])

        flattened_trues = all_trues.reshape(-1, 1, all_trues.shape[-2] * all_trues.shape[-1])
        flattened_trues = flattened_trues.transpose(0, 2, 1)
        flattened_trues = flattened_trues.reshape(-1, flattened_trues.shape[-1])

        y_true_ratio_by_bin, qice_coverage_ratio, y_true = compute_true_coverage_by_gen_QI(
            config=self.model.diffusion_config,
            dataset_object=flattened_preds.shape[0],
            all_true_y=flattened_trues,
            all_generated_y=flattened_preds,
        )

        coverage, _, _ = compute_PICP(
            config=self.model.diffusion_config,
            y_true=y_true,
            all_gen_y=flattened_preds
        )

        print('CARD/TMDM metric: QICE:{:.4f}%, PICP:{:.4f}%'.format(qice_coverage_ratio * 100, coverage * 100))

        pred_for_crps = all_preds_save.reshape(-1, all_preds_save.shape[-3], all_preds_save.shape[-2], all_preds_save.shape[-1])
        true_for_crps = all_trues_save.reshape(-1, all_trues_save.shape[-2], all_trues_save.shape[-1])

        pool = Pool(processes=32)
        crps_worker_results = []
        for feature_idx in range(pred_for_crps.shape[-1]):
            generated_feature = pred_for_crps[:, :, :, feature_idx]
            generated_feature = generated_feature.transpose(0, 2, 1)
            generated_feature = generated_feature.reshape(-1, generated_feature.shape[-1])

            true_feature = true_for_crps[:, :, feature_idx]
            true_feature = true_feature.reshape(-1)
            crps_worker_results.append(pool.apply_async(ccc, args=(feature_idx, generated_feature, true_feature)))

        generated_sum = np.sum(pred_for_crps, axis=-1)
        generated_sum = generated_sum.transpose(0, 2, 1)
        generated_sum = generated_sum.reshape(-1, generated_sum.shape[-1])

        true_sum = np.sum(true_for_crps, axis=-1)
        true_sum = true_sum.reshape(-1)

        crps_sum_async = pool.apply_async(ccc, args=(8, generated_sum, true_sum))

        pool.close()
        pool.join()

        crps_feature_values = []
        for result_idx in range(len(crps_worker_results)):
            crps_feature_values.append(crps_worker_results[result_idx].get())
        crps_feature_values = np.array(crps_feature_values)

        crps_mean = np.mean(crps_feature_values, axis=0).mean()
        crps_sum = crps_sum_async.get().mean()

        print('CRPS', crps_mean, 'CRPS_sum', crps_sum)

        with open("result.txt", 'a') as result_file:
            result_file.write(setting + "  \n")
            result_file.write('mse:{}, mae:{}'.format(mse, mae))
            result_file.write('\n\n')

        np.save(result_dir + 'metrics.npy',
                np.array([mse, mae, rmse, mape, mspe, qice_coverage_ratio * 100, coverage * 100, crps_mean, crps_sum]))
        np.save(result_dir + 'pred.npy', all_preds_save)
        np.save(result_dir + 'true.npy', all_trues_save)

        np.save("./results/{}.npy".format(self.args.model_id), np.array(mse))
        np.save("./results/{}_Ntimes.npy".format(self.args.model_id),
                np.array([mse, mae, rmse, mape, mspe, qice_coverage_ratio * 100, coverage * 100, crps_mean, crps_sum]))

        return

    def predict(self, setting, load=False):
        """
        Simplified predict:
        future prediction = future trend prediction + residual prior mean.
        """
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            checkpoint_dir = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'checkpoint_diffusion.pth'), map_location=self.device))
            self.cond_pred_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'checkpoint_cond.pth'), map_location=self.device))
            self.trend_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'checkpoint_trend.pth'), map_location=self.device))

        all_preds = []

        self.model.eval()
        self.cond_pred_model.eval()
        self.trend_model.eval()

        with torch.no_grad():
            for batch_idx, (history_input, full_target, history_mark, target_mark) in enumerate(pred_loader):
                history_input = history_input.float().to(self.device)
                history_mark = history_mark.float().to(self.device)
                target_mark = target_mark.float().to(self.device)

                history_trend, history_residual = series_decomp(history_input, self.args.trend_kernel)
                future_trend_pred = self.trend_model(history_trend)

                residual_decoder_input = build_residual_decoder_input(
                    history_residual=history_residual,
                    pred_len=self.args.pred_len,
                    label_len=self.args.label_len
                )

                _, residual_prior_mean, _, _ = self.cond_pred_model(
                    history_residual, history_mark, residual_decoder_input, target_mark
                )

                future_prediction = future_trend_pred + residual_prior_mean[:, -self.args.pred_len:, :]
                future_prediction = future_prediction.detach().cpu().numpy()
                all_preds.append(future_prediction)

        all_preds = np.array(all_preds)
        all_preds = all_preds.reshape(-1, all_preds.shape[-2], all_preds.shape[-1])

        result_dir = './results/' + setting + '/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        np.save(result_dir + 'real_prediction.npy', all_preds)
        return
