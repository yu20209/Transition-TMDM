import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalLinear(nn.Module):
    def __init__(self, input_dim, output_dim, num_steps):
        super(ConditionalLinear, self).__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.step_embedding = nn.Embedding(num_steps, output_dim)
        self.step_embedding.weight.data.uniform_()

    def forward(self, input_tensor, step_ids):
        linear_out = self.linear(input_tensor)
        step_scale = self.step_embedding(step_ids)
        conditioned_out = step_scale.view(step_ids.size()[0], -1, self.output_dim) * linear_out
        return conditioned_out


class ConditionalGuidedModel(nn.Module):
    """
    Denoiser network in residual space.
    It predicts epsilon_theta for noisy residual targets.
    """

    def __init__(self, config, mts_args):
        super(ConditionalGuidedModel, self).__init__()
        num_steps = config.diffusion.timesteps + 1
        self.cat_x = config.model.cat_x
        self.cat_prior_mean = config.model.cat_y_pred

        self.output_dim = mts_args.c_out
        self.history_feature_dim = mts_args.CART_input_x_embed_dim

        if self.cat_prior_mean:
            # concat(noisy_target, prior_mean)
            input_dim = self.output_dim * 2
        elif self.cat_x:
            # concat(noisy_target, history_feature)
            input_dim = self.output_dim + self.history_feature_dim
        else:
            input_dim = self.output_dim

        hidden_dim = 128

        self.layer1 = ConditionalLinear(input_dim, hidden_dim, num_steps)
        self.layer2 = ConditionalLinear(hidden_dim, hidden_dim, num_steps)
        self.layer3 = ConditionalLinear(hidden_dim, hidden_dim, num_steps)
        self.output_layer = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, history_feature, noisy_target, prior_mean, step_ids):
        """
        Args:
            history_feature: [B, L, history_feature_dim]
            noisy_target:    [B, L, C]
            prior_mean:      [B, L, C]
            step_ids:        [B]
        """
        if self.cat_prior_mean:
            denoiser_input = torch.cat((noisy_target, prior_mean), dim=-1)
        elif self.cat_x:
            denoiser_input = torch.cat((noisy_target, history_feature), dim=-1)
        else:
            denoiser_input = noisy_target

        if noisy_target.device.type == 'mps':
            denoiser_hidden = self.layer1(denoiser_input, step_ids)
            denoiser_hidden = F.softplus(denoiser_hidden.cpu()).to(noisy_target.device)

            denoiser_hidden = self.layer2(denoiser_hidden, step_ids)
            denoiser_hidden = F.softplus(denoiser_hidden.cpu()).to(noisy_target.device)

            denoiser_hidden = self.layer3(denoiser_hidden, step_ids)
            denoiser_hidden = F.softplus(denoiser_hidden.cpu()).to(noisy_target.device)
        else:
            denoiser_hidden = F.softplus(self.layer1(denoiser_input, step_ids))
            denoiser_hidden = F.softplus(self.layer2(denoiser_hidden, step_ids))
            denoiser_hidden = F.softplus(self.layer3(denoiser_hidden, step_ids))

        predicted_noise = self.output_layer(denoiser_hidden)
        return predicted_noise


class DeterministicFeedForwardNeuralNetwork(nn.Module):

    def __init__(self, dim_in, dim_out, hid_layers,
                 use_batchnorm=False, negative_slope=0.01, dropout_rate=0):
        super(DeterministicFeedForwardNeuralNetwork, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hid_layers = hid_layers
        self.nn_layers = [self.dim_in] + self.hid_layers
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope
        self.dropout_rate = dropout_rate
        layers = self.create_nn_layers()
        self.network = nn.Sequential(*layers)

    def create_nn_layers(self):
        layers = []
        for idx in range(len(self.nn_layers) - 1):
            layers.append(nn.Linear(self.nn_layers[idx], self.nn_layers[idx + 1]))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(self.nn_layers[idx + 1]))
            layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
            layers.append(nn.Dropout(p=self.dropout_rate))
        layers.append(nn.Linear(self.nn_layers[-1], self.dim_out))
        return layers

    def forward(self, x):
        return self.network(x)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False

    def __call__(self, val_cost, epoch, verbose=False):
        score = val_cost

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch + 1
        elif score > self.best_score - self.delta:
            self.counter += 1
            if verbose:
                print("EarlyStopping counter: {} out of {}...".format(
                    self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch + 1
            self.counter = 0
