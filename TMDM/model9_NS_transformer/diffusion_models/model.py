import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(t.size()[0], -1, self.num_out) * out
        return out


class ConditionalGuidedModel(nn.Module):
    """
    The diffusion denoiser in TMDM.
    过渡版中，这个网络仍然不变其功能本质：
    它仍然预测噪声 eps_theta，
    但现在是在 residual space 中工作。
    """

    def __init__(self, config, MTS_args):
        super(ConditionalGuidedModel, self).__init__()
        n_steps = config.diffusion.timesteps + 1
        self.cat_x = config.model.cat_x
        self.cat_y_pred = config.model.cat_y_pred

        self.c_out = MTS_args.c_out
        self.x_embed_dim = MTS_args.CART_input_x_embed_dim

        # 动态输入维度
        # 原版代码这里写死成了 14 / 7，这在改造版里很容易出错
        if self.cat_y_pred:
            # concat(y_t, y_0_hat)
            data_dim = self.c_out * 2
        elif self.cat_x:
            # concat(y_t, x_emb)
            data_dim = self.c_out + self.x_embed_dim
        else:
            data_dim = self.c_out

        hidden_dim = 128

        self.lin1 = ConditionalLinear(data_dim, hidden_dim, n_steps)
        self.lin2 = ConditionalLinear(hidden_dim, hidden_dim, n_steps)
        self.lin3 = ConditionalLinear(hidden_dim, hidden_dim, n_steps)
        self.lin4 = nn.Linear(hidden_dim, self.c_out)

    def forward(self, x, y_t, y_0_hat, t):
        """
        Args:
            x:      encoded history features, [B, L, x_embed_dim]
            y_t:    noisy target at timestep t, [B, L, C]
            y_0_hat:condition mean / prior mean, [B, L, C]
            t:      timestep ids, [B]
        """
        if self.cat_y_pred:
            eps_pred = torch.cat((y_t, y_0_hat), dim=-1)
        elif self.cat_x:
            eps_pred = torch.cat((y_t, x), dim=-1)
        else:
            eps_pred = y_t

        if y_t.device.type == 'mps':
            eps_pred = self.lin1(eps_pred, t)
            eps_pred = F.softplus(eps_pred.cpu()).to(y_t.device)

            eps_pred = self.lin2(eps_pred, t)
            eps_pred = F.softplus(eps_pred.cpu()).to(y_t.device)

            eps_pred = self.lin3(eps_pred, t)
            eps_pred = F.softplus(eps_pred.cpu()).to(y_t.device)
        else:
            eps_pred = F.softplus(self.lin1(eps_pred, t))
            eps_pred = F.softplus(self.lin2(eps_pred, t))
            eps_pred = F.softplus(self.lin3(eps_pred, t))

        eps_pred = self.lin4(eps_pred)
        return eps_pred


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
