import math
import torch
import numpy as np


def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "jsd":
        betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [min(
                1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) /
                (math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2),
                max_beta
            ) for i in range(num_timesteps)]
        )
        if schedule == "cosine_reverse":
            betas = betas.flip(0)
    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [start + 0.5 * (end - start) * (1 - math.cos(t / (num_timesteps - 1) * math.pi))
             for t in range(num_timesteps)]
        )
    return betas


def extract(schedule_tensor, step_ids, reference_tensor):
    reference_shape = reference_tensor.shape
    extracted_value = torch.gather(schedule_tensor, 0, step_ids.to(schedule_tensor.device))
    reshape_shape = [step_ids.shape[0]] + [1] * (len(reference_shape) - 1)
    return extracted_value.reshape(*reshape_shape)


# =========================
# Forward process
# =========================
def q_sample(clean_target, prior_mean, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, step_ids, noise=None):
    """
    Forward diffusion:
        q(target_t | target_0, prior_mean)

    In the transition version:
        clean_target = clean residual target
        prior_mean   = residual prior mean
    """
    if noise is None:
        noise = torch.randn_like(clean_target).to(clean_target.device)

    sqrt_alpha_bar_t = extract(alphas_bar_sqrt, step_ids, clean_target)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, step_ids, clean_target)

    noisy_target = (
        sqrt_alpha_bar_t * clean_target
        + (1 - sqrt_alpha_bar_t) * prior_mean
        + sqrt_one_minus_alpha_bar_t * noise
    )
    return noisy_target


# =========================
# Reverse process: one step
# =========================
def p_sample(model, history_input, history_mark, noisy_target, prior_mean, prior_mean_T,
             step_index, alphas, one_minus_alphas_bar_sqrt):
    """
    Reverse diffusion sampling for one step.

    Args:
        noisy_target: [B, L, C], current noisy target at timestep t
        prior_mean:   [B, L, C], prior mean used as condition input to eps_theta
        prior_mean_T: [B, L, C], mean of p(target_T)
    """
    device = next(model.parameters()).device
    gaussian_noise = torch.randn_like(noisy_target)
    step_ids = torch.tensor([step_index]).to(device)

    alpha_t = extract(alphas, step_ids, noisy_target)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, step_ids, noisy_target)
    sqrt_one_minus_alpha_bar_t_prev = extract(one_minus_alphas_bar_sqrt, step_ids - 1, noisy_target)

    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    sqrt_alpha_bar_t_prev = (1 - sqrt_one_minus_alpha_bar_t_prev.square()).sqrt()

    posterior_coeff_clean = (1 - alpha_t) * sqrt_alpha_bar_t_prev / (sqrt_one_minus_alpha_bar_t.square())
    posterior_coeff_noisy = (sqrt_one_minus_alpha_bar_t_prev.square()) * (alpha_t.sqrt()) / (
        sqrt_one_minus_alpha_bar_t.square()
    )
    posterior_coeff_prior = 1 + (sqrt_alpha_bar_t - 1) * (alpha_t.sqrt() + sqrt_alpha_bar_t_prev) / (
        sqrt_one_minus_alpha_bar_t.square()
    )

    predicted_noise = model(
        history_input,
        history_mark,
        0,
        noisy_target,
        prior_mean,
        step_ids
    ).to(device).detach()

    # Reconstruct clean target
    reconstructed_clean_target = (
        1 / sqrt_alpha_bar_t
    ) * (
        noisy_target
        - (1 - sqrt_alpha_bar_t) * prior_mean_T
        - predicted_noise * sqrt_one_minus_alpha_bar_t
    )

    posterior_mean = (
        posterior_coeff_clean * reconstructed_clean_target
        + posterior_coeff_noisy * noisy_target
        + posterior_coeff_prior * prior_mean_T
    )

    posterior_variance = (
        (sqrt_one_minus_alpha_bar_t_prev.square()) / (sqrt_one_minus_alpha_bar_t.square()) * (1 - alpha_t)
    )

    sampled_previous_target = posterior_mean.to(device) + posterior_variance.sqrt().to(device) * gaussian_noise.to(device)
    return sampled_previous_target


# =========================
# Reverse process: t=1 -> 0
# =========================
def p_sample_t_1to0(model, history_input, history_mark, noisy_target, prior_mean, prior_mean_T,
                    one_minus_alphas_bar_sqrt):
    device = next(model.parameters()).device
    step_ids = torch.tensor([0]).to(device)

    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, step_ids, noisy_target)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()

    predicted_noise = model(
        history_input,
        history_mark,
        0,
        noisy_target,
        prior_mean,
        step_ids
    ).to(device).detach()

    reconstructed_clean_target = (
        1 / sqrt_alpha_bar_t
    ) * (
        noisy_target
        - (1 - sqrt_alpha_bar_t) * prior_mean_T
        - predicted_noise * sqrt_one_minus_alpha_bar_t
    )

    return reconstructed_clean_target.to(device)


# =========================
# Reverse process: full loop
# =========================
def p_sample_loop(model, history_input, history_mark, prior_mean, prior_mean_T,
                  num_steps, alphas, one_minus_alphas_bar_sqrt):
    device = next(model.parameters()).device

    terminal_noise = torch.randn_like(prior_mean_T).to(device)
    current_target = terminal_noise + prior_mean_T  # sample target_T
    target_sequence = [current_target]

    for step_index in reversed(range(1, num_steps)):
        noisy_target = current_target
        current_target = p_sample(
            model,
            history_input,
            history_mark,
            noisy_target,
            prior_mean,
            prior_mean_T,
            step_index,
            alphas,
            one_minus_alphas_bar_sqrt
        )
        target_sequence.append(current_target)

    assert len(target_sequence) == num_steps

    clean_target = p_sample_t_1to0(
        model,
        history_input,
        history_mark,
        target_sequence[-1],
        prior_mean,
        prior_mean_T,
        one_minus_alphas_bar_sqrt
    )
    target_sequence.append(clean_target)
    return target_sequence


# =========================
# Evaluation helper
# =========================
def kld(sample_a, sample_b, grid=(-20, 20), num_grid=400):
    sample_a, sample_b = sample_a.numpy().flatten(), sample_b.numpy().flatten()
    prob_a, _ = np.histogram(sample_a, bins=num_grid, range=[grid[0], grid[1]], density=True)
    prob_a += 1e-7
    prob_b, _ = np.histogram(sample_b, bins=num_grid, range=[grid[0], grid[1]], density=True)
    prob_b += 1e-7
    return (prob_a * np.log(prob_a / prob_b)).sum()
