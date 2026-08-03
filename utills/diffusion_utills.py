import torch

TIMESTEPS = 1000

def get_diffusion_schedules(device):
    betas = torch.linspace(1e-4, 0.02, TIMESTEPS, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    sqrt_ab = torch.sqrt(alpha_bars)
    sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bars)
    return betas, alphas, alpha_bars, sqrt_ab, sqrt_one_minus_ab


def forward_diffusion(x0, t, sqrt_ab, sqrt_one_minus_ab):
    noise = torch.randn_like(x0)
    xt = (sqrt_ab[t, None, None, None] * x0 + sqrt_one_minus_ab[t, None, None, None] * noise)
    return xt, noise


@torch.no_grad()
def ddim_sample(model, style_img, text, shape, device, ddim_steps=50, eta=0.0):
    """Fast DDIM sampling procedure for inference."""
    model.eval()
    b = shape[0]

    betas, alphas, alpha_bars, _, _ = get_diffusion_schedules(device)
    timesteps = torch.linspace(TIMESTEPS - 1, 0, ddim_steps, dtype=torch.long, device=device)
    
    latent = torch.randn(shape, device=device)

    ver_map, hor_map, _, _, _ = model.style_encoder(style_img)
    Q = model.content_encoder(text)
    cond = model.blender(Q, ver_map, hor_map)

    for i in range(len(timesteps)):
        t = timesteps[i]
        prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(-1, device=device)

        time_tensor = torch.full((b,), t, device=device, dtype=torch.long)
        time_emb = model.time_embedding(time_tensor)

        pred_noise = model.unet(latent, cond, time_emb)
        pred_noise = model.final(pred_noise)

        alpha_bar_t = alpha_bars[t]
        alpha_bar_prev = alpha_bars[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=device)

        pred_x0 = (latent - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
        dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * pred_noise
        
        noise = torch.randn_like(latent) if sigma > 0 else 0
        latent = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma * noise

    return latent
