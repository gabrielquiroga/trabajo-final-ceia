import torch

# =====================================================================
# 1. El Planificador de Ruido (DDPM Scheduler)
# =====================================================================
class DDPMScheduler:
    def __init__(self, num_timesteps=100, beta_start=1e-4, beta_end=0.02):
        self.num_timesteps = num_timesteps
        # Betas controlan la varianza del ruido en cada paso
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original_samples, noise, timesteps):
        """
        Inyecta ruido a las muestras originales según el paso de tiempo t.
        Fórmula matemática de DDPM para salto directo al paso t.
        """
        device = original_samples.device
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        
        # Extraer los coeficientes para los timesteps dados
        sqrt_alpha_prod = torch.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus_alpha_prod = torch.sqrt(1 - self.alphas_cumprod[timesteps])

        # Acomodar dimensiones para poder multiplicar con el tensor de nodos [N, 2]
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * epsilon
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples