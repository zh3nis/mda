# mda.py
# Drop-in MDA classification head for neural networks
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MDAHead(nn.Module):
    """
    Drop-in replacement for a Softmax (linear) classification head.
    Computes class logits as: log p(y) + log p(z|y),
    where p(z|y) is a class-conditional mixture of K diagonal Gaussians.
    Use with torch.nn.CrossEntropyLoss.
    """

    def __init__(self, latent_dim: int, num_classes: int, K: int = 2,
                 init_scale: float = 0.1):
        super().__init__()
        self.D = latent_dim
        self.C = num_classes
        self.K = K

        # Gaussian mixture parameters
        self.mu = nn.Parameter(torch.randn(self.C, self.K, self.D) * init_scale)
        self.logvar = nn.Parameter(torch.zeros(self.C, self.K, self.D))
        self.logits_pi = nn.Parameter(torch.zeros(self.C, self.K))  # per-class mixture weights

        # Class prior (initialized uniformly)
        self.register_buffer("log_prior", torch.log(torch.full((self.C,), 1.0 / self.C)))

    def _log_diag_gauss(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Compute log N(z|mu, diag(exp(logvar))) for each component.
        z: [B, D], mu/logvar: [C*K, D] → return [B, C*K]
        """
        B, D = z.shape
        const = -0.5 * D * math.log(2 * math.pi)
        diff = z.unsqueeze(1) - mu.unsqueeze(0)                    # [B, C*K, D]
        inv_var = torch.exp(-logvar).unsqueeze(0)                  # [1, C*K, D]
        quad = -0.5 * (diff * diff * inv_var).sum(dim=-1)          # [B, C*K]
        log_det = -0.5 * logvar.sum(dim=-1).unsqueeze(0)           # [1, C*K]
        return const + log_det + quad                              # [B, C*K]

    def forward(self, z: torch.Tensor):
        """
        Returns class logits suitable for CrossEntropyLoss.
        """
        B, D = z.shape
        assert D == self.D

        mu     = self.mu.view(self.C * self.K, self.D)
        logvar = self.logvar.view(self.C * self.K, self.D)

        log_norm = self._log_diag_gauss(z, mu, logvar).view(B, self.C, self.K)
        log_pi = F.log_softmax(self.logits_pi, dim=-1).unsqueeze(0)  # [1, C, K]
        log_p_z_given_c = torch.logsumexp(log_pi + log_norm, dim=-1)  # [B, C]

        log_prior = self.log_prior.view(1, self.C).expand(B, self.C)

        # numerical safety
        self.logvar.data.clamp_(min=-6.0, max=6.0)

        return log_prior + log_p_z_given_c  # class logits
