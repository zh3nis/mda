import torch
import torch.nn as nn
import torch.nn.functional as F

class MDAHead(nn.Module):
    def __init__(self, d: int, num_classes: int, K: int = 2, init_scale: float = 1.0):
        super().__init__()
        self.C, self.K, self.D = num_classes, K, d
        self.mu        = nn.Parameter(torch.randn(self.C, self.K, self.D) * init_scale)
        self.logits_pi = nn.Parameter(torch.zeros(self.C, self.K))
        # Cholesky factor L of full covariance Σ = L L^T
        L = torch.eye(d).repeat(self.C, self.K, 1, 1)
        self.covL      = nn.Parameter(L)       # lower-triangular
        self.register_buffer("logits_prior", torch.zeros(self.C))

    # ----- log N(z | μ, Σ) via Cholesky; returns [B, C, K] -----
    def _component_logpdf(self, z: torch.Tensor) -> torch.Tensor:
        B, D = z.shape
        CK = self.C * self.K
        mu  = self.mu.view(CK, D)                  # [CK,D]
        L   = self.covL.view(CK, D, D)             # [CK,D,D]

        diff = z.unsqueeze(1) - mu.unsqueeze(0)    # [B, CK, D]
        diff_t = diff.permute(1, 0, 2)             # [CK, B, D]

        # Solve L y = diff^T -> y = L^{-1} diff^T
        y = torch.linalg.solve_triangular(L, diff_t.transpose(1, 2), upper=False)  # [CK, D, B]
        maha = (y * y).sum(dim=1).transpose(0, 1)  # [B, CK]

        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)  # [CK]
        const = D * torch.log(torch.tensor(2.0 * torch.pi, device=z.device, dtype=z.dtype))
        logpdf = -0.5 * (maha + logdet.unsqueeze(0) + const)                    # [B, CK]
        return logpdf.view(B, self.C, self.K)

    # ----- drop-in: return class logits [B, C] -----
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logpdf = self._component_logpdf(z)                # [B, C, K]
        logpi  = F.log_softmax(self.logits_pi, dim=-1)    # [C, K]
        log_pz_c = torch.logsumexp(logpdf + logpi.unsqueeze(0), dim=-1)  # [B,C]
        return self.logits_prior.unsqueeze(0) + log_pz_c
    
    @torch.no_grad()
    def mu_hat(self, z: torch.Tensor):
        logpdf = self._component_logpdf(z)
        logpi  = F.log_softmax(self.logits_pi, -1)
        gamma  = (logpdf + logpi.unsqueeze(0))
        gamma  = (gamma - torch.logsumexp(gamma, dim=(1,2), keepdim=True)).exp()  # [B,C,K]
        Nk = gamma.sum(0).clamp_min(1e-4)                                         # [C,K]
        mu_hat = (gamma.permute(1,2,0) @ z) / Nk[...,None] 
        return mu_hat