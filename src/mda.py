import torch
import torch.nn as nn
import torch.nn.functional as F

class MDAHead(nn.Module):
    def __init__(self, d: int, num_classes: int, K: int = 2, init_scale: float = .1):
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

    # ----- supervised responsibilities over components of the TRUE class -----
    @torch.no_grad()
    def _resp_true_class(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logpdf = self._component_logpdf(z)                # [B, C, K]
        logpi  = F.log_softmax(self.logits_pi, dim=-1)    # [C, K]
        log_r  = logpdf + logpi.unsqueeze(0)              # [B, C, K]
        return F.softmax(log_r[torch.arange(z.size(0)), y], dim=-1)  # [B, K]

    # ----- EM step (one full pass over loader) -----
    @torch.no_grad()
    def em_update(self, encode, loader, device, var_floor: float = 1e-3, jitter: float = 1e-6, momentum: float = 0.2):
        C, K, D = self.C, self.K, self.D
        Nk  = torch.zeros(C, K, device=device)
        S1  = torch.zeros(C, K, D, device=device)
        S2  = torch.zeros(C, K, D, D, device=device)

        self.eval()
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = encode(X)                                 # [B, D]
            r = self._resp_true_class(z, y)               # [B, K]

            for c in range(C):
                m = (y == c)
                if not m.any():
                    continue
                zc = z[m]                                 # [Bc, D]
                rc = r[m]                                 # [Bc, K]
                Nk[c] += rc.sum(0)                        # [K]
                S1[c] += rc.T @ zc                        # [K,D]
                S2[c] += torch.einsum('bk,bd,be->kde', rc, zc, zc)  # [K,D,D]

        Nk_safe = Nk.clamp_min(1e-8)
        mu_new  = S1 / Nk_safe.unsqueeze(-1)                         # [C,K,D]
        Ezz     = S2 / Nk_safe.unsqueeze(-1).unsqueeze(-1)           # [C,K,D,D]
        Sigma   = Ezz - torch.einsum('ckd,cke->ckde', mu_new, mu_new)# [C,K,D,D]

        # regularize & cholesky
        eye = torch.eye(D, device=device).expand(C, K, D, D)
        Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2)) + var_floor * eye
        Sigma = Sigma + jitter * eye
        L_new = torch.linalg.cholesky(Sigma)                         # [C,K,D,D]

        pi_new = (Nk_safe / Nk_safe.sum(-1, keepdim=True)).clamp_min(1e-8)
        logits_pi_new = pi_new.log()

        # EMA updates
        self.mu.data        = momentum * self.mu.data        + (1 - momentum) * mu_new
        self.covL.data      = momentum * self.covL.data      + (1 - momentum) * L_new
        self.logits_pi.data = momentum * self.logits_pi.data + (1 - momentum) * logits_pi_new
