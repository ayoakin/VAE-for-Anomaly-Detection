"""
Disentangled VAE Variants for Anomaly Detection.

This module implements three VAE variants designed to learn more
interpretable, disentangled latent representations:

1. β-VAE: Increases KL weight to encourage independent dimensions
2. FactorVAE: Uses adversarial training to penalize correlations
3. β-TCVAE: Decomposes KL and only penalizes total correlation

These variants may enable "latent traversal" for root cause analysis:
instead of looking at per-feature reconstruction error, we could
identify which latent DIMENSION changed and map it to a physical cause.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
import numpy as np


class BaseVAE(nn.Module):
    """
    Base VAE class with shared architecture.
    All variants inherit from this.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 16,
        dropout: float = 0.1
    ):
        super(BaseVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        prev_dim = latent_dim
        for h_dim in hidden_dims_reversed:
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims_reversed[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            recon, _, _ = self.forward(x)
            error = torch.mean((x - recon) ** 2, dim=1)
        return error


class BetaVAE(BaseVAE):
    """
    β-VAE: VAE with increased KL divergence weight.
    
    Loss = Reconstruction + β × KL Divergence
    
    Higher β encourages the model to use latent dimensions more
    independently, leading to better disentanglement at the cost
    of reconstruction quality.
    
    Reference: Higgins et al., "β-VAE: Learning Basic Visual Concepts
    with a Constrained Variational Framework" (ICLR 2017)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 16,
        beta: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__(input_dim, hidden_dims, latent_dim, dropout)
        self.beta = beta
        self.name = f"β-VAE (β={beta})"
    
    def loss_function(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        β-VAE loss with configurable β weight.
        """
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


class FactorVAE(BaseVAE):
    """
    FactorVAE: VAE with adversarial total correlation penalty.
    
    Uses a discriminator to distinguish between:
    - q(z): actual joint latent distribution (may be correlated)
    - q̄(z): product of marginals (independent by construction)
    
    The discriminator output provides a density ratio estimate
    used to penalize total correlation.
    
    Reference: Kim & Mnih, "Disentangling by Factorising" (ICML 2018)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 16,
        gamma: float = 10.0,
        dropout: float = 0.1
    ):
        super().__init__(input_dim, hidden_dims, latent_dim, dropout)
        self.gamma = gamma
        self.name = f"FactorVAE (γ={gamma})"
        
        # Discriminator: classifies z as "real" (from q(z)) or "permuted" (from q̄(z))
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    
    def permute_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Permute each latent dimension independently across the batch.
        This breaks correlations, creating samples from q̄(z) = Π_j q(z_j).
        """
        B, D = z.size()
        z_permuted = z.clone()
        for d in range(D):
            perm = torch.randperm(B, device=z.device)
            z_permuted[:, d] = z[perm, d]
        return z_permuted
    
    def loss_function(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        FactorVAE loss with total correlation penalty.
        """
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total correlation term via discriminator
        if z is not None:
            # D(z) estimates log(q(z) / q̄(z))
            d_z = self.discriminator(z)
            tc_loss = (d_z[:, 0] - torch.log(1 - torch.sigmoid(d_z[:, 0]) + 1e-10)).mean()
        else:
            tc_loss = torch.tensor(0.0, device=x.device)
        
        total_loss = recon_loss + kl_loss + self.gamma * tc_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'tc_loss': tc_loss
        }
    
    def discriminator_loss(self, z_real: torch.Tensor) -> torch.Tensor:
        """
        Train discriminator to distinguish real z from permuted z.
        """
        z_permuted = self.permute_latent(z_real.detach())
        
        d_real = self.discriminator(z_real.detach())
        d_permuted = self.discriminator(z_permuted)
        
        # Binary cross entropy: real=1, permuted=0
        loss_real = F.binary_cross_entropy_with_logits(
            d_real, torch.ones_like(d_real)
        )
        loss_permuted = F.binary_cross_entropy_with_logits(
            d_permuted, torch.zeros_like(d_permuted)
        )
        
        return 0.5 * (loss_real + loss_permuted)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns recon, mu, logvar, AND z for TC computation."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


class TCVAE(BaseVAE):
    """
    β-TCVAE: Total Correlation VAE with decomposed KL.
    
    Decomposes KL divergence into three terms:
    1. Index-Code MI: I(x; z) - mutual info between input and latent
    2. Total Correlation: KL(q(z) || Π_j q(z_j)) - correlation between dimensions
    3. Dimension-wise KL: Σ_j KL(q(z_j) || p(z_j)) - deviation from prior per dimension
    
    Only the Total Correlation term is weighted by β, providing
    more targeted disentanglement without sacrificing reconstruction.
    
    Reference: Chen et al., "Isolating Sources of Disentanglement in VAEs" (NeurIPS 2018)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 16,
        beta: float = 6.0,
        dropout: float = 0.1
    ):
        super().__init__(input_dim, hidden_dims, latent_dim, dropout)
        self.beta = beta
        self.name = f"β-TCVAE (β={beta})"
    
    def loss_function(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        dataset_size: int = 1000
    ) -> Dict[str, torch.Tensor]:
        """
        β-TCVAE loss with decomposed KL divergence.
        
        Uses minibatch weighted sampling to estimate the TC term.
        """
        batch_size = x.size(0)
        
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # Sample z for TC estimation
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Log q(z|x) for the current batch
        log_qz_given_x = self._log_gaussian(z, mu, logvar).sum(dim=1)
        
        # Log q(z) estimated via minibatch weighted sampling
        log_qz = self._log_qz_estimate(z, mu, logvar, dataset_size)
        
        # Log Π_j q(z_j) - product of marginals
        log_qz_product = self._log_qz_product(z, mu, logvar, dataset_size)
        
        # Log p(z) - prior N(0,1)
        log_pz = self._log_gaussian(
            z, 
            torch.zeros_like(z), 
            torch.zeros_like(z)
        ).sum(dim=1)
        
        # Decomposition:
        # MI: E[log q(z|x) - log q(z)]
        mi_loss = (log_qz_given_x - log_qz).mean()
        
        # TC: E[log q(z) - log Π_j q(z_j)]
        tc_loss = (log_qz - log_qz_product).mean()
        
        # Dimension-wise KL: E[log Π_j q(z_j) - log p(z)]
        dw_kl_loss = (log_qz_product - log_pz).mean()
        
        # Total loss: only TC is weighted by β
        total_loss = recon_loss + mi_loss + self.beta * tc_loss + dw_kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'mi_loss': mi_loss,
            'tc_loss': tc_loss,
            'dw_kl_loss': dw_kl_loss
        }
    
    def _log_gaussian(
        self, 
        x: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """Log probability under Gaussian."""
        return -0.5 * (np.log(2 * np.pi) + logvar + (x - mu).pow(2) / logvar.exp())
    
    def _log_qz_estimate(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        dataset_size: int
    ) -> torch.Tensor:
        """
        Estimate log q(z) using minibatch weighted sampling.
        """
        batch_size, latent_dim = z.size()
        
        # Expand for pairwise computation
        z_expand = z.unsqueeze(1)  # (B, 1, D)
        mu_expand = mu.unsqueeze(0)  # (1, B, D)
        logvar_expand = logvar.unsqueeze(0)  # (1, B, D)
        
        # Log q(z_i | x_j) for all pairs
        log_qz_given_xj = self._log_gaussian(z_expand, mu_expand, logvar_expand)
        log_qz_given_xj = log_qz_given_xj.sum(dim=2)  # (B, B)
        
        # Log-mean-exp with importance weight correction
        log_qz = torch.logsumexp(log_qz_given_xj, dim=1) - np.log(batch_size * dataset_size / batch_size)
        
        return log_qz
    
    def _log_qz_product(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        dataset_size: int
    ) -> torch.Tensor:
        """
        Estimate log Π_j q(z_j) - product of marginals.
        """
        batch_size, latent_dim = z.size()
        
        log_qzj = torch.zeros(batch_size, device=z.device)
        
        for j in range(latent_dim):
            z_j = z[:, j:j+1]  # (B, 1)
            mu_j = mu[:, j:j+1]  # (B, 1)
            logvar_j = logvar[:, j:j+1]  # (B, 1)
            
            # Expand for pairwise
            z_j_expand = z_j.unsqueeze(1)  # (B, 1, 1)
            mu_j_expand = mu_j.unsqueeze(0)  # (1, B, 1)
            logvar_j_expand = logvar_j.unsqueeze(0)  # (1, B, 1)
            
            log_qzj_given_xi = self._log_gaussian(z_j_expand, mu_j_expand, logvar_j_expand)
            log_qzj_given_xi = log_qzj_given_xi.squeeze(2)  # (B, B)
            
            log_qzj += torch.logsumexp(log_qzj_given_xi, dim=1) - np.log(batch_size * dataset_size / batch_size)
        
        return log_qzj


def compute_disentanglement_metrics(
    model: BaseVAE,
    data: torch.Tensor,
    factor_labels: torch.Tensor = None,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Compute disentanglement metrics for a trained VAE.
    
    Metrics:
    - Latent dimension variance: How much each dimension varies
    - Latent correlation: How correlated the dimensions are
    - Active dimensions: How many dimensions have significant variance
    """
    model.eval()
    model.to(device)
    data = data.to(device)
    
    with torch.no_grad():
        mu, logvar = model.encode(data)
        z = mu  # Use mean for analysis
    
    z_np = z.cpu().numpy()
    
    # Variance per dimension
    dim_variances = np.var(z_np, axis=0)
    
    # Correlation matrix
    corr_matrix = np.corrcoef(z_np.T)
    
    # Average absolute off-diagonal correlation (lower = more disentangled)
    n_dims = corr_matrix.shape[0]
    off_diag_mask = ~np.eye(n_dims, dtype=bool)
    avg_correlation = np.mean(np.abs(corr_matrix[off_diag_mask]))
    
    # Active dimensions (variance > 0.1)
    active_dims = np.sum(dim_variances > 0.1)
    
    # KL per dimension (from prior)
    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).mean(dim=0)
    kl_per_dim_np = kl_per_dim.cpu().numpy()
    
    return {
        'dim_variances': dim_variances,
        'correlation_matrix': corr_matrix,
        'avg_off_diagonal_correlation': avg_correlation,
        'active_dimensions': int(active_dims),
        'kl_per_dimension': kl_per_dim_np
    }
