"""
Variational Autoencoder (VAE) implementations for anomaly detection.

Two architectures provided:
1. VAE: Dense/MLP-based VAE for tabular or flattened time-series data
2. ConvVAE: 1D Convolutional VAE for raw time-series preserving temporal structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, List, Optional


class VAE(nn.Module):
    """
    Dense Variational Autoencoder for anomaly detection.
    
    Architecture:
    - Encoder: Input -> Hidden layers -> (μ, log σ²)
    - Reparameterization: z = μ + σ * ε, where ε ~ N(0,1)
    - Decoder: z -> Hidden layers -> Reconstruction
    
    The model learns to compress "normal" data into a structured latent space.
    Anomalies produce high reconstruction error and/or unusual latent positions.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 16,
        dropout: float = 0.1,
        beta: float = 1.0
    ):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer sizes for encoder
            latent_dim: Dimension of latent space
            dropout: Dropout probability
            beta: Weight for KL divergence (β-VAE). Higher = more regularized latent space
        """
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
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
        
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
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
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            mu: Mean of latent distribution (batch, latent_dim)
            logvar: Log variance of latent distribution (batch, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε
        Allows gradients to flow through sampling.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            z: Sampled latent vector
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction.
        
        Args:
            z: Latent vector (batch, latent_dim)
            
        Returns:
            Reconstructed input (batch, input_dim)
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode -> sample -> decode.
        
        Args:
            x: Input tensor (batch, input_dim)
            
        Returns:
            recon: Reconstructed input
            mu: Latent mean
            logvar: Latent log variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def loss_function(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss: Reconstruction + β * KL divergence.
        
        Args:
            x: Original input
            recon: Reconstructed input
            mu: Latent mean
            logvar: Latent log variance
            
        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss (MSE)
            kl_loss: KL divergence from N(0,1)
        """
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample reconstruction error for anomaly scoring.
        
        Args:
            x: Input tensor (batch, input_dim)
            
        Returns:
            Per-sample MSE (batch,)
        """
        self.eval()
        with torch.no_grad():
            recon, _, _ = self.forward(x)
            error = torch.mean((x - recon) ** 2, dim=1)
        return error
    
    def get_feature_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-feature reconstruction error for root cause analysis.
        
        Args:
            x: Input tensor (batch, input_dim)
            
        Returns:
            Per-feature squared error (batch, input_dim)
        """
        self.eval()
        with torch.no_grad():
            recon, _, _ = self.forward(x)
            error = (x - recon) ** 2
        return error
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation for visualization.
        
        Args:
            x: Input tensor (batch, input_dim)
            
        Returns:
            Latent vectors (batch, latent_dim)
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu


class ConvVAE(nn.Module):
    """
    1D Convolutional VAE for time-series anomaly detection.
    
    Preserves temporal structure through conv layers.
    Better for raw vibration signals where local patterns matter.
    """
    
    def __init__(
        self,
        sequence_length: int,
        n_channels: int = 1,
        latent_dim: int = 16,
        beta: float = 1.0
    ):
        """
        Args:
            sequence_length: Length of input time series
            n_channels: Number of input channels (sensors)
            latent_dim: Dimension of latent space
            beta: β-VAE weight
        """
        super(ConvVAE, self).__init__()
        
        self.sequence_length = sequence_length
        self.n_channels = n_channels
        self.latent_dim = latent_dim
        self.beta = beta
        
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )
        
        self._compute_flat_size()
        
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        
        self.fc_decode = nn.Linear(latent_dim, self.flat_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(32, n_channels, kernel_size=7, stride=2, padding=3, output_padding=1),
        )
        
    def _compute_flat_size(self):
        """Compute flattened size after encoder convolutions."""
        dummy = torch.zeros(1, self.n_channels, self.sequence_length)
        with torch.no_grad():
            out = self.encoder(dummy)
        self.encoder_out_shape = out.shape[1:]
        self.flat_size = out.numel()
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input sequence to latent parameters."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to sequence."""
        h = self.fc_decode(z)
        h = h.view(h.size(0), *self.encoder_out_shape)
        recon = self.decoder(h)
        
        if recon.size(2) != self.sequence_length:
            recon = F.interpolate(recon, size=self.sequence_length, mode='linear', align_corners=False)
        return recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def loss_function(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss."""
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample reconstruction error."""
        self.eval()
        with torch.no_grad():
            recon, _, _ = self.forward(x)
            error = torch.mean((x - recon) ** 2, dim=(1, 2))
        return error
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation."""
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu
