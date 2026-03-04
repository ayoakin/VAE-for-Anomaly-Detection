"""
Training utilities for VAE anomaly detection.

Handles:
- Training loop with early stopping
- Learning rate scheduling
- Checkpointing
- Training history tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import json

from .vae_model import VAE


class VAETrainer:
    """
    Trainer class for VAE models.
    
    Implements best practices:
    - Training on healthy data only (unsupervised anomaly detection)
    - Early stopping based on validation loss
    - Learning rate scheduling
    - Gradient clipping for stability
    """
    
    def __init__(
        self,
        model: VAE,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "auto",
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Args:
            model: VAE model to train
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength
            device: 'cuda', 'mps', 'cpu', or 'auto'
            checkpoint_dir: Directory for saving model checkpoints
        """
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon': [],
            'val_recon': [],
            'train_kl': [],
            'val_kl': [],
            'lr': []
        }
        
    def train(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 64,
        early_stopping_patience: int = 20,
        grad_clip: float = 1.0,
        val_split: float = 0.1
    ) -> Dict:
        """
        Train the VAE on healthy data.
        
        Args:
            X_train: Training data (healthy samples only)
            X_val: Optional validation data
            epochs: Maximum training epochs
            batch_size: Mini-batch size
            early_stopping_patience: Stop if no improvement for this many epochs
            grad_clip: Gradient clipping value
            val_split: Fraction of training data to use for validation if X_val not provided
            
        Returns:
            Training history dictionary
        """
        if X_val is None:
            n_val = int(len(X_train) * val_split)
            indices = np.random.permutation(len(X_train))
            X_val = X_train[indices[:n_val]]
            X_train = X_train[indices[n_val:]]
        
        train_tensor = torch.FloatTensor(X_train)
        val_tensor = torch.FloatTensor(X_val)
        
        train_dataset = TensorDataset(train_tensor)
        val_dataset = TensorDataset(val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\nTraining VAE for {epochs} epochs...")
        print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        print("-" * 60)
        
        for epoch in range(epochs):
            train_metrics = self._train_epoch(train_loader, grad_clip)
            val_metrics = self._validate(val_loader)
            
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_recon'].append(train_metrics['recon'])
            self.history['val_recon'].append(val_metrics['recon'])
            self.history['train_kl'].append(train_metrics['kl'])
            self.history['val_kl'].append(val_metrics['kl'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            self.scheduler.step(val_metrics['loss'])
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                self._save_checkpoint('best_model.pt', epoch, val_metrics['loss'])
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_metrics['loss']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Recon: {val_metrics['recon']:.4f} | "
                      f"KL: {val_metrics['kl']:.4f}")
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        self._load_checkpoint('best_model.pt')
        print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
        
        self._save_history()
        
        return self.history
    
    def _train_epoch(self, loader: DataLoader, grad_clip: float) -> Dict:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        n_batches = 0
        
        for (batch,) in loader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            recon, mu, logvar = self.model(batch)
            loss, recon_loss, kl_loss = self.model.loss_function(batch, recon, mu, logvar)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'recon': total_recon / n_batches,
            'kl': total_kl / n_batches
        }
    
    def _validate(self, loader: DataLoader) -> Dict:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        n_batches = 0
        
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                recon, mu, logvar = self.model(batch)
                loss, recon_loss, kl_loss = self.model.loss_function(batch, recon, mu, logvar)
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
                n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'recon': total_recon / n_batches,
            'kl': total_kl / n_batches
        }
    
    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'model_config': {
                'input_dim': self.model.input_dim,
                'latent_dim': self.model.latent_dim,
                'beta': self.model.beta
            }
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def _save_history(self):
        """Save training history to JSON."""
        with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
