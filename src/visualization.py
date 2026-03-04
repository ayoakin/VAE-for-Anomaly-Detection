"""
Visualization tools for VAE anomaly detection.

Provides:
1. Latent space visualization (2D/3D scatter plots)
2. Reconstruction error plots
3. Feature contribution heatmaps
4. Training history plots
5. Time-series reconstruction comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple
from sklearn.manifold import TSNE
import warnings

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 11


class LatentSpaceVisualizer:
    """
    Visualize the VAE latent space.
    
    For latent dimensions > 2, uses dimensionality reduction (UMAP or t-SNE).
    """
    
    def __init__(self, method: str = "umap", random_state: int = 42):
        """
        Args:
            method: 'umap' or 'tsne' for dimensionality reduction
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.random_state = random_state
        self.reducer = None
        
    def fit_transform(
        self,
        latent_vectors: np.ndarray,
        n_components: int = 2
    ) -> np.ndarray:
        """
        Reduce latent vectors to 2D/3D for visualization.
        
        Args:
            latent_vectors: Latent representations (n_samples, latent_dim)
            n_components: Target dimensions (2 or 3)
            
        Returns:
            Reduced representations (n_samples, n_components)
        """
        if latent_vectors.shape[1] <= n_components:
            return latent_vectors[:, :n_components]
        
        if self.method == "umap" and UMAP_AVAILABLE:
            self.reducer = umap.UMAP(
                n_components=n_components,
                random_state=self.random_state,
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean'
            )
        else:
            if self.method == "umap" and not UMAP_AVAILABLE:
                warnings.warn("UMAP not available, falling back to t-SNE")
            self.reducer = TSNE(
                n_components=n_components,
                random_state=self.random_state,
                perplexity=min(30, len(latent_vectors) - 1),
                n_iter=1000
            )
        
        return self.reducer.fit_transform(latent_vectors)
    
    def plot_latent_space(
        self,
        latent_vectors: np.ndarray,
        labels: Optional[np.ndarray] = None,
        title: str = "VAE Latent Space",
        figsize: Tuple[int, int] = (10, 8),
        alpha: float = 0.6,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create scatter plot of latent space.
        
        Args:
            latent_vectors: Latent representations
            labels: Optional labels (0 = normal, 1 = anomaly)
            title: Plot title
            figsize: Figure size
            alpha: Point transparency
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        setup_plot_style()
        
        coords = self.fit_transform(latent_vectors, n_components=2)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if labels is not None:
            normal_mask = labels == 0
            anomaly_mask = labels == 1
            
            ax.scatter(
                coords[normal_mask, 0],
                coords[normal_mask, 1],
                c='steelblue',
                label='Normal',
                alpha=alpha,
                s=30,
                edgecolors='white',
                linewidth=0.5
            )
            ax.scatter(
                coords[anomaly_mask, 0],
                coords[anomaly_mask, 1],
                c='crimson',
                label='Anomaly',
                alpha=alpha,
                s=50,
                marker='x',
                linewidth=2
            )
            ax.legend(loc='upper right')
        else:
            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=np.arange(len(coords)),
                cmap='viridis',
                alpha=alpha,
                s=30
            )
            plt.colorbar(scatter, ax=ax, label='Sample Index')
        
        method_name = "UMAP" if (self.method == "umap" and UMAP_AVAILABLE) else "t-SNE"
        ax.set_xlabel(f'{method_name} Dimension 1')
        ax.set_ylabel(f'{method_name} Dimension 2')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_latent_3d(
        self,
        latent_vectors: np.ndarray,
        labels: Optional[np.ndarray] = None,
        title: str = "VAE Latent Space (3D)",
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create 3D scatter plot of latent space."""
        setup_plot_style()
        
        coords = self.fit_transform(latent_vectors, n_components=3)
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is not None:
            normal_mask = labels == 0
            anomaly_mask = labels == 1
            
            ax.scatter(
                coords[normal_mask, 0],
                coords[normal_mask, 1],
                coords[normal_mask, 2],
                c='steelblue',
                label='Normal',
                alpha=0.5,
                s=20
            )
            ax.scatter(
                coords[anomaly_mask, 0],
                coords[anomaly_mask, 1],
                coords[anomaly_mask, 2],
                c='crimson',
                label='Anomaly',
                alpha=0.8,
                s=50,
                marker='x'
            )
            ax.legend()
        else:
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=np.arange(len(coords)),
                cmap='viridis',
                alpha=0.6,
                s=20
            )
        
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def plot_reconstruction_error(
    errors: np.ndarray,
    labels: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    title: str = "Reconstruction Error Distribution",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot reconstruction error distribution and/or time series.
    
    Args:
        errors: Reconstruction errors per sample
        labels: Optional labels (0 = normal, 1 = anomaly)
        threshold: Anomaly threshold line
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax1 = axes[0]
    if labels is not None:
        normal_errors = errors[labels == 0]
        anomaly_errors = errors[labels == 1]
        
        ax1.hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='steelblue', density=True)
        ax1.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', color='crimson', density=True)
        ax1.legend()
    else:
        ax1.hist(errors, bins=50, alpha=0.7, color='steelblue', density=True)
    
    if threshold is not None:
        ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
        ax1.legend()
    
    ax1.set_xlabel('Reconstruction Error')
    ax1.set_ylabel('Density')
    ax1.set_title('Error Distribution')
    
    ax2 = axes[1]
    ax2.plot(errors, alpha=0.7, linewidth=0.5, color='steelblue')
    
    if threshold is not None:
        ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        
        if labels is not None:
            anomaly_indices = np.where(labels == 1)[0]
            ax2.scatter(anomaly_indices, errors[anomaly_indices], 
                       c='crimson', s=20, alpha=0.7, label='True Anomalies', zorder=5)
    
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Reconstruction Error')
    ax2.set_title('Error Over Samples')
    ax2.legend()
    
    plt.suptitle(title, y=1.02, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_feature_contributions(
    root_cause_results: List[Dict],
    sample_idx: int = 0,
    top_k: int = 10,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature contributions to reconstruction error (root cause analysis).
    
    Args:
        root_cause_results: Output from AnomalyDetector.analyze_root_cause()
        sample_idx: Which sample to visualize
        top_k: Number of top features to show
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    result = root_cause_results[sample_idx]
    contributors = result['top_contributors'][:top_k]
    
    features = [c['feature_name'] for c in contributors]
    contributions = [c['contribution_pct'] for c in contributors]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(features)))[::-1]
    bars = ax.barh(range(len(features)), contributions, color=colors)
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    
    ax.set_xlabel('Contribution to Error (%)')
    ax.set_title(f'Root Cause Analysis - Top {top_k} Contributing Features\n'
                 f'(Total Error: {result["total_reconstruction_error"]:.4f})')
    
    for i, (bar, pct) in enumerate(zip(bars, contributions)):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: Dict,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training history (losses over epochs).
    
    Args:
        history: Training history from VAETrainer
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], label='Train', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss')
    ax1.legend()
    ax1.set_yscale('log')
    
    ax2 = axes[1]
    ax2.plot(epochs, history['train_recon'], label='Train', linewidth=2)
    ax2.plot(epochs, history['val_recon'], label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Reconstruction Loss')
    ax2.set_title('Reconstruction Loss')
    ax2.legend()
    
    ax3 = axes[2]
    ax3.plot(epochs, history['train_kl'], label='Train', linewidth=2)
    ax3.plot(epochs, history['val_kl'], label='Validation', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('KL Divergence')
    ax3.set_title('KL Divergence')
    ax3.legend()
    
    plt.suptitle('VAE Training History', y=1.02, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_reconstruction_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sample_idx: int = 0,
    n_features: int = 4,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare original vs reconstructed signals.
    
    Args:
        original: Original input samples
        reconstructed: VAE reconstructions
        sample_idx: Which sample to visualize
        n_features: Number of features/channels to show
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    orig = original[sample_idx]
    recon = reconstructed[sample_idx]
    
    n_features = min(n_features, len(orig))
    
    fig, axes = plt.subplots(n_features, 1, figsize=figsize, sharex=True)
    if n_features == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(orig[i::n_features][:500], label='Original', alpha=0.8, linewidth=1)
        ax.plot(recon[i::n_features][:500], label='Reconstructed', alpha=0.8, linewidth=1)
        ax.set_ylabel(f'Feature {i}')
        if i == 0:
            ax.legend(loc='upper right')
    
    axes[-1].set_xlabel('Sample')
    plt.suptitle(f'Original vs Reconstructed (Sample {sample_idx})', y=1.01)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_anomaly_timeline(
    errors: np.ndarray,
    predictions: np.ndarray,
    labels: Optional[np.ndarray] = None,
    threshold: float = None,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot anomaly detection over time.
    
    Useful for predictive maintenance visualization.
    
    Args:
        errors: Reconstruction errors
        predictions: Predicted labels
        labels: True labels (optional)
        threshold: Anomaly threshold
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    ax1 = axes[0]
    ax1.plot(errors, linewidth=1, alpha=0.8, color='steelblue')
    
    if threshold is not None:
        ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: {threshold:.4f}')
        
        anomaly_regions = predictions == 1
        ax1.fill_between(range(len(errors)), 0, errors.max(),
                        where=anomaly_regions, alpha=0.2, color='red',
                        label='Predicted Anomaly')
    
    ax1.set_ylabel('Reconstruction Error')
    ax1.set_title('Anomaly Detection Timeline')
    ax1.legend(loc='upper left')
    
    ax2 = axes[1]
    ax2.fill_between(range(len(predictions)), 0, predictions,
                    alpha=0.7, color='crimson', label='Predicted')
    
    if labels is not None:
        ax2.plot(labels, 'k--', linewidth=2, alpha=0.7, label='Ground Truth')
        ax2.legend()
    
    ax2.set_xlabel('Sample Index (Time)')
    ax2.set_ylabel('Anomaly Flag')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Normal', 'Anomaly'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
