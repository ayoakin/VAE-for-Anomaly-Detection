"""
Anomaly detection and root cause analysis using trained VAE.

This module provides:
1. Anomaly scoring based on reconstruction error
2. Threshold determination methods
3. Root cause analysis via feature contribution
4. Latent space analysis for drift detection
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix
import warnings

from .vae_model import VAE


class AnomalyDetector:
    """
    Anomaly detector using trained VAE model.
    
    Detection Methods:
    1. Reconstruction Error: High error = anomaly
    2. Latent Distance: Distance from training distribution center
    3. Combined Score: Weighted combination of both
    
    Root Cause Analysis:
    - Per-feature reconstruction error ranking
    - Gradient-based attribution
    """
    
    def __init__(
        self,
        model: VAE,
        device: str = "auto"
    ):
        """
        Args:
            model: Trained VAE model
            device: Computation device
        """
        self.model = model
        
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.threshold = None
        self.train_errors = None
        self.train_latent_mean = None
        self.train_latent_std = None
        
    def fit_threshold(
        self,
        X_train: np.ndarray,
        method: str = "percentile",
        percentile: float = 99,
        n_std: float = 3.0
    ):
        """
        Determine anomaly threshold from training data.
        
        Args:
            X_train: Normal training samples
            method: 'percentile' or 'std' (standard deviation)
            percentile: Percentile for threshold (if method='percentile')
            n_std: Number of standard deviations (if method='std')
        """
        print("Computing reconstruction errors on training data...")
        self.train_errors = self.get_reconstruction_errors(X_train)
        
        if method == "percentile":
            self.threshold = np.percentile(self.train_errors, percentile)
            print(f"Threshold set to {percentile}th percentile: {self.threshold:.6f}")
        elif method == "std":
            mean_error = np.mean(self.train_errors)
            std_error = np.std(self.train_errors)
            self.threshold = mean_error + n_std * std_error
            print(f"Threshold set to mean + {n_std}*std: {self.threshold:.6f}")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print("Computing latent space statistics...")
        train_tensor = torch.FloatTensor(X_train).to(self.device)
        train_latent = self.model.get_latent(train_tensor).cpu().numpy()
        self.train_latent_mean = np.mean(train_latent, axis=0)
        self.train_latent_std = np.std(train_latent, axis=0)
        
        return self.threshold
    
    def get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error for each sample.
        
        Args:
            X: Input samples (n_samples, n_features)
            
        Returns:
            Per-sample reconstruction error (n_samples,)
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        errors = self.model.get_reconstruction_error(X_tensor)
        return errors.cpu().numpy()
    
    def get_anomaly_scores(
        self,
        X: np.ndarray,
        use_latent_distance: bool = False,
        latent_weight: float = 0.3
    ) -> np.ndarray:
        """
        Compute anomaly scores for samples.
        
        Args:
            X: Input samples
            use_latent_distance: Include latent space distance in score
            latent_weight: Weight for latent distance component
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        recon_errors = self.get_reconstruction_errors(X)
        
        if not use_latent_distance or self.train_latent_mean is None:
            return recon_errors
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        latent = self.model.get_latent(X_tensor).cpu().numpy()
        
        normalized_latent = (latent - self.train_latent_mean) / (self.train_latent_std + 1e-10)
        latent_distances = np.sqrt(np.sum(normalized_latent ** 2, axis=1))
        
        recon_normalized = (recon_errors - np.mean(self.train_errors)) / (np.std(self.train_errors) + 1e-10)
        latent_normalized = (latent_distances - np.mean(latent_distances)) / (np.std(latent_distances) + 1e-10)
        
        scores = (1 - latent_weight) * recon_normalized + latent_weight * latent_normalized
        
        return scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X: Input samples
            
        Returns:
            Binary labels (0 = normal, 1 = anomaly)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call fit_threshold() first.")
        
        errors = self.get_reconstruction_errors(X)
        return (errors > self.threshold).astype(int)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate anomaly detection performance.
        
        Args:
            X_test: Test samples
            y_test: True labels (0 = normal, 1 = anomaly)
            verbose: Print results
            
        Returns:
            Dictionary of evaluation metrics
        """
        errors = self.get_reconstruction_errors(X_test)
        y_pred = self.predict(X_test)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc = roc_auc_score(y_test, errors)
        
        precision, recall, thresholds = precision_recall_curve(y_test, errors)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_f1_idx]
        best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else self.threshold
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        metrics = {
            'auc_roc': auc,
            'f1_score': f1_score(y_test, y_pred),
            'best_f1': best_f1,
            'best_threshold': best_threshold,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'threshold_used': self.threshold,
            'normal_error_mean': np.mean(errors[y_test == 0]),
            'normal_error_std': np.std(errors[y_test == 0]),
            'anomaly_error_mean': np.mean(errors[y_test == 1]),
            'anomaly_error_std': np.std(errors[y_test == 1])
        }
        
        if verbose:
            print("\n" + "=" * 50)
            print("ANOMALY DETECTION RESULTS")
            print("=" * 50)
            print(f"AUC-ROC Score:     {metrics['auc_roc']:.4f}")
            print(f"F1 Score:          {metrics['f1_score']:.4f}")
            print(f"Best Possible F1:  {metrics['best_f1']:.4f}")
            print(f"Precision:         {metrics['precision']:.4f}")
            print(f"Recall:            {metrics['recall']:.4f}")
            print("-" * 50)
            print(f"True Positives:    {metrics['true_positives']}")
            print(f"False Positives:   {metrics['false_positives']}")
            print(f"True Negatives:    {metrics['true_negatives']}")
            print(f"False Negatives:   {metrics['false_negatives']}")
            print("-" * 50)
            print(f"Normal Error:      {metrics['normal_error_mean']:.4f} ± {metrics['normal_error_std']:.4f}")
            print(f"Anomaly Error:     {metrics['anomaly_error_mean']:.4f} ± {metrics['anomaly_error_std']:.4f}")
            print("=" * 50)
        
        return metrics
    
    def analyze_root_cause(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Perform root cause analysis on anomalous samples.
        
        Identifies which features contributed most to the reconstruction error.
        
        Args:
            X: Anomalous samples to analyze
            feature_names: Names of input features
            top_k: Number of top contributing features to return
            
        Returns:
            List of dictionaries with root cause analysis per sample
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        feature_errors = self.model.get_feature_reconstruction_error(X_tensor).cpu().numpy()
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        results = []
        for i in range(len(X)):
            sample_errors = feature_errors[i]
            
            sorted_indices = np.argsort(sample_errors)[::-1]
            
            top_contributors = []
            total_error = np.sum(sample_errors)
            
            for j, idx in enumerate(sorted_indices[:top_k]):
                contribution = {
                    'rank': j + 1,
                    'feature_name': feature_names[idx],
                    'feature_index': int(idx),
                    'squared_error': float(sample_errors[idx]),
                    'contribution_pct': float(sample_errors[idx] / total_error * 100)
                }
                top_contributors.append(contribution)
            
            results.append({
                'sample_index': i,
                'total_reconstruction_error': float(total_error),
                'top_contributors': top_contributors
            })
        
        return results
    
    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """
        Get latent space representation of samples.
        
        Args:
            X: Input samples
            
        Returns:
            Latent vectors (n_samples, latent_dim)
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        return self.model.get_latent(X_tensor).cpu().numpy()
    
    def detect_drift(
        self,
        X_new: np.ndarray,
        significance_level: float = 0.05
    ) -> Dict:
        """
        Detect if new data has drifted from training distribution.
        
        Uses statistical tests on latent representations.
        
        Args:
            X_new: New data to test
            significance_level: P-value threshold for drift detection
            
        Returns:
            Drift analysis results
        """
        if self.train_latent_mean is None:
            raise ValueError("Training statistics not computed. Call fit_threshold() first.")
        
        new_latent = self.get_latent_representation(X_new)
        
        drift_detected = []
        p_values = []
        
        for dim in range(new_latent.shape[1]):
            expected_mean = self.train_latent_mean[dim]
            expected_std = self.train_latent_std[dim]
            
            t_stat, p_value = stats.ttest_1samp(new_latent[:, dim], expected_mean)
            p_values.append(p_value)
            drift_detected.append(p_value < significance_level)
        
        overall_drift = any(drift_detected)
        
        return {
            'drift_detected': overall_drift,
            'dimensions_with_drift': sum(drift_detected),
            'p_values': p_values,
            'drift_per_dimension': drift_detected,
            'mean_latent_shift': np.mean(np.abs(np.mean(new_latent, axis=0) - self.train_latent_mean))
        }
