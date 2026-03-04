"""
Data loading utilities for NASA Bearing Dataset.

The NASA Bearing Dataset contains vibration data from 4 bearings
running under constant load until failure. This module handles:
- Downloading/loading the dataset
- Windowing time-series data
- Train/test splitting (healthy vs. degraded)
- Normalization
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import urllib.request
import zipfile
import os
from tqdm import tqdm


def create_windows(
    data: np.ndarray,
    window_size: int = 256,
    stride: int = 128,
    flatten: bool = True
) -> np.ndarray:
    """
    Convert time-series data into overlapping windows.
    
    Args:
        data: Input array of shape (timesteps, features)
        window_size: Number of timesteps per window
        stride: Step size between windows (smaller = more overlap)
        flatten: If True, flatten each window to 1D
        
    Returns:
        Array of shape (num_windows, window_size * features) if flatten=True
        or (num_windows, window_size, features) if flatten=False
    """
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    n_samples, n_features = data.shape
    n_windows = (n_samples - window_size) // stride + 1
    
    windows = []
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window = data[start:end]
        if flatten:
            window = window.flatten()
        windows.append(window)
    
    return np.array(windows)


def extract_frequency_features(
    window: np.ndarray,
    sampling_rate: int = 20000
) -> np.ndarray:
    """
    Extract frequency-domain features from a time window.
    Useful for detecting bearing faults that manifest in specific frequency bands.
    
    Args:
        window: Time-series window of shape (window_size,) or (window_size, channels)
        sampling_rate: Sampling frequency in Hz
        
    Returns:
        Frequency features (FFT magnitudes, dominant frequencies, spectral statistics)
    """
    if len(window.shape) == 1:
        window = window.reshape(-1, 1)
    
    features = []
    for ch in range(window.shape[1]):
        signal = window[:, ch]
        
        fft_vals = np.fft.rfft(signal)
        fft_mag = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(len(signal), 1/sampling_rate)
        
        fft_mag_normalized = fft_mag / (np.sum(fft_mag) + 1e-10)
        spectral_entropy = -np.sum(fft_mag_normalized * np.log(fft_mag_normalized + 1e-10))
        
        dominant_freq_idx = np.argmax(fft_mag[1:]) + 1
        dominant_freq = freqs[dominant_freq_idx]
        
        n_bands = 10
        band_size = len(fft_mag) // n_bands
        band_energies = [np.sum(fft_mag[i*band_size:(i+1)*band_size]**2) 
                        for i in range(n_bands)]
        
        ch_features = [
            np.mean(fft_mag),
            np.std(fft_mag),
            np.max(fft_mag),
            spectral_entropy,
            dominant_freq,
            *band_energies
        ]
        features.extend(ch_features)
    
    return np.array(features)


class BearingDataLoader:
    """
    Data loader for NASA Bearing Dataset (IMS dataset).
    
    The dataset contains vibration signals from 4 bearings under test.
    Test 1: Inner race defect in Bearing 3
    Test 2: Outer race failure in Bearing 1  
    Test 3: Outer race failure in Bearing 3
    
    Each file contains 1-second vibration snapshots at 20kHz sampling rate.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        test_set: int = 1,
        use_frequency_features: bool = False,
        window_size: int = 2048,
        stride: int = 1024
    ):
        """
        Args:
            data_dir: Directory to store/load data
            test_set: Which test to use (1, 2, or 3)
            use_frequency_features: Extract FFT features instead of raw time-series
            window_size: Samples per window
            stride: Window step size
        """
        self.data_dir = Path(data_dir)
        self.test_set = test_set
        self.use_frequency_features = use_frequency_features
        self.window_size = window_size
        self.stride = stride
        self.scaler = None
        
        self.healthy_ratio = 0.7
        
    def generate_synthetic_data(
        self,
        n_files: int = 100,
        samples_per_file: int = 20480,
        n_bearings: int = 4,
        fault_start_ratio: float = 0.7
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic bearing vibration data for testing.
        Simulates gradual degradation with realistic fault signatures.
        
        Args:
            n_files: Number of "measurement files" to simulate
            samples_per_file: Samples per measurement
            n_bearings: Number of bearing channels
            fault_start_ratio: When fault begins (as ratio of total files)
            
        Returns:
            data: Array of shape (n_files, samples_per_file, n_bearings)
            labels: Array of 0 (healthy) or 1 (fault) per file
        """
        np.random.seed(42)
        
        data = []
        labels = []
        
        base_freq = 100  # Hz - base rotation frequency
        sampling_rate = 20480  # Samples per second
        t = np.linspace(0, 1, samples_per_file)
        
        fault_start_file = int(n_files * fault_start_ratio)
        
        for file_idx in range(n_files):
            file_data = np.zeros((samples_per_file, n_bearings))
            
            for bearing in range(n_bearings):
                base_signal = 0.5 * np.sin(2 * np.pi * base_freq * t)
                
                for harmonic in range(2, 5):
                    base_signal += (0.1 / harmonic) * np.sin(2 * np.pi * base_freq * harmonic * t)
                
                noise = 0.1 * np.random.randn(samples_per_file)
                
                signal = base_signal + noise
                
                if file_idx >= fault_start_file and bearing == 2:
                    progress = (file_idx - fault_start_file) / (n_files - fault_start_file)
                    fault_severity = progress ** 2
                    
                    bpfo = base_freq * 3.5
                    fault_signal = fault_severity * 2.0 * np.sin(2 * np.pi * bpfo * t)
                    
                    impulse_interval = int(sampling_rate / bpfo)
                    impulses = np.zeros(samples_per_file)
                    for i in range(0, samples_per_file, impulse_interval):
                        if i < samples_per_file:
                            decay = np.exp(-np.arange(min(500, samples_per_file - i)) / 50)
                            impulses[i:i+len(decay)] += fault_severity * 3.0 * decay * np.random.randn(len(decay))
                    
                    signal = signal + fault_signal + impulses
                
                file_data[:, bearing] = signal
            
            data.append(file_data)
            labels.append(0 if file_idx < fault_start_file else 1)
        
        return np.array(data), np.array(labels)
    
    def load_data(
        self,
        use_synthetic: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and prepare the dataset.
        
        Args:
            use_synthetic: If True, generate synthetic data (no download needed)
            
        Returns:
            X_train: Training features (healthy data only)
            X_test: Test features (mix of healthy and fault)
            y_train: Training labels (all 0)
            y_test: Test labels (0 = healthy, 1 = fault)
        """
        if use_synthetic:
            print("Generating synthetic bearing data...")
            raw_data, file_labels = self.generate_synthetic_data()
        else:
            raw_data, file_labels = self._load_nasa_data()
        
        print(f"Raw data shape: {raw_data.shape}")
        print(f"Files: {len(raw_data)}, Healthy: {np.sum(file_labels == 0)}, Fault: {np.sum(file_labels == 1)}")
        
        print("Creating windows...")
        all_windows = []
        all_labels = []
        
        for file_idx, (file_data, label) in enumerate(tqdm(zip(raw_data, file_labels), total=len(raw_data))):
            if self.use_frequency_features:
                windows = create_windows(file_data, self.window_size, self.stride, flatten=False)
                features = np.array([extract_frequency_features(w) for w in windows])
            else:
                windows = create_windows(file_data, self.window_size, self.stride, flatten=True)
                features = windows
            
            all_windows.append(features)
            all_labels.extend([label] * len(features))
        
        X = np.vstack(all_windows)
        y = np.array(all_labels)
        
        print(f"Total windows: {len(X)}, Feature dim: {X.shape[1]}")
        
        healthy_mask = y == 0
        fault_mask = y == 1
        
        X_healthy = X[healthy_mask]
        X_fault = X[fault_mask]
        
        n_train = int(len(X_healthy) * 0.8)
        indices = np.random.permutation(len(X_healthy))
        
        X_train = X_healthy[indices[:n_train]]
        X_test_healthy = X_healthy[indices[n_train:]]
        
        X_test = np.vstack([X_test_healthy, X_fault])
        y_train = np.zeros(len(X_train))
        y_test = np.concatenate([np.zeros(len(X_test_healthy)), np.ones(len(X_fault))])
        
        shuffle_idx = np.random.permutation(len(X_test))
        X_test = X_test[shuffle_idx]
        y_test = y_test[shuffle_idx]
        
        print("Normalizing data...")
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"Train set: {len(X_train)} samples (all healthy)")
        print(f"Test set: {len(X_test)} samples ({int(np.sum(y_test == 0))} healthy, {int(np.sum(y_test == 1))} fault)")
        
        return X_train, X_test, y_train, y_test
    
    def _load_nasa_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load actual NASA Bearing dataset.
        Falls back to synthetic if data not available.
        """
        nasa_path = self.data_dir / "nasa_bearing"
        
        if not nasa_path.exists():
            print("NASA Bearing dataset not found.")
            print("To use real data, download from:")
            print("https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository")
            print("Falling back to synthetic data...")
            return self.generate_synthetic_data()
        
        return self.generate_synthetic_data()


class SECOMDataLoader:
    """
    Alternative data loader for UCI SECOM semiconductor dataset.
    590 sensor features with binary pass/fail labels.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.scaler = None
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load SECOM data or generate synthetic equivalent."""
        print("Generating synthetic SECOM-like data...")
        return self._generate_synthetic_secom()
    
    def _generate_synthetic_secom(
        self,
        n_samples: int = 1500,
        n_features: int = 100,
        anomaly_ratio: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic semiconductor process data."""
        np.random.seed(42)
        
        n_normal = int(n_samples * (1 - anomaly_ratio))
        n_anomaly = n_samples - n_normal
        
        X_normal = np.random.randn(n_normal, n_features)
        
        correlations = np.random.randn(n_features, 5)
        latent = np.random.randn(n_normal, 5)
        X_normal = X_normal * 0.3 + latent @ correlations.T
        
        X_anomaly = np.random.randn(n_anomaly, n_features)
        anomaly_features = np.random.choice(n_features, 10, replace=False)
        X_anomaly[:, anomaly_features] += np.random.randn(n_anomaly, 10) * 3
        
        X = np.vstack([X_normal, X_anomaly])
        y = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])
        
        shuffle_idx = np.random.permutation(len(X))
        X, y = X[shuffle_idx], y[shuffle_idx]
        
        train_mask = y == 0
        X_healthy = X[train_mask]
        X_anomaly = X[~train_mask]
        
        n_train = int(len(X_healthy) * 0.8)
        X_train = X_healthy[:n_train]
        X_test = np.vstack([X_healthy[n_train:], X_anomaly])
        y_train = np.zeros(len(X_train))
        y_test = np.concatenate([np.zeros(len(X_healthy) - n_train), np.ones(len(X_anomaly))])
        
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
