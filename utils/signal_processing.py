# 3. utils/signal_processing.py

import numpy as np
import scipy.io
from scipy import signal
from typing import Union, Tuple
import os

def load_mat_signal(mat_path: str, key: str = None) -> np.ndarray:
    """
    Load signal from .mat file (CWRU dataset format)
    
    Args:
        mat_path: Path to .mat file
        key: Specific key to extract (auto-detect if None)
    
    Returns:
        1D numpy array containing the vibration signal
    """
    try:
        mat_data = scipy.io.loadmat(mat_path)
        
        # Remove metadata keys
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        
        if key:
            if key in mat_data:
                sig = mat_data[key]
            else:
                raise KeyError(f"Key '{key}' not found in {mat_path}")
        else:
            # Auto-detect the signal key (usually the longest array)
            signal_key = None
            max_length = 0
            for k in data_keys:
                if isinstance(mat_data[k], np.ndarray):
                    if mat_data[k].size > max_length:
                        max_length = mat_data[k].size
                        signal_key = k
            
            if signal_key is None:
                raise ValueError(f"No signal data found in {mat_path}")
            
            sig = mat_data[signal_key]
        
        # Flatten if needed
        if sig.ndim > 1:
            sig = sig.flatten()
            
        return sig.astype(np.float32)
        
    except Exception as e:
        raise IOError(f"Error loading {mat_path}: {str(e)}")

def normalize_signals(signals: np.ndarray) -> np.ndarray:
    """
    Apply global Z-score normalization across all signals
    
    Args:
        signals: Array of shape (N, signal_length)
    
    Returns:
        Normalized signals with zero mean and unit variance
    """
    # Compute global statistics
    all_data = signals.flatten()
    mu = np.mean(all_data)
    sigma = np.std(all_data)
    
    # Apply normalization
    normalized = (signals - mu) / (sigma + 1e-12)  # Add small epsilon for stability
    
    return normalized.astype(np.float32)

def compute_envelope_spectrum(signal_data: np.ndarray, fs: int = 12000, 
                            max_freq: float = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute envelope spectrum of a vibration signal for fault analysis
    
    Args:
        signal_data: 1D vibration signal
        fs: Sampling frequency in Hz
        max_freq: Maximum frequency to return
    
    Returns:
        frequencies: Frequency array
        spectrum: Envelope spectrum magnitude
    """
    # Compute envelope using Hilbert transform
    analytic_signal = signal.hilbert(signal_data)
    envelope = np.abs(analytic_signal)
    
    # Remove DC component
    envelope = envelope - np.mean(envelope)
    
    # Compute FFT of envelope
    n_fft = len(envelope)
    freqs = np.fft.fftfreq(n_fft, 1/fs)
    spectrum = np.abs(np.fft.fft(envelope))
    
    # Take positive frequencies only
    pos_freqs = freqs[:n_fft//2]
    pos_spectrum = spectrum[:n_fft//2]
    
    # Limit to max_freq
    if max_freq:
        max_idx = np.where(pos_freqs <= max_freq)[0][-1]
        pos_freqs = pos_freqs[:max_idx]
        pos_spectrum = pos_spectrum[:max_idx]
    
    return pos_freqs, pos_spectrum

def compute_fault_frequencies(shaft_speed_rpm: float = 1750, 
                            ball_diameter: float = 7.94,
                            pitch_diameter: float = 39.04,
                            num_balls: int = 9,
                            contact_angle: float = 0) -> dict:
    """
    Calculate theoretical bearing fault frequencies for CWRU dataset
    
    Args:
        shaft_speed_rpm: Shaft rotational speed in RPM
        ball_diameter: Ball diameter in mm
        pitch_diameter: Pitch diameter in mm  
        num_balls: Number of rolling elements
        contact_angle: Contact angle in degrees
    
    Returns:
        Dictionary with fault frequencies in Hz
    """
    # Convert to Hz
    shaft_freq = shaft_speed_rpm / 60.0
    
    # Convert contact angle to radians
    contact_angle_rad = np.radians(contact_angle)
    
    # Ball pass frequency outer race (BPFO)
    bpfo = (num_balls / 2) * shaft_freq * (1 - (ball_diameter / pitch_diameter) * np.cos(contact_angle_rad))
    
    # Ball pass frequency inner race (BPFI) 
    bpfi = (num_balls / 2) * shaft_freq * (1 + (ball_diameter / pitch_diameter) * np.cos(contact_angle_rad))
    
    # Ball spin frequency (BSF)
    bsf = (pitch_diameter / (2 * ball_diameter)) * shaft_freq * (1 - ((ball_diameter / pitch_diameter) * np.cos(contact_angle_rad))**2)
    
    # Fundamental train frequency (FTF)
    ftf = (shaft_freq / 2) * (1 - (ball_diameter / pitch_diameter) * np.cos(contact_angle_rad))
    
    return {
        'shaft_freq': shaft_freq,
        'BPFO': bpfo,
        'BPFI': bpfi, 
        'BSF': bsf,
        'FTF': ftf
    }

def calculate_signal_statistics(signal_data: np.ndarray) -> dict:
    """
    Calculate statistical features of vibration signal
    
    Args:
        signal_data: 1D vibration signal
    
    Returns:
        Dictionary with statistical metrics
    """
    from scipy import stats
    
    # Basic statistics
    rms = np.sqrt(np.mean(signal_data**2))
    peak = np.max(np.abs(signal_data))
    crest_factor = peak / rms if rms > 0 else 0
    
    # Higher order moments
    skewness = stats.skew(signal_data)
    kurt = stats.kurtosis(signal_data)
    
    return {
        'rms': rms,
        'peak': peak,
        'crest_factor': crest_factor,
        'skewness': skewness,
        'kurtosis': kurt,
        'mean': np.mean(signal_data),
        'std': np.std(signal_data)
    }
