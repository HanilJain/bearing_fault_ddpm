# 5. models/dataset.py - Complete dataset code from paper

import os
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
import sys
sys.path.append('..')
from utils.signal_processing import load_mat_signal, normalize_signals

class BearingSignalDataset(Dataset):
    """
    Recursively loads .mat files and exhaustively segments each
    into overlapping windows of length 'segment_length' with hop 'segment_step'.
    
    This implements the exact dataset loading described in Section IV of the paper.
    """
    
    _LABELS = {"healthy": 0, "ir": 1, "or": 2, "b": 3}
    
    def __init__(
        self,
        data_dir: str,
        segment_length: int = 3000,
        segment_step: int = 750,
        severity_codes: List[str] = None,
        include_normal: bool = True,
    ):
        """
        Args:
            data_dir: Root directory containing CWRU dataset
            segment_length: Length of each segment (3000 = 0.25s at 12kHz)
            segment_step: Step size for overlapping segments (75% overlap)
            severity_codes: List of fault severity codes to include
            include_normal: Whether to include healthy/normal data
        """
        super().__init__()
        self.segment_length = segment_length
        self.segment_step = segment_step
        self.severity_codes = [s.lower() for s in (severity_codes or ["007"])]
        
        self.segments: List[np.ndarray] = []
        self.labels: List[int] = []
        
        # Discover files and assign labels
        mat_files: List[Tuple[str, int]] = []
        
        for root, _, fnames in os.walk(data_dir):
            parts = [p.lower() for p in root.split(os.sep)]
            
            for fname in fnames:
                if not fname.lower().endswith(".mat"):
                    continue
                    
                f_low = fname.lower()
                
                # Check for healthy/normal files
                if include_normal and ("normal" in f_low or any("normal" in p for p in parts)):
                    mat_files.append((os.path.join(root, fname), self._LABELS["healthy"]))
                    continue
                
                # Check severity code match
                sev_path = any(code in p for p in parts for code in self.severity_codes)
                sev_file = any(code in f_low for code in self.severity_codes)
                
                if not (sev_path or sev_file):
                    continue
                
                # Determine fault type
                if any("ir" in p for p in parts) or "ir" in f_low:
                    lbl = self._LABELS["ir"]
                elif any("or" in p for p in parts) or "or" in f_low:
                    lbl = self._LABELS["or"] 
                elif any("b" in p for p in parts) or "b" in f_low:
                    lbl = self._LABELS["b"]
                else:
                    continue
                    
                mat_files.append((os.path.join(root, fname), lbl))
        
        if not mat_files:
            raise RuntimeError(f"No matching .mat files found in {data_dir!r}")
        
        # Load and segment signals
        print(f"Loading {len(mat_files)} files...")
        for i, (path, lbl) in enumerate(mat_files):
            try:
                sig = load_mat_signal(path).astype(np.float32)
                
                # Exhaustive segmentation with overlap
                for start in range(0, len(sig) - segment_length + 1, segment_step):
                    crop = sig[start : start + segment_length]
                    self.segments.append(crop)
                    self.labels.append(lbl)
                    
                if (i + 1) % 10 == 0:
                    print(f"Processed {i+1}/{len(mat_files)} files")
                    
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
                continue
        
        # Convert to numpy and apply global Z-normalization
        if not self.segments:
            raise RuntimeError("No segments were successfully loaded")
            
        arr = np.stack(self.segments, axis=0)
        arr = normalize_signals(arr)
        self.segments = arr
        self.labels = np.array(self.labels)
        
        self.num_samples = len(self.segments)
        
        print(f"[BearingSignalDataset] Loaded {self.num_samples} segments "
              f"from {len(mat_files)} files ({self._class_histogram()})")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int):
        seg = torch.from_numpy(self.segments[idx]).unsqueeze(0)  # Add channel dim
        return seg, self.labels[idx]
    
    def _class_histogram(self) -> str:
        cnt = Counter(self.labels)
        return ", ".join(f"{k}:{cnt[v]}" for k, v in self._LABELS.items())


class RandomCropDataset(Dataset):
    """
    Alternative dataset that loads each .mat file fully and returns random crops.
    This was used in the paper for more dynamic data augmentation.
    
    Total samples = num_files * num_crops per epoch
    """
    
    _LABELS = {"healthy": 0, "ir": 1, "or": 2, "b": 3}
    
    def __init__(
        self,
        data_dir: str,
        segment_length: int = 3000,
        num_crops: int = 50,
        severity_codes: List[str] = None,
        include_normal: bool = True,
    ):
        """
        Args:
            data_dir: Root directory containing CWRU dataset
            segment_length: Length of each random crop
            num_crops: Number of crops per file per epoch
            severity_codes: List of fault severity codes to include
            include_normal: Whether to include healthy/normal data
        """
        super().__init__()
        self.segment_length = segment_length
        self.num_crops = num_crops
        self.severity_codes = [s.lower() for s in (severity_codes or ["007"])]
        
        # Discover files and assign labels
        files: List[Tuple[str, int]] = []
        
        for root, _, fnames in os.walk(data_dir):
            parts = [p.lower() for p in root.split(os.sep)]
            
            for fname in fnames:
                if not fname.lower().endswith(".mat"):
                    continue
                    
                f_low = fname.lower()
                
                # Check for healthy/normal files
                if include_normal and ("normal" in f_low or any("normal" in p for p in parts)):
                    files.append((os.path.join(root, fname), self._LABELS["healthy"]))
                    continue
                
                # Check severity code match
                sev_path = any(code in p for p in parts for code in self.severity_codes)
                sev_file = any(code in f_low for code in self.severity_codes)
                
                if not (sev_path or sev_file):
                    continue
                
                # Determine fault type
                if any("ir" in p for p in parts) or "ir" in f_low:
                    lbl = self._LABELS["ir"]
                elif any("or" in p for p in parts) or "or" in f_low:
                    lbl = self._LABELS["or"]
                elif any("b" in p for p in parts) or "b" in f_low:
                    lbl = self._LABELS["b"]
                else:
                    continue
                    
                files.append((os.path.join(root, fname), lbl))
        
        if not files:
            raise RuntimeError(f"No .mat files found in {data_dir!r}")
        
        # Load full signals
        self.signals: List[np.ndarray] = []
        self.labels: List[int] = []
        
        print(f"Loading {len(files)} files for random cropping...")
        for i, (path, lbl) in enumerate(files):
            try:
                sig = load_mat_signal(path).astype(np.float32)
                
                if len(sig) < segment_length:
                    print(f"Warning: File {path} too short ({len(sig)} < {segment_length}), skipping")
                    continue
                    
                self.signals.append(sig)
                self.labels.append(lbl)
                
                if (i + 1) % 10 == 0:
                    print(f"Loaded {i+1}/{len(files)} files")
                    
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
                continue
        
        if not self.signals:
            raise RuntimeError("No signals were successfully loaded")
        
        # Apply global Z-normalization
        all_data = np.concatenate(self.signals, axis=0)
        mu, sigma = all_data.mean(), all_data.std()
        
        for i in range(len(self.signals)):
            self.signals[i] = (self.signals[i] - mu) / (sigma + 1e-12)
        
        self.num_files = len(self.signals)
        self.total_samples = self.num_files * self.num_crops
        
        print(f"[RandomCropDataset] {self.num_files} files, "
              f"{self.total_samples} crops/epoch, "
              f"class dist: {self._class_histogram()}")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx: int):
        file_idx = idx // self.num_crops
        sig = self.signals[file_idx]
        
        # Pick a random starting point for crop
        max_start = len(sig) - self.segment_length
        start = np.random.randint(0, max_start + 1)
        
        seg = sig[start : start + self.segment_length]
        seg = torch.from_numpy(seg).unsqueeze(0)  # Add channel dimension
        
        return seg, self.labels[file_idx]
    
    def _class_histogram(self) -> str:
        cnt = Counter(self.labels)
        return ", ".join(f"{k}:{cnt[v]}" for k, v in self._LABELS.items())



















































