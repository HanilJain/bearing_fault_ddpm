# 9. sample.py - Complete sampling/generation script
import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Optional
from tqdm import tqdm

# Import project modules
from models.models import ImprovedUNet
from utils.signal_processing import compute_envelope_spectrum, calculate_signal_statistics
from config import *

class DDPMSampler:
    """
    DDPM sampler for generating synthetic bearing fault signals
    Implements Algorithm 2 from the research paper
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize sampler with trained model
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use for inference
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self.load_model(model_path)
        
        # Setup diffusion parameters (should match training)
        self.T = T
        self.betas = torch.linspace(BETA_START, BETA_END, self.T)
        self.alphas = 1 - self.betas
        self.alpha_cum = torch.cumprod(self.alphas, dim=0)
        
        # Move to device
        self.alpha_cum = self.alpha_cum.to(self.device)
        
        print(f"Model loaded successfully with {self.T} diffusion steps")
    
    def load_model(self, model_path: str) -> ImprovedUNet:
        """Load trained model from checkpoint"""
        model = ImprovedUNet(
            in_channels=IN_CHANNELS,
            base_channels=BASE_CHANNELS,
            channel_mults=CHANNEL_MULTS,
            num_heads=NUM_ATTENTION_HEADS,
            time_emb_dim=TIME_EMB_DIM
        )
        
        # Load state dict
        if model_path.endswith('.pth'):
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    model.load_state_dict(checkpoint)
            else:
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def sample(self, num_samples: int = 1, show_progress: bool = True) -> List[np.ndarray]:
        """
        Generate synthetic signals using reverse diffusion
        Implements Algorithm 2 from the paper
        
        Args:
            num_samples: Number of signals to generate
            show_progress: Whether to show progress bar
        
        Returns:
            List of generated signals as numpy arrays
        """
        generated_signals = []
        
        with torch.no_grad():
            for sample_idx in tqdm(range(num_samples), desc="Generating samples", disable=not show_progress):
                # Start from pure noise: x_T ~ N(0, I)
                x = torch.randn(1, 1, SEGMENT_LENGTH, device=self.device)
                
                # Reverse diffusion process
                timesteps = range(self.T - 1, -1, -1)
                if show_progress and num_samples == 1:
                    timesteps = tqdm(timesteps, desc="Reverse diffusion steps")
                
                for t in timesteps:
                    t_tensor = torch.tensor([t], device=self.device)
                    
                    # Predict noise with model
                    noise_pred = self.model(x, t_tensor)
                    
                    # Compute x_0 estimate: x̂_0 = (x_t - √(1-ᾱ_t) * ε̂) / √ᾱ_t
                    sqrt_alpha_cum_t = torch.sqrt(self.alpha_cum[t])
                    sqrt_one_minus_alpha_cum_t = torch.sqrt(1 - self.alpha_cum[t])
                    
                    x0_pred = (x - sqrt_one_minus_alpha_cum_t * noise_pred) / sqrt_alpha_cum_t
                    
                    if t > 0:
                        # Sample x_{t-1} using predicted x_0 and added noise
                        sqrt_alpha_cum_t_prev = torch.sqrt(self.alpha_cum[t-1])
                        sqrt_one_minus_alpha_cum_t_prev = torch.sqrt(1 - self.alpha_cum[t-1])
                        
                        # Add noise for sampling uncertainty
                        z = torch.randn_like(x)
                        x = sqrt_alpha_cum_t_prev * x0_pred + sqrt_one_minus_alpha_cum_t_prev * z
                    else:
                        # Final step: x_0 = x̂_0 (no added noise)
                        x = x0_pred
                
                # Convert to numpy and store
                signal = x.squeeze().cpu().numpy()
                generated_signals.append(signal)
        
        return generated_signals
    
    def sample_batch(self, batch_size: int = 8, show_progress: bool = True) -> np.ndarray:
        """
        Generate a batch of signals simultaneously for efficiency
        
        Args:
            batch_size: Number of signals to generate in parallel
            show_progress: Whether to show progress bar
        
        Returns:
            Array of generated signals (batch_size, signal_length)
        """
        with torch.no_grad():
            # Start from pure noise: x_T ~ N(0, I)
            x = torch.randn(batch_size, 1, SEGMENT_LENGTH, device=self.device)
            
            # Reverse diffusion process
            timesteps = range(self.T - 1, -1, -1)
            if show_progress:
                timesteps = tqdm(timesteps, desc="Reverse diffusion steps")
            
            for t in timesteps:
                t_tensor = torch.tensor([t] * batch_size, device=self.device)
                
                # Predict noise with model
                noise_pred = self.model(x, t_tensor)
                
                # Compute x_0 estimate
                sqrt_alpha_cum_t = torch.sqrt(self.alpha_cum[t])
                sqrt_one_minus_alpha_cum_t = torch.sqrt(1 - self.alpha_cum[t])
                
                x0_pred = (x - sqrt_one_minus_alpha_cum_t * noise_pred) / sqrt_alpha_cum_t
                
                if t > 0:
                    # Sample x_{t-1}
                    sqrt_alpha_cum_t_prev = torch.sqrt(self.alpha_cum[t-1])
                    sqrt_one_minus_alpha_cum_t_prev = torch.sqrt(1 - self.alpha_cum[t-1])
                    
                    z = torch.randn_like(x)
                    x = sqrt_alpha_cum_t_prev * x0_pred + sqrt_one_minus_alpha_cum_t_prev * z
                else:
                    x = x0_pred
            
            # Convert to numpy
            signals = x.squeeze(1).cpu().numpy()  # Remove channel dimension
            
        return signals


def visualize_samples(signals: List[np.ndarray], save_path: str = None, title: str = "Generated Signals"):
    """Visualize generated signals"""
    n_samples = len(signals)
    n_cols = min(4, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_samples == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, signal in enumerate(signals):
        ax = axes[i] if n_samples > 1 else axes[0]
        
        # Time axis (0.25 seconds at 12 kHz)
        time = np.arange(len(signal)) / SAMPLING_RATE
        
        ax.plot(time, signal, 'b-', linewidth=0.8)
        ax.set_title(f'Generated Signal {i+1}', fontsize=10)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def analyze_samples(signals: List[np.ndarray], save_dir: str = None):
    """Analyze generated signals and compute statistics"""
    print("\\nAnalyzing generated signals...")
    
    # Compute statistics for each signal
    all_stats = []
    for i, signal in enumerate(signals):
        stats = calculate_signal_statistics(signal)
        all_stats.append(stats)
        
        if i < 5:  # Print stats for first 5 signals
            print(f"\\nSignal {i+1} statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value:.4f}")
    
    # Aggregate statistics
    stat_keys = all_stats[0].keys()
    aggregated = {}
    
    for key in stat_keys:
        values = [stats[key] for stats in all_stats]
        aggregated[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    print(f"\\nAggregated statistics across {len(signals)} samples:")
    print("-" * 50)
    for key, stats in aggregated.items():
        print(f"{key:12s}: {stats['mean']:8.4f} ± {stats['std']:6.4f} "
              f"[{stats['min']:6.4f}, {stats['max']:6.4f}]")
    
    # Save statistics if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save detailed statistics
        with open(os.path.join(save_dir, "signal_statistics.json"), 'w') as f:
            json.dump({
                'individual_stats': all_stats,
                'aggregated_stats': aggregated,
                'num_samples': len(signals)
            }, f, indent=2)
        
        print(f"\\nStatistics saved to: {os.path.join(save_dir, 'signal_statistics.json')}")
    
    return all_stats, aggregated


def main():
    """Main sampling function"""
    parser = argparse.ArgumentParser(description='Generate synthetic bearing fault signals using trained DDPM')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--num_samples', type=int, default=10,
                      help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for generation (if > 1, uses batch sampling)')
    parser.add_argument('--output_dir', type=str, default='generated_samples',
                      help='Directory to save generated samples and analysis')
    parser.add_argument('--save_signals', action='store_true',
                      help='Save generated signals as numpy arrays')
    parser.add_argument('--visualize', action='store_true', default=True,
                      help='Create visualizations of generated signals')
    parser.add_argument('--analyze', action='store_true', default=True,
                      help='Perform statistical analysis of generated signals')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize sampler
    print("Initializing DDPM sampler...")
    sampler = DDPMSampler(args.model_path)
    
    # Generate samples
    print(f"\\nGenerating {args.num_samples} synthetic signals...")
    
    if args.batch_size > 1 and args.num_samples > args.batch_size:
        # Generate in batches
        all_signals = []
        remaining = args.num_samples
        
        while remaining > 0:
            current_batch = min(args.batch_size, remaining)
            batch_signals = sampler.sample_batch(current_batch, show_progress=True)
            all_signals.extend([signal for signal in batch_signals])
            remaining -= current_batch
            
            print(f"Generated {len(all_signals)}/{args.num_samples} signals")
    
    else:
        # Generate individually
        all_signals = sampler.sample(args.num_samples, show_progress=True)
    
    print(f"\\nGeneration completed! Generated {len(all_signals)} signals.")
    
    # Save signals if requested
    if args.save_signals:
        signals_array = np.array(all_signals)
        save_path = os.path.join(args.output_dir, "generated_signals.npy")
        np.save(save_path, signals_array)
        print(f"Signals saved to: {save_path}")
    
    # Visualize samples
    if args.visualize:
        n_vis = min(8, len(all_signals))  # Visualize up to 8 signals
        vis_path = os.path.join(args.output_dir, "sample_visualization.png")
        visualize_samples(all_signals[:n_vis], vis_path, 
                         f"Generated Bearing Fault Signals (First {n_vis})")
    
    # Analyze samples
    if args.analyze:
        analyze_samples(all_signals, args.output_dir)
    
    print(f"\\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

