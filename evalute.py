# 10. evaluate.py - Evaluation script
import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from scipy import signal
from sklearn.metrics import mean_squared_error
import seaborn as sns

# Import project modules  
from models.dataset import BearingSignalDataset
from utils.signal_processing import (
    compute_envelope_spectrum, 
    calculate_signal_statistics,
    compute_fault_frequencies
)
from sample import DDPMSampler
from config import *

class SignalEvaluator:
    """
    Comprehensive evaluation of synthetic vs real bearing fault signals
    """
    
    def __init__(self, real_data_path: str):
        """
        Initialize evaluator with real data
        
        Args:
            real_data_path: Path to real CWRU dataset
        """
        self.real_data_path = real_data_path
        self.load_real_data()
        
        # Theoretical fault frequencies for CWRU dataset
        self.fault_freqs = compute_fault_frequencies()
        print(f"Theoretical fault frequencies: {self.fault_freqs}")
    
    def load_real_data(self):
        """Load real bearing fault data for comparison"""
        print("Loading real bearing fault data...")
        
        dataset = BearingSignalDataset(
            data_dir=self.real_data_path,
            segment_length=SEGMENT_LENGTH,
            segment_step=SEGMENT_LENGTH,  # No overlap for evaluation
            severity_codes=SEVERITY_CODES,
            include_normal=INCLUDE_NORMAL
        )
        
        self.real_signals = []
        self.real_labels = []
        
        # Convert to list of signals
        for i in range(len(dataset)):
            signal, label = dataset[i]
            self.real_signals.append(signal.squeeze().numpy())
            self.real_labels.append(label)
        
        print(f"Loaded {len(self.real_signals)} real signals")
        
        # Group by fault type
        self.real_by_class = {0: [], 1: [], 2: [], 3: []}  # healthy, ir, or, ball
        for signal, label in zip(self.real_signals, self.real_labels):
            self.real_by_class[label].append(signal)
        
        class_names = ["Healthy", "Inner Race", "Outer Race", "Ball"]
        for class_id, signals in self.real_by_class.items():
            print(f"  {class_names[class_id]}: {len(signals)} signals")
    
    def time_domain_comparison(self, synthetic_signals: List[np.ndarray], 
                             save_dir: str = None) -> Dict:
        """
        Compare time-domain characteristics between real and synthetic signals
        """
        print("\\nPerforming time-domain comparison...")
        
        # Calculate statistics for both datasets
        real_stats = [calculate_signal_statistics(sig) for sig in self.real_signals[:len(synthetic_signals)]]
        synthetic_stats = [calculate_signal_statistics(sig) for sig in synthetic_signals]
        
        # Aggregate statistics
        metrics = ['rms', 'peak', 'crest_factor', 'skewness', 'kurtosis']
        comparison = {}
        
        for metric in metrics:
            real_values = [stats[metric] for stats in real_stats]
            synthetic_values = [stats[metric] for stats in synthetic_stats]
            
            comparison[metric] = {
                'real_mean': np.mean(real_values),
                'real_std': np.std(real_values),
                'synthetic_mean': np.mean(synthetic_values),
                'synthetic_std': np.std(synthetic_values),
                'mse': mean_squared_error(real_values[:len(synthetic_values)], synthetic_values)
            }
        
        # Create comparison plot
        if save_dir:
            self.plot_statistical_comparison(comparison, save_dir)
        
        return comparison
    
    def frequency_domain_comparison(self, synthetic_signals: List[np.ndarray],
                                  save_dir: str = None) -> Dict:
        """
        Compare frequency-domain characteristics (envelope spectra)
        """
        print("\\nPerforming frequency-domain comparison...")
        
        # Compute envelope spectra
        real_spectra = []
        synthetic_spectra = []
        
        print("Computing envelope spectra for real signals...")
        for signal in self.real_signals[:len(synthetic_signals)]:
            freqs, spectrum = compute_envelope_spectrum(signal)
            real_spectra.append((freqs, spectrum))
        
        print("Computing envelope spectra for synthetic signals...")
        for signal in synthetic_signals:
            freqs, spectrum = compute_envelope_spectrum(signal)
            synthetic_spectra.append((freqs, spectrum))
        
        # Average spectra
        freqs = real_spectra[0][0]  # All should have same frequency array
        real_avg = np.mean([spec[1] for spec in real_spectra], axis=0)
        synthetic_avg = np.mean([spec[1] for spec in synthetic_spectra], axis=0)
        
        # Compute frequency domain similarity
        freq_mse = mean_squared_error(real_avg, synthetic_avg)
        freq_correlation = np.corrcoef(real_avg, synthetic_avg)[0, 1]
        
        # Check fault frequency preservation
        fault_freq_analysis = self.analyze_fault_frequencies(freqs, real_avg, synthetic_avg)
        
        comparison = {
            'frequency_mse': freq_mse,
            'frequency_correlation': freq_correlation,
            'fault_frequencies': fault_freq_analysis
        }
        
        # Create frequency comparison plot
        if save_dir:
            self.plot_frequency_comparison(freqs, real_avg, synthetic_avg, save_dir)
        
        return comparison
    
    def analyze_fault_frequencies(self, freqs: np.ndarray, 
                                real_spectrum: np.ndarray, 
                                synthetic_spectrum: np.ndarray) -> Dict:
        """Analyze how well fault frequencies are preserved"""
        analysis = {}
        
        # Check each theoretical fault frequency
        for freq_name, freq_value in self.fault_freqs.items():
            if freq_name == 'shaft_freq':
                continue
                
            # Find closest frequency bin
            freq_idx = np.argmin(np.abs(freqs - freq_value))
            actual_freq = freqs[freq_idx]
            
            # Get spectral magnitudes at fault frequency
            real_magnitude = real_spectrum[freq_idx]
            synthetic_magnitude = synthetic_spectrum[freq_idx]
            
            analysis[freq_name] = {
                'frequency_hz': freq_value,
                'actual_frequency_hz': actual_freq,
                'real_magnitude': real_magnitude,
                'synthetic_magnitude': synthetic_magnitude,
                'magnitude_ratio': synthetic_magnitude / real_magnitude if real_magnitude > 0 else 0
            }
        
        return analysis
    
    def plot_statistical_comparison(self, comparison: Dict, save_dir: str):
        """Plot statistical comparison between real and synthetic signals"""
        metrics = list(comparison.keys())
        real_means = [comparison[m]['real_mean'] for m in metrics]
        real_stds = [comparison[m]['real_std'] for m in metrics]
        synthetic_means = [comparison[m]['synthetic_mean'] for m in metrics]
        synthetic_stds = [comparison[m]['synthetic_std'] for m in metrics]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, real_means, width, yerr=real_stds, 
                      label='Real Signals', alpha=0.8, capsize=5)
        bars2 = ax.bar(x + width/2, synthetic_means, width, yerr=synthetic_stds,
                      label='Synthetic Signals', alpha=0.8, capsize=5)
        
        ax.set_xlabel('Statistical Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Statistical Comparison: Real vs Synthetic Signals')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "statistical_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Statistical comparison plot saved to: {os.path.join(save_dir, 'statistical_comparison.png')}")
    
    def plot_frequency_comparison(self, freqs: np.ndarray, real_spectrum: np.ndarray, 
                                synthetic_spectrum: np.ndarray, save_dir: str):
        """Plot frequency domain comparison"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(freqs, real_spectrum, 'b-', linewidth=2, label='Real Signals (Average)', alpha=0.8)
        ax.plot(freqs, synthetic_spectrum, 'r-', linewidth=2, label='Synthetic Signals (Average)', alpha=0.8)
        
        # Mark theoretical fault frequencies
        fault_colors = ['orange', 'green', 'purple', 'brown']
        for i, (freq_name, freq_value) in enumerate(self.fault_freqs.items()):
            if freq_name != 'shaft_freq' and freq_value < freqs.max():
                ax.axvline(freq_value, color=fault_colors[i % len(fault_colors)], 
                          linestyle='--', alpha=0.7, label=f'{freq_name}: {freq_value:.1f} Hz')
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title('Envelope Spectrum Comparison: Real vs Synthetic Signals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(2000, freqs.max()))  # Limit to 2 kHz as in paper
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "frequency_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Frequency comparison plot saved to: {os.path.join(save_dir, 'frequency_comparison.png')}")
    
    def comprehensive_evaluation(self, synthetic_signals: List[np.ndarray],
                               save_dir: str) -> Dict:
        """
        Perform comprehensive evaluation matching the paper's methodology
        """
        print(f"\\nPerforming comprehensive evaluation on {len(synthetic_signals)} synthetic signals...")
        os.makedirs(save_dir, exist_ok=True)
        
        # Time domain comparison
        time_comparison = self.time_domain_comparison(synthetic_signals, save_dir)
        
        # Frequency domain comparison
        freq_comparison = self.frequency_domain_comparison(synthetic_signals, save_dir)
        
        # Overall evaluation results
        evaluation_results = {
            'num_synthetic_samples': len(synthetic_signals),
            'num_real_samples': len(self.real_signals),
            'time_domain': time_comparison,
            'frequency_domain': freq_comparison,
            'theoretical_fault_frequencies': self.fault_freqs
        }
        
        # Save evaluation results
        with open(os.path.join(save_dir, "evaluation_results.json"), 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                    return float(obj)
                elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
                    return int(obj)
                return obj
            
            json.dump(evaluation_results, f, indent=2, default=convert_numpy)
        
        # Print summary
        print("\\nEvaluation Summary:")
        print("=" * 50)
        
        print("\\nTime Domain Statistics:")
        for metric, values in time_comparison.items():
            print(f"  {metric:12s}: MSE = {values['mse']:.6f}")
        
        print("\\nFrequency Domain:")
        print(f"  Spectrum MSE:         {freq_comparison['frequency_mse']:.6f}")
        print(f"  Spectrum Correlation: {freq_comparison['frequency_correlation']:.4f}")
        
        print("\\nFault Frequency Analysis:")
        for freq_name, analysis in freq_comparison['fault_frequencies'].items():
            ratio = analysis['magnitude_ratio']
            print(f"  {freq_name:8s}: {analysis['frequency_hz']:6.1f} Hz, Magnitude Ratio = {ratio:.3f}")
        
        print(f"\\nDetailed results saved to: {os.path.join(save_dir, 'evaluation_results.json')}")
        
        return evaluation_results


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate synthetic bearing fault signals')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained DDPM model')
    parser.add_argument('--real_data_path', type=str, required=True,
                      help='Path to real CWRU dataset')
    parser.add_argument('--num_synthetic', type=int, default=100,
                      help='Number of synthetic signals to generate for evaluation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
    parser.add_argument('--load_synthetic', type=str, default=None,
                      help='Path to pre-generated synthetic signals (.npy file)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    print("Initializing signal evaluator...")
    evaluator = SignalEvaluator(args.real_data_path)
    
    # Get synthetic signals
    if args.load_synthetic:
        print(f"Loading pre-generated synthetic signals from: {args.load_synthetic}")
        synthetic_signals_array = np.load(args.load_synthetic)
        synthetic_signals = [signal for signal in synthetic_signals_array]
    else:
        print("Generating synthetic signals for evaluation...")
        sampler = DDPMSampler(args.model_path)
        synthetic_signals = sampler.sample(args.num_synthetic, show_progress=True)
    
    # Perform evaluation
    results = evaluator.comprehensive_evaluation(synthetic_signals, args.output_dir)
    
    print(f"\\nEvaluation completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()