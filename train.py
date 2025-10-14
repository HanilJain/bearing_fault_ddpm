# 8. train.py - Complete training script from the research paper
import os
import sys
from collections import Counter
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import json
from datetime import datetime

# Import project modules
from models.models import ImprovedUNet
from models.dataset import BearingSignalDataset, RandomCropDataset
from config import *

class DDPMTrainer:
    """
    Complete DDPM trainer implementing the exact methodology from the paper
    """
    
    def __init__(self, config_dict=None):
        """Initialize trainer with configuration"""
        # Use global config or provided config
        self.T = config_dict.get('T', T) if config_dict else T
        self.beta_start = config_dict.get('BETA_START', BETA_START) if config_dict else BETA_START
        self.beta_end = config_dict.get('BETA_END', BETA_END) if config_dict else BETA_END
        
        # Precompute diffusion schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.T)
        self.alphas = 1 - self.betas
        self.alpha_cum = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum = torch.sqrt(self.alpha_cum)
        self.sqrt_one_minus_alpha_cum = torch.sqrt(1 - self.alpha_cum)
        
        # Training parameters
        self.num_epochs = config_dict.get('NUM_EPOCHS', NUM_EPOCHS) if config_dict else NUM_EPOCHS
        self.batch_size = config_dict.get('BATCH_SIZE', BATCH_SIZE) if config_dict else BATCH_SIZE
        self.learning_rate = config_dict.get('LEARNING_RATE', LEARNING_RATE) if config_dict else LEARNING_RATE
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Move diffusion parameters to device
        self.alpha_cum = self.alpha_cum.to(self.device)
        self.sqrt_alpha_cum = self.sqrt_alpha_cum.to(self.device)
        self.sqrt_one_minus_alpha_cum = self.sqrt_one_minus_alpha_cum.to(self.device)
    
    def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor):
        """
        Apply forward diffusion process (add noise)
        Implements Equation (2) from the paper
        """
        # Get noise scaling factors for timesteps t
        sqrt_ac = self.sqrt_alpha_cum[t].view(-1, 1, 1)
        sqrt_omc = self.sqrt_one_minus_alpha_cum[t].view(-1, 1, 1)
        
        # Sample noise
        noise = torch.randn_like(x0)
        
        # Apply forward diffusion: x_t = sqrt(α̂_t) * x_0 + sqrt(1-α̂_t) * ε
        x_t = sqrt_ac * x0 + sqrt_omc * noise
        
        return x_t, noise
    
    def setup_data(self, data_dir: str, use_random_crop: bool = True):
        """
        Setup dataset and dataloader with balanced sampling
        """
        if use_random_crop:
            # Use RandomCropDataset as described in the paper
            dataset = RandomCropDataset(
                data_dir=data_dir,
                segment_length=SEGMENT_LENGTH,
                num_crops=50,  # 50 crops per file as mentioned in paper
                severity_codes=SEVERITY_CODES,
                include_normal=INCLUDE_NORMAL
            )
            
            # Setup weighted sampling for balanced classes per epoch
            file_labels = dataset.labels  # One label per file
            num_files = len(file_labels)
            
            # Compute class weights
            counts = Counter(file_labels)
            class_weights = {cls: 1.0/count for cls, count in counts.items()}
            
            # Sample weights for each dataset index
            sample_weights = [
                class_weights[file_labels[idx // dataset.num_crops]]
                for idx in range(len(dataset))
            ]
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
        else:
            # Use exhaustive segmentation
            dataset = BearingSignalDataset(
                data_dir=data_dir,
                segment_length=SEGMENT_LENGTH,
                segment_step=SEGMENT_STEP,
                severity_codes=SEVERITY_CODES,
                include_normal=INCLUDE_NORMAL
            )
            
            # Setup weighted sampling for balanced classes
            counts = Counter(dataset.labels)
            class_weights = {cls: 1.0/count for cls, count in counts.items()}
            sample_weights = [class_weights[label] for label in dataset.labels]
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY if self.device.type == "cuda" else False,
            drop_last=True
        )
        
        self.dataset = dataset
        self.dataloader = dataloader
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Batches per epoch: {len(dataloader)}")
        
        return dataset, dataloader
    
    def setup_model(self):
        """Initialize model with paper's architecture"""
        model = ImprovedUNet(
            in_channels=IN_CHANNELS,
            base_channels=BASE_CHANNELS,
            channel_mults=CHANNEL_MULTS,
            num_heads=NUM_ATTENTION_HEADS,
            time_emb_dim=TIME_EMB_DIM
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        model = model.to(self.device)
        self.model = model
        
        return model
    
    def train(self, save_dir: str = "output"):
        """
        Main training loop implementing Algorithm 1 from the paper
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup optimizer (Adam with paper's hyperparameters)
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=OPTIMIZER_BETAS
        )
        
        # Loss function (MSE as in paper)
        mse_loss = nn.MSELoss()
        
        # Training tracking
        epoch_losses = []
        best_loss = float('inf')
        
        print(f"\\nStarting training for {self.num_epochs} epochs...")
        print("=" * 60)
        
        for epoch in tqdm(range(self.num_epochs), desc="Epochs"):
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            
            # Training loop over batches
            for x0, _ in tqdm(self.dataloader, desc="Batches", leave=False):
                x0 = x0.to(self.device)
                batch_size = x0.size(0)
                
                # Sample random timesteps for each sample in batch
                t = torch.randint(0, self.T, (batch_size,), device=self.device)
                
                # Apply forward diffusion to get noisy samples
                x_t, noise = self.forward_diffusion(x0, t)
                
                # Predict noise with model
                noise_pred = self.model(x_t, t)
                
                # Compute loss (MSE between predicted and true noise)
                loss = mse_loss(noise_pred, noise)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping (optional, not mentioned in paper but good practice)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Calculate average loss for epoch
            avg_loss = total_loss / num_batches
            epoch_losses.append(avg_loss)
            
            # Print progress
            tqdm.write(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.6f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'config': {
                        'T': self.T,
                        'beta_start': self.beta_start,
                        'beta_end': self.beta_end,
                        'num_epochs': self.num_epochs,
                        'batch_size': self.batch_size,
                        'learning_rate': self.learning_rate
                    }
                }, os.path.join(save_dir, "best_model.pth"))
            
            # Save checkpoint every 50 epochs
            if (epoch + 1) % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"))
        
        # Save final model
        torch.save(self.model.state_dict(), os.path.join(save_dir, "trained_model.pth"))
        
        # Save training history
        with open(os.path.join(save_dir, "training_history.json"), 'w') as f:
            json.dump({
                'epoch_losses': epoch_losses,
                'best_loss': best_loss,
                'total_epochs': self.num_epochs,
                'final_loss': epoch_losses[-1] if epoch_losses else None
            }, f, indent=2)
        
        # Plot and save training curve
        self.plot_training_curve(epoch_losses, save_dir)
        
        print(f"\\nTraining completed!")
        print(f"Best loss: {best_loss:.6f}")
        print(f"Final loss: {epoch_losses[-1]:.6f}")
        print(f"Model saved to: {os.path.join(save_dir, 'trained_model.pth')}")
        
        return epoch_losses
    
    def plot_training_curve(self, losses, save_dir):
        """Plot and save training loss curve"""
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(losses) + 1), losses, marker='o', linewidth=2, markersize=3)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("MSE Loss", fontsize=12)
        plt.title("DDPM Training Loss Curve", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale often better for loss curves
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(save_dir, "training_curve.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, "training_curve.pdf"), bbox_inches='tight')
        
        print(f"Training curve saved to: {os.path.join(save_dir, 'training_curve.png')}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Bearing Fault DDPM')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Path to CWRU dataset directory')
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Directory to save outputs')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                      help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                      help='Learning rate')
    parser.add_argument('--random_crop', action='store_true', default=True,
                      help='Use random crop dataset')
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    config_override = {
        'NUM_EPOCHS': args.epochs,
        'BATCH_SIZE': args.batch_size,
        'LEARNING_RATE': args.lr
    }
    
    # Initialize trainer
    print("Initializing DDPM Trainer...")
    trainer = DDPMTrainer(config_override)
    
    # Setup data
    print("\\nSetting up dataset...")
    trainer.setup_data(args.data_dir, use_random_crop=args.random_crop)
    
    # Setup model
    print("\\nInitializing model...")
    trainer.setup_model()
    
    # Start training
    print(f"\\nStarting training with:")
    print(f"  - Dataset: {args.data_dir}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Device: {trainer.device}")
    
    # Train model
    losses = trainer.train(args.output_dir)
    
    print("\\nTraining completed successfully!")


if __name__ == "__main__":
    main()
