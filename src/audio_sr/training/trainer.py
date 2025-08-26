import os
import gc
import json
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from audio_sr.models.losses import STFTLoss

class AudioSRTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = self.model.to(self.device)
        
        # Setup loss function
        self.criterion = STFTLoss(
            fft_sizes=config['stft_loss']['fft_sizes'],
            hop_sizes=config['stft_loss']['hop_sizes'],
            win_lengths=config['stft_loss']['win_lengths'],
            window=config['stft_loss']['window']
        ).to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=5, factor=0.5
        )
        
        # Setup training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        # Create output directories
        os.makedirs(config['dirs']['results_dir'], exist_ok=True)
        os.makedirs(config['dirs']['checkpoint_dir'], exist_ok=True)
    
    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.start_epoch = checkpoint['epoch'] + 1
            
            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint.get('train_losses', [])
            if 'val_losses' in checkpoint:
                self.val_losses = checkpoint.get('val_losses', [])
            
            if 'best_val_loss' in checkpoint:
                self.best_val_loss = checkpoint['best_val_loss']
            elif 'val_loss' in checkpoint:
                self.best_val_loss = checkpoint['val_loss']
            
            if 'learning_rates' in checkpoint:
                self.learning_rates = checkpoint['learning_rates']
            
            print(f"Resumed training from epoch {self.start_epoch}")
            print(f"Previous best validation loss: {self.best_val_loss:.4f}")
            
            # Update scheduler with last validation loss
            if len(self.val_losses) > 0:
                self.scheduler.step(self.val_losses[-1])
            
            return True
        else:
            print(f"No checkpoint found at {checkpoint_path}")
            return False
    
    def train(self, num_epochs):
        grad_accum_steps = self.config['training']['gradient_accumulation_steps']
        
        for epoch in range(self.start_epoch, num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            self.optimizer.zero_grad()
            
            for i, (low_res, high_res) in enumerate(tqdm(self.train_loader, 
                                                        desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
                low_res, high_res = low_res.to(self.device), high_res.to(self.device)
                
                # Forward pass
                outputs = self.model(low_res)
                
                # Apply STFT loss - ensure outputs and targets are properly shaped
                outputs = outputs.squeeze(1) if outputs.dim() > 2 else outputs
                high_res = high_res.squeeze(1) if high_res.dim() > 2 else high_res
                
                loss = self.criterion(outputs, high_res) / grad_accum_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights after accumulating gradients
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(self.train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                train_loss += loss.item() * grad_accum_steps
                
                # Clear memory periodically
                if i % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            train_loss = train_loss / len(self.train_loader)
            self.train_losses.append(train_loss)
            
            # Validation phase
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Track learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Adjust learning rate
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(os.path.join(self.config['dirs']['checkpoint_dir'], 'best_model.pt'), 
                                     epoch, train_loss, val_loss)
                print(f"Saved new best model with validation loss: {val_loss:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 3 == 0:
                self.save_checkpoint(
                    os.path.join(self.config['dirs']['checkpoint_dir'], f'model_checkpoint_epoch_{epoch+1}.pt'),
                    epoch, train_loss, val_loss
                )
                
                # Plot training curves
                self.plot_training_curves(epoch)
                
                # Save metrics
                self.save_metrics()
                
                # Garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Save final model
        torch.save(self.model.state_dict(), 
                   os.path.join(self.config['dirs']['checkpoint_dir'], 'audio_sr_final_model.pt'))
        
        # Plot final training curves
        self.plot_training_curves(num_epochs-1, final=True)
        
        print(f"Training completed! Best validation loss: {self.best_val_loss:.4f}")
        
        return self.train_losses, self.val_losses
    
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for low_res, high_res in tqdm(self.val_loader, desc="Validation"):
                low_res, high_res = low_res.to(self.device), high_res.to(self.device)
                
                outputs = self.model(low_res)
                
                # Apply STFT loss - ensure outputs and targets are properly shaped
                outputs = outputs.squeeze(1) if outputs.dim() > 2 else outputs
                high_res = high_res.squeeze(1) if high_res.dim() > 2 else high_res
                
                loss = self.criterion(outputs, high_res)
                
                val_loss += loss.item()
        
        val_loss = val_loss / len(self.val_loader)
        return val_loss
    
    def save_checkpoint(self, checkpoint_path, epoch, train_loss, val_loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
    
    def plot_training_curves(self, epoch, final=False):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        epochs_range = list(range(1, len(self.train_losses) + 1))
        ax1.plot(epochs_range, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs_range, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('STFT Loss')
        
        if final:
            ax1.set_title('Training and Validation Loss (Final)')
        else:
            ax1.set_title(f'Training and Validation Loss (Epoch {epoch+1})')
            
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Add annotation for best validation loss
        if self.val_losses:
            min_val_epoch = self.val_losses.index(min(self.val_losses)) + 1
            min_val_loss = min(self.val_losses)
            ax1.annotate(f'Best: {min_val_loss:.4f}',
                         xy=(min_val_epoch, min_val_loss),
                         xytext=(min_val_epoch, min_val_loss * 1.1),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                         fontsize=10)
        
        # Learning rate plot
        ax2.plot(epochs_range, self.learning_rates, 'g-')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        if final:
            fig.savefig(os.path.join(self.config['dirs']['results_dir'], 'final_training_plots.png'))
        else:
            fig.savefig(os.path.join(self.config['dirs']['results_dir'], 'loss_curves.png'))
        
        plt.close()
    
    def save_metrics(self):
        metrics_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epochs_completed": len(self.train_losses),
            "best_val_loss": min(self.val_losses) if self.val_losses else None,
            "best_epoch": self.val_losses.index(min(self.val_losses)) + 1 if self.val_losses else None,
            "final_learning_rate": self.learning_rates[-1] if self.learning_rates else None,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates
        }
        
        metrics_file = os.path.join(self.config['dirs']['results_dir'], "metrics.json")
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=4)
            print(f"Training metrics saved to {metrics_file}")
        except Exception as e:
            print(f"Error saving metrics to {metrics_file}: {e}")