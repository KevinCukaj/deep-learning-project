import os
import yaml
import argparse
import torch
import random
import numpy as np
import pickle

from audio_sr.models.unet import AudioUNet
from audio_sr.data.dataset import create_dataset_splits
from audio_sr.training.trainer import AudioSRTrainer

def main():
    parser = argparse.ArgumentParser(description="Train Audio Super-Resolution model")
    parser.add_argument("--config", type=str, default="config/default.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint file to resume training")
    parser.add_argument("--dataset_splits", type=str, default=None,
                       help="Path to saved dataset splits")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds
    torch.manual_seed(config['training']['seed'])
    random.seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    
    # Check for GPU
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        print("No GPU available, using CPU")
    
    # Create dataset splits or load existing ones
    if args.dataset_splits and os.path.exists(args.dataset_splits):
        print(f"Loading dataset splits from {args.dataset_splits}")
        with open(args.dataset_splits, 'rb') as f:
            dataset_info = pickle.load(f)
        
        train_loader = torch.utils.data.DataLoader(
            dataset_info['train_dataset'],
            batch_size=config['training']['batch_size'],
            shuffle=True,
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            dataset_info['val_dataset'],
            batch_size=config['training']['batch_size'],
            shuffle=False,
            pin_memory=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            dataset_info['test_dataset'],
            batch_size=config['training']['batch_size'],
            shuffle=False,
            pin_memory=True
        )
    else:
        train_loader, val_loader, test_loader = create_dataset_splits(
            config['dataset']['root_dir'],
            batch_size=config['training']['batch_size'],
            train_ratio=config['training']['train_ratio'],
            val_ratio=config['training']['val_ratio'],
            test_ratio=config['training']['test_ratio'],
            segment_length=config['dataset']['segment_length'],
            sr_orig=config['dataset']['sr_orig'],
            sr_low=config['dataset']['sr_low'],
            num_workers=config['training']['num_workers'],
            seed=config['training']['seed'],
            max_files=config['dataset']['max_files']
        )
        
        # Save dataset splits for future use
        dataset_info = {
            'train_dataset': train_loader.dataset,
            'val_dataset': val_loader.dataset,
            'test_dataset': test_loader.dataset,
            'batch_size': train_loader.batch_size,
            'segment_length': config['dataset']['segment_length']
        }
        
        dataset_splits_path = os.path.join(config['dirs']['results_dir'], "dataset_splits.pkl")
        os.makedirs(os.path.dirname(dataset_splits_path), exist_ok=True)
        
        with open(dataset_splits_path, 'wb') as f:
            pickle.dump(dataset_info, f)
        
        print(f"Dataset splits saved to {dataset_splits_path}")
    
    print(f"Dataset ready: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    
    # Initialize model
    model = AudioUNet()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = AudioSRTrainer(model, train_loader, val_loader, config)
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train the model
    trainer.train(config['training']['num_epochs'])
    
    print("Training completed.")
    
if __name__ == "__main__":
    main()