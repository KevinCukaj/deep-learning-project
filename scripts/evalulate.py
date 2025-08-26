#!/usr/bin/env python3
import os
import yaml
import argparse
import torch
import pickle
import json
from datetime import datetime

from audio_sr.models.unet import AudioUNet
from audio_sr.data.dataset import create_dataset_splits
from audio_sr.training.metrics import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Evaluate Audio Super-Resolution model")
    parser.add_argument("--config", type=str, default="config/default.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--dataset_splits", type=str, default=None,
                       help="Path to saved dataset splits")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save evaluation results")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    if args.dataset_splits and os.path.exists(args.dataset_splits):
        print(f"Loading dataset splits from {args.dataset_splits}")
        with open(args.dataset_splits, 'rb') as f:
            dataset_info = pickle.load(f)
        
        test_loader = torch.utils.data.DataLoader(
            dataset_info['test_dataset'],
            batch_size=config['training']['batch_size'],
            shuffle=False,
            pin_memory=True
        )
    else:
        # Create dataset with same settings but only need test loader
        _, _, test_loader = create_dataset_splits(
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
    
    # Initialize model
    model = AudioUNet()
    
    # Load the model
    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Evaluate the model
    results = evaluate_model(model, test_loader, device)
    
    # Save results if output path is provided
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        results_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": args.model,
            "test_dataset_size": len(test_loader.dataset),
            "metrics": results
        }
        
        with open(args.output, 'w') as f:
            json.dump(results_data, f, indent=4)
            
        print(f"Evaluation results saved to {args.output}")
    
if __name__ == "__main__":
    main()