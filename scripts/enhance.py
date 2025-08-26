import os
import yaml
import argparse
import torch

from audio_sr.models.unet import AudioUNet
from audio_sr.utils.audio_io import enhance_audio, create_downsampled_audio, visualize_waveforms

def main():
    parser = argparse.ArgumentParser(description="Enhance audio using trained model")
    parser.add_argument("--config", type=str, default="config/default.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input audio file")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to save enhanced audio")
    parser.add_argument("--create_low_quality", action="store_true",
                       help="Create low quality version of input if it's high quality")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = AudioUNet()
    
    # Load the model
    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Input/output paths
    input_path = args.input
    output_path = args.output
    
    # Create low quality version if requested
    if args.create_low_quality:
        print(f"Creating low quality version of {input_path}")
        low_quality_path = os.path.splitext(input_path)[0] + "_lowquality" + os.path.splitext(input_path)[1]
        original, low_quality = create_downsampled_audio(
            input_path, 
            low_quality_path, 
            target_sr=config['dataset']['sr_low']
        )
        
        # Visualize downsampled audio
        visualize_waveforms(original, low_quality, 
                           "Original High-Quality Audio", 
                           "Generated Low-Quality Audio")
        
        print(f"Low quality version saved to {low_quality_path}")
        
        # Use the low quality version as input
        input_path = low_quality_path
    
    # Enhance the audio
    print(f"Enhancing {input_path}")
    original, enhanced = enhance_audio(model, input_path, output_path, device)
    
    # Visualize the results
    visualize_waveforms(original, enhanced,
                       "Original/Low-Quality Audio",
                       "Enhanced Audio")
    
    print(f"Enhanced audio saved to {output_path}")
    
if __name__ == "__main__":
    main()