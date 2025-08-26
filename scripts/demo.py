import os
import sys
import argparse
import torch

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_sr.models.unet import AudioUNet
from audio_sr.utils.audio_io import enhance_audio, create_downsampled_audio, visualize_waveforms

def main():
    argparser = argparse.ArgumentParser(description="Audio Super-Resolution Demo")
    argparser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt", help="Path to the trained model")
    argparser.add_argument("--input_path", type=str, default="samples/sample.mp3", help="Path to the input audio file")
    argparser.add_argument("--low_quality_path", type=str, default="samples/sample_lowquality.mp3", help="Destination path for the low-quality audio file")
    argparser.add_argument("--output_path", type=str, default="results/sample_enhanced.wav", help="Destionation path for the output audio file")
    args = argparser.parse_args()

    # Set paths
    model_path = args.model_path
    input_path = args.input_path
    low_quality_path = args.low_quality_path
    output_path = args.output_path
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train a model first or specify the correct path.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize model
    model = AudioUNet()
    
    # Load the model
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Create low quality version
    print(f"Creating low quality version of {input_path}")
    original, low_quality = create_downsampled_audio(input_path, low_quality_path)
    
    # Visualize the downsampled audio
    visualize_waveforms(original, low_quality, 
                       "Original High-Quality Audio", 
                       "Generated Low-Quality Audio")
    
    # Enhance the audio
    print(f"Enhancing {low_quality_path}")
    original, enhanced = enhance_audio(model, low_quality_path, output_path)
    
    # Visualize the results
    visualize_waveforms(original, enhanced,
                       "Low-Quality Audio",
                       "Enhanced Audio")
    
    print(f"Enhanced audio saved to {output_path}")
    
if __name__ == "__main__":
    main()