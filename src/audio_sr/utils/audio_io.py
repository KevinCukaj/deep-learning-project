import torch
import torchaudio
import os
import matplotlib.pyplot as plt

def enhance_audio(model, input_audio_path, output_audio_path, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()

    # Load and preprocess the input audio
    waveform, sample_rate = torchaudio.load(input_audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to the original high sampling rate if necessary
    if sample_rate != 44100:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 44100)

    # Normalize
    waveform = waveform / torch.max(torch.abs(waveform))

    # Process in chunks to avoid memory issues
    chunk_size = 65536  # Adjust based on available memory
    enhanced_chunks = []
    total_chunks = (waveform.shape[1] + chunk_size - 1) // chunk_size

    for i in range(0, waveform.shape[1], chunk_size):
        chunk_num = i // chunk_size + 1
        chunk = waveform[:, i:i+chunk_size]

        # Pad if necessary
        if chunk.shape[1] < chunk_size:
            padding = chunk_size - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, padding))

        # Move to device
        chunk = chunk.to(device)

        # Enhance
        with torch.no_grad():
            enhanced_chunk = model(chunk)

        # Remove padding if added
        if i + chunk_size > waveform.shape[1]:
            original_length = waveform.shape[1] - i
            enhanced_chunk = enhanced_chunk[:, :original_length]

        # Add to list
        enhanced_chunks.append(enhanced_chunk.cpu())

    # Concatenate chunks
    enhanced_waveform = torch.cat(enhanced_chunks, dim=1)

    # Ensure the tensor is 2D [channels, time] before saving
    if enhanced_waveform.dim() != 2:
        if enhanced_waveform.dim() > 2:
            # If it has more than 2 dimensions, flatten all but the first dimension
            enhanced_waveform = enhanced_waveform.reshape(enhanced_waveform.shape[0], -1)
        elif enhanced_waveform.dim() == 1:
            # If it's 1D, add a channel dimension
            enhanced_waveform = enhanced_waveform.unsqueeze(0)

    # Save the enhanced audio
    os.makedirs(os.path.dirname(output_audio_path) if os.path.dirname(output_audio_path) else '.', exist_ok=True)
    torchaudio.save(output_audio_path, enhanced_waveform, 44100)

    # Return waveforms for visualization
    return waveform.cpu(), enhanced_waveform


def create_downsampled_audio(input_path, output_path, target_sr=16000):
    # Load the audio
    waveform, sample_rate = torchaudio.load(input_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 1. Downsample to low SR
    waveform_low = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    
    # 2. Upsample back to original SR
    waveform_low = torchaudio.functional.resample(waveform_low, target_sr, 44100)
    
    # 3. Apply low-pass filter to simulate bandwidth limitation
    cutoff_freq = target_sr / 2 * 0.6  # 60% of Nyquist frequency for the low sample rate
    waveform_low = torchaudio.functional.lowpass_biquad(
        waveform_low,
        44100,
        cutoff_freq
    )
    
    # Save the low-quality audio
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    torchaudio.save(output_path, waveform_low, 44100)
    
    return waveform, waveform_low


def visualize_waveforms(original, enhanced, title1="Original Audio", title2="Enhanced Audio"):
    plt.figure(figsize=(15, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(original.numpy()[0])
    plt.title(title1)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    plt.subplot(2, 1, 2)
    plt.plot(enhanced.numpy()[0])
    plt.title(title2)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()