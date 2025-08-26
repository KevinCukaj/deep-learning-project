import torch
import torch.nn.functional as F
from tqdm import tqdm
from audio_sr.models.losses import STFTLoss

def evaluate_model(model, test_loader, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()

    # Initialize STFT loss for evaluation
    stft_criterion = STFTLoss(
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=None,
        window='hann'
    ).to(device)

    # Metrics
    total_l1_loss = 0.0
    total_mse_loss = 0.0
    total_stft_loss = 0.0

    with torch.no_grad():
        for i, (low_res, high_res) in enumerate(tqdm(test_loader, desc="Evaluating")):
            low_res, high_res = low_res.to(device), high_res.to(device)

            # Forward pass
            outputs = model(low_res)

            # Ensure correct dimensions for waveform processing
            outputs_waveform = outputs.squeeze(1) if outputs.dim() > 2 else outputs
            high_res_waveform = high_res.squeeze(1) if high_res.dim() > 2 else high_res

            # Calculate metrics
            l1_loss = F.l1_loss(outputs_waveform, high_res_waveform)
            mse_loss = F.mse_loss(outputs_waveform, high_res_waveform)
            stft_loss = stft_criterion(outputs_waveform, high_res_waveform)

            total_l1_loss += l1_loss.item()
            total_mse_loss += mse_loss.item()
            total_stft_loss += stft_loss.item()

    # Calculate average metrics
    avg_l1_loss = total_l1_loss / len(test_loader)
    avg_mse_loss = total_mse_loss / len(test_loader)
    avg_stft_loss = total_stft_loss / len(test_loader)

    print(f"Test Results:")
    print(f"- Average L1 Loss: {avg_l1_loss:.4f}")
    print(f"- Average MSE Loss: {avg_mse_loss:.4f}")
    print(f"- Average STFT Loss: {avg_stft_loss:.4f}")

    return {
        'l1_loss': avg_l1_loss,
        'mse_loss': avg_mse_loss,
        'stft_loss': avg_stft_loss
    }