import torch
import torch.nn as nn

class STFTLoss(nn.Module):
    def __init__(self, fft_sizes=[512, 1024, 2048, 4096], 
                 hop_sizes=[128, 256, 512, 1024], 
                 win_lengths=[512, 1024, 2048, 4096], 
                 window='hann'):
        super(STFTLoss, self).__init__()

        # Use multiple FFT sizes for multi-resolution analysis
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths if win_lengths else fft_sizes

        # Register windows for each FFT size
        for i in range(len(self.fft_sizes)):
            self.register_buffer(f'window_{i}', torch.hann_window(self.win_lengths[i]))

    def stft(self, x, fft_size, hop_size, win_length, window):
        # Handle different PyTorch versions
        if hasattr(torch, 'stft'):
            try:
                # For PyTorch 1.7+
                return torch.stft(
                    x, fft_size, hop_size, win_length, window,
                    return_complex=True)
            except TypeError:
                # For older PyTorch versions
                stft_result = torch.stft(
                    x, fft_size, hop_size, win_length, window)
                real, imag = stft_result.unbind(-1)
                return torch.complex(real, imag)
        else:
            raise RuntimeError("Current PyTorch version doesn't support torch.stft")

    def compute_spectral_convergence(self, x_mag, y_mag):
        return torch.norm(x_mag - y_mag, p='fro') / torch.norm(y_mag, p='fro')

    def compute_magnitude_loss(self, x_mag, y_mag):
        return torch.mean(torch.abs(x_mag - y_mag))

    def forward(self, x, y):
        # Ensure inputs are the same shape
        if x.size() != y.size():
            raise ValueError(f"Inputs must have same size, got {x.size()} and {y.size()}")

        # Ensure inputs are 2D (batch, samples)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        # Initialize losses
        sc_loss = 0.0
        mag_loss = 0.0

        # Compute loss for each FFT size
        for i in range(len(self.fft_sizes)):
            x_stft = self.stft(x, self.fft_sizes[i], self.hop_sizes[i],
                               self.win_lengths[i], getattr(self, f'window_{i}'))
            y_stft = self.stft(y, self.fft_sizes[i], self.hop_sizes[i],
                               self.win_lengths[i], getattr(self, f'window_{i}'))

            # Compute magnitudes
            x_mag = torch.abs(x_stft)
            y_mag = torch.abs(y_stft)

            # Accumulate losses
            sc_loss += self.compute_spectral_convergence(x_mag, y_mag)
            mag_loss += self.compute_magnitude_loss(x_mag, y_mag)

        # Average over number of FFT sizes
        sc_loss = sc_loss / len(self.fft_sizes)
        mag_loss = mag_loss / len(self.fft_sizes)

        # Total loss (weighted sum)
        loss = sc_loss + mag_loss

        return loss