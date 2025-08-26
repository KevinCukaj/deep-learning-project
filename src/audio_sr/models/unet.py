import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioUNet(nn.Module):
    def __init__(self):
        super(AudioUNet, self).__init__()

        # Encoder
        self.enc1 = nn.Conv1d(1, 64, kernel_size=15, stride=1, padding=7)
        self.enc2 = nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7)
        self.enc3 = nn.Conv1d(128, 256, kernel_size=15, stride=2, padding=7)
        self.enc4 = nn.Conv1d(256, 512, kernel_size=15, stride=2, padding=7)

        # Decoder
        self.dec4 = nn.ConvTranspose1d(512, 256, kernel_size=16, stride=2, padding=7)
        self.dec3 = nn.ConvTranspose1d(512, 128, kernel_size=16, stride=2, padding=7)
        self.dec2 = nn.ConvTranspose1d(256, 64, kernel_size=16, stride=2, padding=7)
        self.dec1 = nn.Conv1d(128, 1, kernel_size=15, stride=1, padding=7)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.bn_dec4 = nn.BatchNorm1d(256)
        self.bn_dec3 = nn.BatchNorm1d(128)
        self.bn_dec2 = nn.BatchNorm1d(64)

    def forward(self, x):
        # Add channel dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Encoder
        e1 = F.leaky_relu(self.bn1(self.enc1(x)), 0.2)
        e2 = F.leaky_relu(self.bn2(self.enc2(e1)), 0.2)
        e3 = F.leaky_relu(self.bn3(self.enc3(e2)), 0.2)
        e4 = F.leaky_relu(self.bn4(self.enc4(e3)), 0.2)

        # Decoder with skip connections
        d4 = F.relu(self.bn_dec4(self.dec4(e4)))
        d4 = torch.cat([d4, e3], dim=1)  # Skip connection

        d3 = F.relu(self.bn_dec3(self.dec3(d4)))
        d3 = torch.cat([d3, e2], dim=1)  # Skip connection

        d2 = F.relu(self.bn_dec2(self.dec2(d3)))
        d2 = torch.cat([d2, e1], dim=1)  # Skip connection

        d1 = torch.tanh(self.dec1(d2))

        return d1