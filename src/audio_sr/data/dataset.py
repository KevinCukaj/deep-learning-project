import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class AudioSuperResolutionDataset(Dataset):
    def __init__(self, root_dir, segment_length=16384, sr_orig=44100, sr_low=16000, max_files=None):
        self.root_dir = root_dir
        self.segment_length = segment_length
        self.sr_orig = sr_orig
        self.sr_low = sr_low

        # Recursively find all mp3 files in the directory structure
        self.file_list = []
        print(f"Scanning for MP3 files in {root_dir} (recursive)...")

        # Walk through all subdirectories
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.mp3'):
                    full_path = os.path.join(dirpath, filename)
                    self.file_list.append(full_path)

        print(f"Found {len(self.file_list)} MP3 files")

        # Shuffle the file list to ensure randomness across folders
        random.shuffle(self.file_list)

        # Limit the number of files if specified
        if max_files is not None:
            self.file_list = self.file_list[:max_files]
            print(f"Limited dataset to {len(self.file_list)} files")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]

        # Load audio file
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # If there's an error, return a random valid index instead
            return self.__getitem__(random.randint(0, len(self.file_list) - 1))

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != self.sr_orig:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sr_orig)

        # Randomly select segment
        if waveform.shape[1] > self.segment_length:
            start_idx = torch.randint(0, waveform.shape[1] - self.segment_length, (1,))
            waveform = waveform[:, start_idx:start_idx + self.segment_length]
        else:
            # Pad if audio is shorter than segment_length
            padding = self.segment_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Create low resolution version with proper low-pass filtering
        # 1. Downsample to low SR
        waveform_low = torchaudio.functional.resample(waveform, self.sr_orig, self.sr_low)
        # 2. Upsample back to original SR
        waveform_low = torchaudio.functional.resample(waveform_low, self.sr_low, self.sr_orig)

        # 3. Apply low-pass filter to simulate bandwidth limitation
        cutoff_freq = self.sr_low / 2 * 0.6  # 60% of Nyquist frequency for the low sample rate
        waveform_low = torchaudio.functional.lowpass_biquad(
            waveform_low,
            self.sr_orig,
            cutoff_freq
        )

        # Ensure both waveforms have the same length after resampling and filtering
        if waveform_low.shape[1] != self.segment_length:
             if waveform_low.shape[1] > self.segment_length:
                waveform_low = waveform_low[:, :self.segment_length]
             else:
                padding = self.segment_length - waveform_low.shape[1]
                waveform_low = torch.nn.functional.pad(waveform_low, (0, padding))

        # Normalize
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        waveform_low = waveform_low / (torch.max(torch.abs(waveform_low)) + 1e-8)

        return waveform_low.squeeze(0), waveform.squeeze(0)


def create_dataset_splits(dataset_path, batch_size=8, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                          segment_length=16384, sr_orig=44100, sr_low=16000, num_workers=2, seed=42,
                          max_files=1000):
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)

    # Create the dataset with recursive file search and file limit
    full_dataset = AudioSuperResolutionDataset(
        dataset_path,
        segment_length=segment_length,
        sr_orig=sr_orig,
        sr_low=sr_low,
        max_files=max_files
    )

    # Split the dataset
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    print(f"Dataset split: {train_size} training, {val_size} validation, {test_size} test samples")

    # Create a generator for reproducible splits
    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Create data loaders with worker initialization to ensure proper randomization
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        import numpy as np
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader