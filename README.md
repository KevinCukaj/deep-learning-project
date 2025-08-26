# deep-learning-project

Deep Learning project for the Sapienza's DL course, held by Emanuele Rodolà.

## Abstract

Deep learning approach for audio super-resolution. We implement a U-Net architecture with skip connections that directly operates on raw waveforms. Our method captures both temporal and frequency characteristics of audio signals, effectively reconstructing the high-frequency components lost in low-resolution recordings. Experimental results demonstrate improvements in both objective metrics and subjective quality when applied to music samples from the Free Music Archive (FMA) dataset.

## Requirements

- [uv](https://github.com/astral-sh/uv)

## Usage

### Configuration

Edit the `config/default.yaml` file to adjust parameters:

```yaml
# Dataset configuration
dataset:
    root_dir: "/path/to/your/dataset"  # Change this to your dataset location
    sr_orig: 44100
    sr_low: 16000
  # ...
```

### Demo

```bash
uv run scripts/demo.py
```

### Downloading the dataset

```bash
uv run scripts/dataset.py
```

### Training

```bash
uv run scripts/train.py --config config/default.yaml
```

To resume training from a checkpoint:

```bash
uv run scripts/train.py --config config/default.yaml --checkpoint checkpoints/model_checkpoint_epoch_30.pt
```

### Evaluation

```bash
uv run scripts/evaluate.py --config config/default.yaml --model checkpoints/best_model.pt --output results/evaluation.json
```

### Enhancement

To enhance a low-quality audio file:

```bash
uv run scripts/enhance.py --model checkpoints/best_model.pt --input audio/lowquality.mp3 --output audio/enhanced.wav
```

To create a low-quality version and then enhance it:

```bash
uv run scripts/enhance.py --model checkpoints/best_model.pt --input audio/original.mp3 --output audio/enhanced.wav --create_low_quality
```

## Dataset

This project uses the [fma_small](https://github.com/mdeff/fma.git) dataset
