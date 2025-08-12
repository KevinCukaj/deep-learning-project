import demucs
import demucs.separate
import librosa
from scipy.signal import butter, lfilter
import soundfile as sf

CUTOFF_FREQ = 4000.0
ORIGINAL_SAMPLE_RATE = 44100
TARGET_SAMPLE_RATE = 16000

def butter_lowpass(cutoff, sr):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(5, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_low_pass(y):
    b, a = butter_lowpass(CUTOFF_FREQ, TARGET_SAMPLE_RATE)
    y_filtered = lfilter(b, a, y)
    return y_filtered

def load_and_downsample(track_path):
    # Load full-quality music
    y, orig_sr = librosa.load(track_path, sr=ORIGINAL_SAMPLE_RATE)
    # Simulate low-quality version (e.g., for BWE or SR)
    y_low_rate = librosa.resample(y, orig_sr=orig_sr, target_sr=TARGET_SAMPLE_RATE)
    return y_low_rate

def main():
    # separate source in bass, drums, other, vocals
    # the pre-trained model is mdx_extra
    # demucs.separate.main(["--mp3", "-n", "mdx_extra", "./test-tracks/track1.mp3"])

    train_dir  = "/home/kevin/Documents/deep-learning-project/test-tracks"
    track_path = "./test-tracks/track1.mp3"

    # Downsample
    y_low_rate = load_and_downsample(track_path)

    # Apply the filter (allowing low frequencies to pass through while attenuating higher frequencies)
    y_low_rate_low_freq = apply_low_pass(y_low_rate)

    # Save the low-quality
    sf.write("./test-tracks/track_LR_LF.mp3", y_low_rate_low_freq, TARGET_SAMPLE_RATE)

    # 1. Apply HiFi-GAN model to upsample again

    # 2. Apply a model to bandwidth extension

if __name__ == "__main__":
    main()
