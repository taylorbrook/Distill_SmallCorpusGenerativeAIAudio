"""Quick diagnostic: compare decoder output range vs training data mel range."""
import torch
from distill.audio.spectrogram import AudioSpectrogram
from distill.audio.io import load_audio
from distill.audio.validation import collect_audio_files
from pathlib import Path

spec = AudioSpectrogram()

data_dir = Path("data/datasets/imported")
files = collect_audio_files(data_dir)[:10]

print("--- Training data mel stats (peak-normalized 1s chunks) ---")
all_maxes = []
for f in files:
    af = load_audio(f, target_sample_rate=48000)
    wav = af.waveform.float().unsqueeze(0) if isinstance(af.waveform, torch.Tensor) else torch.from_numpy(af.waveform).float().unsqueeze(0)  # [1, samples]
    wav = wav[:, :48000]
    peak = wav.abs().max()
    if peak > 1e-6:
        wav = wav / peak
    mel = spec.waveform_to_mel(wav.unsqueeze(0))
    m_max = mel.max().item()
    m_mean = mel.mean().item()
    above_5 = (mel > 5).float().mean().item() * 100
    linear = torch.expm1(mel.clamp(min=0))
    print(
        f"  {f.name[:30]:30s} log1p max={m_max:.3f} mean={m_mean:.3f}"
        f" >5: {above_5:.1f}% | linear max={linear.max().item():.1f}"
    )
    all_maxes.append(m_max)

print(f"\nTraining data max: {max(all_maxes):.3f}")
print(f"Linear equivalent: {torch.expm1(torch.tensor(max(all_maxes))).item():.1f}")
print(f"\nDecoder outputs max up to 10.66 -> linear {torch.expm1(torch.tensor(10.66)).item():.1f}")
print(f"That's {torch.expm1(torch.tensor(10.66)).item() / torch.expm1(torch.tensor(max(all_maxes))).item():.0f}x the training data max")
