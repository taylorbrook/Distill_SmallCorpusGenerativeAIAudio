"""Diagnostic: test audio quality at each stage of the pipeline.

Outputs:
1. diag_original.wav       - raw training audio chunk (peak normalized)
2. diag_roundtrip_mel.wav  - waveform -> mel -> waveform (no VAE, just mel+griffinlim)
3. diag_roundtrip_vae.wav  - waveform -> mel -> VAE encode -> decode -> waveform
4. diag_generated.wav      - random latent -> decode -> waveform
"""
import torch
import numpy as np
import soundfile as sf
from distill.audio.spectrogram import AudioSpectrogram
from distill.audio.io import load_audio
from distill.audio.validation import collect_audio_files
from distill.models.vae import ConvVAE
from pathlib import Path


def save_wav(path, audio_np, sr=48000):
    """Save 1-D float32 numpy array as WAV."""
    sf.write(path, audio_np, sr, format="WAV", subtype="PCM_24")


device = torch.device("cuda")
spec = AudioSpectrogram()
spec.to(device)

# Load a training file - take mono
data_dir = Path("data/datasets/imported")
files = collect_audio_files(data_dir)
f = files[0]
print(f"Source: {f.name}")

af = load_audio(f, target_sample_rate=48000)
wav = af.waveform
if isinstance(wav, torch.Tensor):
    wav = wav.float()
else:
    wav = torch.from_numpy(wav).float()

# Mono: take first channel if stereo
if wav.ndim == 2:
    wav = wav[0]
wav = wav[:48000]  # 1 second

# Peak normalize like training does
peak = wav.abs().max()
if peak > 1e-6:
    wav = wav / peak

save_wav("diag_original.wav", wav.numpy())
print("Saved diag_original.wav")

# Test 1: mel roundtrip (no VAE) - isolates Griffin-Lim quality
wav_for_mel = wav.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 48000]
mel = spec.waveform_to_mel(wav_for_mel)
print(f"Mel shape: {mel.shape}, range: [{mel.min():.3f}, {mel.max():.3f}]")

wav_roundtrip = spec.mel_to_waveform(mel)
wav_rt_np = wav_roundtrip.squeeze().cpu().numpy()
pk = np.abs(wav_rt_np).max()
if pk > 1e-6:
    wav_rt_np = wav_rt_np / pk * 0.89
save_wav("diag_roundtrip_mel.wav", wav_rt_np)
print("Saved diag_roundtrip_mel.wav (mel+griffinlim only, no VAE)")

# Load model
model_dir = Path("data/models")
models = sorted(model_dir.glob("*.distill"))
model_path = models[-1]
print(f"Loading model: {model_path.name}")

checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
latent_dim = checkpoint.get("latent_dim", 64)
model = ConvVAE(latent_dim=latent_dim).to(device)

mel_shape = spec.get_mel_shape(48000)
dummy = torch.randn(1, 1, mel_shape[0], mel_shape[1]).to(device)
with torch.no_grad():
    model(dummy)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Test 2: VAE roundtrip
with torch.no_grad():
    recon, mu, logvar = model(mel)
    print(f"Recon range: [{recon.min():.3f}, {recon.max():.3f}]")
    print(f"MSE: {((recon - mel) ** 2).mean().item():.4f}")
    wav_vae = spec.mel_to_waveform(recon)
    wav_vae_np = wav_vae.squeeze().cpu().numpy()
    pk = np.abs(wav_vae_np).max()
    if pk > 1e-6:
        wav_vae_np = wav_vae_np / pk * 0.89
    save_wav("diag_roundtrip_vae.wav", wav_vae_np)
    print("Saved diag_roundtrip_vae.wav (full VAE roundtrip)")

# Test 3: random generation
with torch.no_grad():
    z = torch.randn(1, latent_dim).to(device)
    gen_mel = model.decode(z, target_shape=mel_shape)
    print(f"Generated mel range: [{gen_mel.min():.3f}, {gen_mel.max():.3f}]")
    wav_gen = spec.mel_to_waveform(gen_mel)
    wav_gen_np = wav_gen.squeeze().cpu().numpy()
    pk = np.abs(wav_gen_np).max()
    if pk > 1e-6:
        wav_gen_np = wav_gen_np / pk * 0.89
    save_wav("diag_generated.wav", wav_gen_np)
    print("Saved diag_generated.wav (random latent generation)")

print("\nListen to compare where artifacts appear:")
print("  1. diag_original.wav       - source audio (clean)")
print("  2. diag_roundtrip_mel.wav  - mel+griffinlim only (Griffin-Lim quality ceiling)")
print("  3. diag_roundtrip_vae.wav  - VAE encode+decode (adds VAE error on top)")
print("  4. diag_generated.wav      - random generation (novel content)")
