"""Regression tests for native TorchCodec audio decoding and Kaldi feature scaling."""

import math
import wave
from pathlib import Path

import pytest
import torch
import torchaudio.compliance.kaldi as kaldi
from torchcodec.decoders import AudioDecoder

from chunkformer.dataset.processor import decode_wav


@pytest.fixture
def wav_path(tmp_path: Path) -> Path:
    path = tmp_path / "stereo.wav"
    frames = bytearray()
    for i in range(8000):
        left = int(12000 * math.sin(2 * math.pi * 440 * i / 8000))
        right = int(6000 * math.sin(2 * math.pi * 440 * i / 8000))
        frames.extend(left.to_bytes(2, "little", signed=True))
        frames.extend(right.to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(frames)
    return path


def test_decode_wav_path_and_bytes(wav_path: Path):
    for source in (str(wav_path), wav_path.read_bytes()):
        sample = decode_wav({"key": "clip", "wav": source})
        assert sample["sample_rate"] == 8000
        assert sample["wav"].shape == (2, 8000)
        assert sample["wav"].dtype == torch.float32
        assert sample["wav"].abs().max() <= 1.0


def test_decode_wav_range(wav_path: Path):
    sample = decode_wav(
        {"key": "clip", "wav": wav_path.read_bytes(), "start": 0.25, "end": 0.75}
    )
    assert sample["sample_rate"] == 8000
    assert sample["wav"].shape == (2, 4000)


def test_mono_resampling_and_kaldi_pcm_scaling(wav_path: Path):
    decoded = AudioDecoder(
        str(wav_path), sample_rate=16000, num_channels=1
    ).get_all_samples()
    assert decoded.sample_rate == 16000
    assert decoded.data.shape == (1, 16000)
    assert decoded.data.dtype == torch.float32

    pcm_scale = decoded.data * (1 << 15)
    features = kaldi.fbank(
        pcm_scale,
        sample_frequency=16000,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
    )
    assert features.shape == (98, 80)
    assert torch.isfinite(features).all()
