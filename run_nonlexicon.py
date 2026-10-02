# -*- coding: utf-8 -*-
"""Nonlexicon labeler (CVNT branch) standalone runner.

Loads nonlexicon_labeler.onnx + vocab.json + config.json from this folder,
labels non-lexical phonemes (e.g. AP/EP breath sounds) frame by frame
in an audio file, and optionally plots the result.

Usage:
    python run_nonlexicon.py <audio_file> [--threshold 0.5] [--plot out.png]
"""
import argparse
import json
import pathlib

import numpy as np
import onnxruntime as ort

HERE = pathlib.Path(__file__).parent


def load_audio(path: str, target_sr: int) -> np.ndarray:
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != target_sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def to_intervals(prob: np.ndarray, frame_duration: float,
                 threshold: float = 0.5, max_gap: int = 5,
                 min_frames: int = 10) -> list[tuple[float, float, float]]:
    """Convert a per-frame probability curve into time intervals.

    Returns a list of (start_sec, end_sec, confidence).
    """
    frames = np.where(prob >= threshold)[0]
    if len(frames) == 0:
        return []
    groups, start, prev = [], frames[0], frames[0]
    for f in frames[1:]:
        if f - prev > max_gap:
            groups.append((start, prev))
            start = f
        prev = f
    groups.append((start, prev))

    intervals = []
    for s, e in groups:
        seg = prob[s:e + 1]
        if len(seg) < min_frames:
            continue
        conf = float(np.mean(seg))
        intervals.append((s * frame_duration, (e + 1) * frame_duration, conf))
    return intervals


def main():
    parser = argparse.ArgumentParser(description="Run standalone nonlexicon labeler")
    parser.add_argument("audio", help="input audio file (wav/mp3/flac...)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="probability threshold for a frame to count (default 0.5)")
    parser.add_argument("--min-frames", type=int, default=10,
                        help="minimum interval length in frames (default 10)")
    parser.add_argument("--plot", metavar="PNG", default=None,
                        help="save a probability curve plot to this png")
    args = parser.parse_args()

    with open(HERE / "vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(HERE / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    class_names = ["None", *vocab["non_lexical_phonemes"]]
    mel_cfg = config["mel_spec_config"]
    frame_duration = mel_cfg["hop_size"] / mel_cfg["sample_rate"]

    wav = load_audio(args.audio, mel_cfg["sample_rate"])

    so = ort.SessionOptions()
    session = ort.InferenceSession(
        str(HERE / "nonlexicon_labeler.onnx"), so,
        providers=["CPUExecutionProvider"],
    )
    logits = session.run(["cvnt_logits"], {"waveform": wav[None, :]})[0]  # [1, C, T]

    # trim to actual audio length
    wav_length = len(wav) / mel_cfg["sample_rate"]
    num_frames = int((wav_length * mel_cfg["sample_rate"] + 0.5) / mel_cfg["hop_size"])
    probs = softmax(logits, axis=1)[0][:, :num_frames]  # [C, T]

    print(f"audio: {args.audio} ({wav_length:.2f}s, {num_frames} frames)")
    for i, name in enumerate(class_names):
        if name == "None":
            continue
        intervals = to_intervals(probs[i], frame_duration,
                                 threshold=args.threshold, min_frames=args.min_frames)
        print(f"\n[{name}] {len(intervals)} interval(s):")
        for start, end, conf in intervals:
            print(f"  {start:8.3f}s - {end:8.3f}s  (len {end - start:6.3f}s, conf {conf:.3f})")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(num_frames) * frame_duration
        fig, ax = plt.subplots(figsize=(14, 4))
        for i, name in enumerate(class_names):
            ax.plot(t, probs[i], label=name, lw=0.8)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("probability")
        ax.set_ylim(-0.02, 1.02)
        ax.legend()
        ax.set_title("non-lexical phoneme probabilities")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nplot saved to: {args.plot}")


if __name__ == "__main__":
    main()
