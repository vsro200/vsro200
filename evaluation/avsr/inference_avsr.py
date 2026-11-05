"""
Single-clip AVSR (Audio-Visual Speech Recognition) inference.

Combines our VSR model (video) with a fine-tuned Whisper model (audio).
At every decoding step the two next-token distributions are mixed with a
weight computed from their entropies, so the branch that is less certain at
that step counts for less (see beam_search_fusion.py).

The VSR model uses the MultiVSR architecture (Prajwal et al., 2025),
trained on our VSRo-200 dataset.

Example usage:
    python inference_avsr.py --fpath samples_avsr/sample_1_babble_SNR-5.avi
    python inference_avsr.py --fpath samples_avsr/sample_1_babble_SNR-5.avi --mode audio
    python inference_avsr.py --fpath samples_avsr/sample_1_babble_SNR-5.avi --mode all

Input clips must be .avi or .mp4 files with:
  - a face-cropped video track (resized to 160x160, then centre-cropped to 96x96)
  - an audio track (mono 16 kHz is used as is; anything else is converted)

The clips in `samples_avsr/` already carry noise in their audio track. The
video is the clean face crop; only the audio is degraded.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import warnings

# Suppress autocast warnings on CPU (decorators in models.py are static)
warnings.filterwarnings("ignore", message=".*CUDA is not available.*")

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

# Add the repository root to path so we can import from vsr_inference/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from vsr_inference.models import build_model, build_visual_encoder
from vsr_inference.tokenizer import get_tokenizer
from vsr_inference.metrics import compute_wer_cer, lookup_reference, print_metrics_block

from beam_search_fusion import MODES, beam_search_fusion


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_VSR_MODEL_REPO = "vsro200/models-vsro200"
DEFAULT_VSR_MODEL_FILENAME = "model_200h_auto.pt"
DEFAULT_WHISPER_MODEL = "vsro200/whisper-small-vsro200"
DEFAULT_VTP_PATH = os.path.join(REPO_ROOT, "checkpoints/feature_extractor.pth")
DEFAULT_METADATA = os.path.join(
    os.path.dirname(__file__), "samples_avsr", "samples_avsr_metadata.csv"
)

START_PROMPT = "<|startoftranscript|><|ro|><|transcribe|><|notimestamps|>"


# ============================================================
# UTILITIES
# ============================================================

def clean_prediction(text: str) -> str:
    """Remove Whisper-style special tokens from the prediction."""
    for token in [
        "<|startoftranscript|>", "<|ro|>", "<|transcribe|>",
        "<|notimestamps|>", "<|endoftext|>",
    ]:
        text = text.replace(token, "")
    return text.strip()


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation (keeping hyphens), normalize whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text)
    return " ".join(text.split()).strip()


# ============================================================
# I/O
# ============================================================

def read_video(fpath: str, device: torch.device) -> torch.Tensor:
    """Read a face-cropped .mp4/.avi clip -> tensor of shape [1, 3, T, 96, 96]."""
    from decord import VideoReader, bridge
    bridge.set_bridge("native")

    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Video file not found: {fpath}")

    with open(fpath, "rb") as f:
        vr = VideoReader(f, width=160, height=160)
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()

    frames = frames.astype(np.float32) / 255.0
    frames = torch.from_numpy(frames).to(device).unsqueeze(0)
    frames = frames.permute(0, 4, 1, 2, 3)  # [1, C, T, H, W]

    crop_x = (frames.size(3) - 96) // 2
    crop_y = (frames.size(4) - 96) // 2
    return frames[:, :, :, crop_x:crop_x + 96, crop_y:crop_y + 96]


def _read_audio_ffmpeg(fpath: str, target_sr: int) -> torch.Tensor:
    """Decode the audio track with ffmpeg into float32 samples, mono.

    Samples are requested as 32-bit float, so a float track is returned bit
    for bit, and an integer track gets the same scaling torchaudio applies.
    Channels are averaged here rather than by ffmpeg, whose downmix matrix
    does not always weight them equally.
    """
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("Reading audio needs either a working torchaudio.load "
                           "or ffmpeg/ffprobe on PATH.")

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels,sample_rate", "-of", "json", fpath],
        capture_output=True, text=True,
    )
    streams = json.loads(probe.stdout or "{}").get("streams", [])
    if not streams:
        raise RuntimeError(f"No audio track in {fpath}")
    channels = int(streams[0]["channels"])

    cmd = ["ffmpeg", "-v", "error", "-i", fpath, "-vn", "-f", "f32le"]
    if int(streams[0]["sample_rate"]) != target_sr:
        cmd += ["-ar", str(target_sr)]
    cmd.append("pipe:1")
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode the audio of {fpath}:\n"
                           f"{out.stderr.decode(errors='replace')}")

    samples = np.frombuffer(out.stdout, dtype="<f4").reshape(-1, channels)
    return torch.from_numpy(samples.mean(axis=1).astype(np.float32))


def read_audio(fpath: str, target_sr: int = 16000) -> torch.Tensor:
    """Read the audio track of a video file -> mono 16 kHz tensor of shape [T].

    torchaudio is tried first. Recent torchaudio versions moved decoding to a
    separate package (torchcodec), so where `torchaudio.load` is unavailable
    the track is decoded with ffmpeg instead.
    """
    try:
        import torchaudio
        waveform, sr = torchaudio.load(fpath)
    except Exception as e:
        print(f"[audio] torchaudio could not read the file ({type(e).__name__}); "
              f"decoding with ffmpeg")
        return _read_audio_ffmpeg(fpath, target_sr)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.mean(dim=0)


# ============================================================
# MODEL LOADING
# ============================================================

def load_vsr_models(vsr_filename: str, vtp_path: str, device: torch.device):
    """Load the VSR encoder-decoder and the VTP visual encoder."""
    if not os.path.isfile(vtp_path):
        raise FileNotFoundError(
            f"VTP checkpoint not found: {vtp_path}\n"
            f"   Run: bash scripts/setup.sh (from the repository root)"
        )

    visual_encoder = build_visual_encoder().to(device).eval()
    s = torch.load(vtp_path, map_location=device)["state_dict"]
    new_s = {}
    for k, v in s.items():
        if "face_encoder" not in k:
            continue
        new_s[k.replace("module.face_encoder.", "")] = v
    visual_encoder.load_state_dict(new_s)
    for p in visual_encoder.parameters():
        p.requires_grad = False

    print(f"[load] Downloading VSR model from {DEFAULT_VSR_MODEL_REPO}/checkpoints/{vsr_filename} ...")
    lm_path = hf_hub_download(
        repo_id=DEFAULT_VSR_MODEL_REPO,
        filename=f"checkpoints/{vsr_filename}",
        repo_type="model",
    )
    model = build_model().to(device).eval()
    model.load_state_dict(torch.load(lm_path, map_location=device))

    return model, visual_encoder


def load_whisper(whisper_repo: str, device: torch.device):
    """Load the fine-tuned Whisper model and its feature extractor.

    Only the feature extractor (audio -> log-mel) is needed: decoding uses the
    MultiVSR tokenizer, shared by both branches. Loading the full
    WhisperProcessor would also load Whisper's own tokenizer for nothing.
    """
    print(f"[load] Downloading Whisper from {whisper_repo} ...")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_repo).to(device)
    whisper_model.eval()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_repo)
    return whisper_model, feature_extractor


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    video: torch.Tensor,        # [1, 3, T, 96, 96]
    audio: torch.Tensor,        # [T_audio]
    vsr_model,
    visual_encoder,
    whisper_model,
    feature_extractor,
    tokenizer,
    device: torch.device,
    modes=("fusion",),
    beam_size: int = 5,
    max_len: int = 256,
) -> dict:
    """Encode the clip once, then decode it in each requested mode.

    Returns {mode: (transcription, mean_alpha)}; mean_alpha is None outside
    fusion mode. Autocast wraps the decoding as well as the encoders, as in
    the evaluation reported in the paper.
    """
    start_prompt_ids = tokenizer.encode(START_PROMPT)
    results = {}

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            # Audio path: Whisper encoder
            mel = feature_extractor(
                audio.cpu().numpy(), sampling_rate=16000, return_tensors="pt"
            ).input_features.to(device)
            audio_embeds = whisper_model.model.encoder(mel).last_hidden_state

            # Video path: VTP + VSR encoder
            vid_emb = visual_encoder(video)
            B, T, _ = vid_emb.size()
            src_mask = torch.ones((B, 1, T), device=device).bool()
            video_memory, _ = vsr_model.encode(vid_emb, src_mask)

            for mode in modes:
                stats = {} if mode == "fusion" else None
                beam_outs, _ = beam_search_fusion(
                    whisper_model=whisper_model.model,
                    whisper_proj=whisper_model.proj_out,
                    audio_embeds=audio_embeds,
                    multivsr_model=vsr_model,
                    video_memory=video_memory,
                    src_mask=src_mask,
                    bos_indices=start_prompt_ids,
                    eos_index=tokenizer.eot,
                    pad_index=0,
                    max_output_length=max_len,
                    size=beam_size,
                    n_best=1,
                    mode=mode,
                    stats=stats,
                )

                if not beam_outs[0]:
                    text = ""
                else:
                    best_ids = beam_outs[0][0]
                    if isinstance(best_ids, torch.Tensor):
                        best_ids = best_ids.tolist()
                    text = clean_prediction(tokenizer.decode(best_ids))

                alpha = (float(np.mean(stats["alpha"]))
                         if stats and stats.get("alpha") else None)
                results[mode] = (text, alpha)

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single-clip AVSR inference with entropy-weighted fusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fpath", type=str, required=True,
                        help="Path to the input video file (.avi or .mp4, face crop with audio)")
    parser.add_argument("--vsr_model", type=str, default=DEFAULT_VSR_MODEL_FILENAME,
                        help="Checkpoint filename in vsro200/models-vsro200 (e.g. model_200h_auto.pt)")
    parser.add_argument("--whisper_model", type=str, default=DEFAULT_WHISPER_MODEL,
                        help="HuggingFace repo for the Whisper model")
    parser.add_argument("--vtp_path", type=str, default=DEFAULT_VTP_PATH,
                        help="Path to the VTP feature extractor checkpoint")
    parser.add_argument("--mode", type=str, default="fusion",
                        choices=list(MODES) + ["all"],
                        help="fusion, audio only, video only, or all three on the same clip")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--metadata", type=str, default=DEFAULT_METADATA,
                        help="Path to metadata CSV (used to look up reference and compute WER/CER if available)")
    parser.add_argument("--device", type=str, default=None,
                        help="Explicit device (cuda / cpu). Default: auto.")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[device] {device}")

    # Load
    tokenizer = get_tokenizer()
    vsr_model, visual_encoder = load_vsr_models(args.vsr_model, args.vtp_path, device)
    whisper_model, feature_extractor = load_whisper(args.whisper_model, device)
    print("[load] All models loaded")

    # Read
    video = read_video(args.fpath, device)
    audio = read_audio(args.fpath, target_sr=16000)
    print(f"[video] Frames extracted: {tuple(video.shape)}")
    print(f"[audio] Samples loaded:   {audio.shape[0]} ({audio.shape[0]/16000:.2f}s @ 16 kHz)")

    # Inference
    modes = list(MODES) if args.mode == "all" else [args.mode]
    print(f"[infer] Running inference (mode={args.mode}) ...")
    results = run_inference(
        video=video,
        audio=audio,
        vsr_model=vsr_model,
        visual_encoder=visual_encoder,
        whisper_model=whisper_model,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        device=device,
        modes=modes,
        beam_size=args.beam_size,
        max_len=args.max_len,
    )

    reference = lookup_reference(args.metadata, args.fpath)
    print("─" * 70)
    print(f"File:           {args.fpath}")

    if args.mode != "all":
        text, alpha = results[args.mode]
        print(f"Mode:           {args.mode}")
        print(f"Transcription:  {text}")
        if alpha is not None:
            print(f"Mean alpha:     {alpha:.3f}  (weight of the audio branch)")
        print_metrics_block(reference, text, normalize_text)
    else:
        if reference is not None:
            print(f"Reference:      {reference}")
        for mode in modes:
            text, alpha = results[mode]
            print()
            print(f"[{mode}]")
            print(f"  Transcription:  {text}")
            if reference is not None:
                wer, cer = compute_wer_cer(normalize_text(reference),
                                           normalize_text(text))
                if wer is not None:
                    print(f"  WER / CER:      {wer * 100:.2f}% / {cer * 100:.2f}%")
            if alpha is not None:
                print(f"  Mean alpha:     {alpha:.3f}  (weight of the audio branch)")
    print("─" * 70)


if __name__ == "__main__":
    main()
