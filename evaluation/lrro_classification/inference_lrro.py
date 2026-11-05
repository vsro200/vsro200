"""
Single-clip word classification on LRRo using our pre-trained encoder
and a lightweight attention-pooling + MLP head.

Pipeline:
  1. Load clip frames (preprocessing strategy chosen via --strategy)
  2. Pass through the (frozen) VTP visual encoder + VSR transformer encoder
  3. Pool temporally with attention; classify with MLP

The VTP visual encoder and the VSR encoder are FIXED (not configurable):
  - VTP: original feature extractor from VGG Oxford
  - VSR encoder: from vsro200/models-vsro200/checkpoints/model_200h_auto.pt

The MLP and preprocessing strategy ARE configurable. Each preprocessing
strategy has its own MLPs trained on it (separate for the LAB and WILD
splits of LRRo).

Example usage:
    # Default: 64_bottom strategy + LAB MLP
    python inference_lrro.py --clip_dir /path/to/lrro/clip_folder

    # Use 96_resize + Wild MLP:
    python inference_lrro.py --clip_dir /path/to/clip --strategy 96_resize --split wild

Class names come from lrro_lab_classes.json and lrro_wild_classes.json in
this folder: the alphabetical order of the word folders across LRRo's train,
val and test splits, which is how the label indices were built in training.

LRRo dataset must be obtained separately from the official source:
    https://bionescu.aimultimedialab.ro/LRRo.html
"""

import argparse
import json
import os
import sys
import warnings

# Suppress autocast warnings on CPU
warnings.filterwarnings("ignore", message=".*CUDA is not available.*")

import torch
from huggingface_hub import hf_hub_download

# Add the repository root to path so we can import from ro_vsr/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from vsr_inference.models import build_model, build_visual_encoder

from preprocessing import PREPROCESSING_FNS
from model import MLP


# ============================================================
# DEFAULTS
# ============================================================

VSR_MODEL_REPO = "vsro200/models-vsro200"            # fixed
VSR_MODEL_FILENAME = "model_200h_auto.pt"            # fixed
MLP_REPO = "vsro200/mlp-lrro-vsro200"                # fixed
VTP_PATH = os.path.join(REPO_ROOT, "checkpoints/feature_extractor.pth")  # fixed

DEFAULT_STRATEGY = "64_bottom"
DEFAULT_SPLIT = "lab"


# ============================================================
# LRRo VOCABULARIES
# ============================================================
# LAB has 48 word classes, WILD has 21, each including a `distractor` class.
# Training indexed them as sorted(set of word folders) over the train, val
# and test splits together. The two JSON files store exactly that mapping,
# {word: index}, so the labels do not depend on how a user's copy of LRRo
# is laid out.

CLASS_MAP_FILES = {
    "lab": os.path.join(os.path.dirname(os.path.abspath(__file__)), "lrro_lab_classes.json"),
    "wild": os.path.join(os.path.dirname(os.path.abspath(__file__)), "lrro_wild_classes.json"),
}


def resolve_class_names(num_classes: int, split: str,
                        class_map_file: str = None) -> tuple:
    """
    Returns (class_names, source), where class_names[i] is the word for
    output i of the MLP.

    Uses --class_map if given, otherwise the JSON file bundled for the split.
    Accepts {word: index} or {index: word}. A list whose length differs from
    the MLP's output size is an error, not a warning: the predictions would
    still look plausible, just attached to the wrong words.
    """
    path = class_map_file or CLASS_MAP_FILES[split]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Class map not found: {path}")

    with open(path, encoding="utf-8") as f:
        mapping = json.load(f)
    if all(str(k).isdigit() for k in mapping.keys()):
        names = [mapping[str(i)] for i in range(len(mapping))]
    else:
        names = sorted(mapping.keys(), key=lambda w: mapping[w])

    if len(names) != num_classes:
        raise ValueError(
            f"{path} lists {len(names)} classes but the {split} MLP has "
            f"{num_classes} outputs. Use the class map that matches the split."
        )
    return names, f"from {os.path.basename(path)}"


# ============================================================
# MODEL LOADING
# ============================================================

def load_encoder_models(device: torch.device):
    """Load the (frozen) VTP visual encoder and the VSR transformer encoder."""
    if not os.path.isfile(VTP_PATH):
        raise FileNotFoundError(
            f"VTP checkpoint not found: {VTP_PATH}\n"
            f"   Run: bash scripts/setup.sh (from the repository root)"
        )

    print(f"[load] Loading VTP visual encoder from {VTP_PATH}")
    visual_encoder = build_visual_encoder().to(device).eval()
    s = torch.load(VTP_PATH, map_location=device)["state_dict"]
    new_s = {
        k.replace("module.face_encoder.", ""): v
        for k, v in s.items() if "face_encoder" in k
    }
    visual_encoder.load_state_dict(new_s)
    for p in visual_encoder.parameters():
        p.requires_grad = False

    print(f"[load] Downloading VSR encoder from {VSR_MODEL_REPO}/checkpoints/{VSR_MODEL_FILENAME}")
    enc_path = hf_hub_download(
        repo_id=VSR_MODEL_REPO,
        filename=f"checkpoints/{VSR_MODEL_FILENAME}",
        repo_type="model",
    )
    model = build_model().to(device).eval()
    model.load_state_dict(torch.load(enc_path, map_location=device))
    for p in model.parameters():
        p.requires_grad = False

    return visual_encoder, model


def load_mlp(strategy: str, split: str, device: torch.device):
    """
    Load the MLP head from HuggingFace. The MLP repo is structured as:
        <strategy>/best_lab_clf.pt
        <strategy>/best_wild_clf.pt

    The input size, hidden size and number of classes are all read from the
    saved state_dict, so nothing here has to be kept in sync by hand.
    """
    filename = f"{strategy}/best_{split}_clf.pt"
    print(f"[load] Downloading MLP from {MLP_REPO}/{filename}")
    mlp_path = hf_hub_download(
        repo_id=MLP_REPO, filename=filename, repo_type="model",
    )
    state_dict = torch.load(mlp_path, map_location=device)

    hidden_dim, input_dim = state_dict["net.0.weight"].shape
    num_classes = state_dict["net.4.weight"].shape[0]
    print(f"[load] MLP has {num_classes} output classes")

    clf = MLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    ).to(device).eval()
    clf.load_state_dict(state_dict)
    return clf, num_classes


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    clip_dir: str,
    strategy: str,
    visual_encoder,
    encoder_model,
    classifier,
    device: torch.device,
    top_k: int = 5,
):
    """Returns a list of (class_idx, probability) tuples sorted by probability."""
    load_fn = PREPROCESSING_FNS[strategy]
    video = load_fn(clip_dir)
    if video is None:
        raise ValueError(f"No frames found in {clip_dir}")

    video_t = torch.FloatTensor(video).unsqueeze(0).to(device)  # (1, 3, T, 96, 96)
    print(f"[video] Frames extracted: {tuple(video_t.shape)}")

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            vid_emb = visual_encoder(video_t)
            B, T, _ = vid_emb.size()
            src_mask_enc = torch.ones((B, 1, T), device=device).bool()
            memory, _ = encoder_model.encode(vid_emb, src_mask_enc)
            # memory: (1, T_enc, D)

        # The head runs in fp32, outside autocast. In training it learned from
        # encoder outputs saved as float32 and was trained in float32, so this
        # is the precision it was fitted at.
        memory = memory.float()
        T_enc = memory.size(1)
        mlp_mask = torch.ones((1, 1, T_enc), dtype=torch.bool, device=device)
        logits, _ = classifier(memory, mlp_mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()

    top_probs, top_idx = probs.topk(min(top_k, probs.size(0)))
    return [(int(i), float(p)) for i, p in zip(top_idx.tolist(), top_probs.tolist())]


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="LRRo word classification on a single clip",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clip_dir", type=str, required=True,
        help="Folder containing the clip's .jpg frames (numbered 0.jpg, 1.jpg, ...)",
    )
    parser.add_argument(
        "--strategy", type=str, default=DEFAULT_STRATEGY,
        choices=list(PREPROCESSING_FNS.keys()),
        help="Preprocessing strategy: how to place LRRo frames on the 96x96 canvas",
    )
    parser.add_argument(
        "--split", type=str, default=DEFAULT_SPLIT, choices=["lab", "wild"],
        help="Which MLP head to use (lab: 48 classes, wild: 21 classes)",
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of top predictions to display",
    )
    parser.add_argument(
        "--class_map", type=str, default=None,
        help="Optional JSON mapping words to class indices. Defaults to the "
             "file bundled for the chosen split.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Explicit device (cuda / cpu). Default: auto-detect.",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[device] {device}")

    if not os.path.isdir(args.clip_dir):
        print(f"❌ Clip folder not found: {args.clip_dir}")
        sys.exit(1)

    # Load models
    visual_encoder, encoder_model = load_encoder_models(device)
    classifier, num_classes = load_mlp(args.strategy, args.split, device)

    # Detect mismatch between clip dataset and selected split (warning shown
    # AFTER inference so it's not lost in the download logs)
    split_mismatch = None
    if "Lab_LRRo" in args.clip_dir and args.split != "lab":
        split_mismatch = "lab"
    elif "Wild_LRRo" in args.clip_dir and args.split != "wild":
        split_mismatch = "wild"

    class_names, source = resolve_class_names(
        num_classes=num_classes,
        split=args.split,
        class_map_file=args.class_map,
    )
    print(f"[load] Class names {source}")

    # Run inference
    print("[infer] Running inference ...")
    predictions = run_inference(
        clip_dir=args.clip_dir,
        strategy=args.strategy,
        visual_encoder=visual_encoder,
        encoder_model=encoder_model,
        classifier=classifier,
        device=device,
        top_k=args.top_k,
    )

    # Output
    print("─" * 70)
    print(f"Clip:            {args.clip_dir}")
    print(f"Strategy:        {args.strategy}")
    print(f"MLP split:       {args.split}  ({num_classes} classes)")

    # If clip is in a folder named after its true word, show that
    true_word = None
    parts = os.path.normpath(args.clip_dir).split(os.sep)
    if len(parts) >= 2:
        candidate = parts[-2]  # parent folder name
        if candidate in class_names:
            true_word = candidate
            print(f"True label:      {true_word}")

    print(f"Top-{args.top_k} predictions:")
    for rank, (idx, prob) in enumerate(predictions, start=1):
        word = class_names[idx]
        marker = "  ←" if word == true_word else ""
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {rank}. {word:<20s} {bar} {prob*100:5.2f}%{marker}")
    print("─" * 70)

    # Surface mismatches at the end where they're easy to spot
    if split_mismatch is not None:
        dataset_name = "Lab_LRRo_data_set" if split_mismatch == "lab" else "Wild_LRRo_data_set"
        print()
        print(f"⚠️  WARNING: Clip is in {dataset_name} but you used --split={args.split}.")
        print(f"   The {args.split} MLP has different vocabulary than the {split_mismatch} MLP.")
        print(f"   Re-run with --split {split_mismatch} for meaningful predictions.")


if __name__ == "__main__":
    main()
