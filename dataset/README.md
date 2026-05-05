# VSRo-200 Dataset

This folder contains the scripts needed to **reconstruct the VSRo-200
dataset** from its source YouTube videos.

We do not redistribute the source videos themselves. Instead, we
publish on HuggingFace the timestamps and transcripts that define the
dataset, and provide here the scripts to download and cut the clips
from YouTube.

## Dataset on HuggingFace

The VSRo-200 metadata is hosted at
[vsro200/vsro200](https://huggingface.co/datasets/vsro200/vsro200). Each split is a CSV with
the format:

```csv
file_path,start,end,transcript,gender
MfbPvQLQv54/00402,78.61,85.13,"...",M
MfbPvQLQv54/00404,142.32,147.03,"...",M
```

Where:

- `file_path` is `<youtube_id>/<5-digit-index>` and identifies each clip
- `start`, `end` are timestamps in seconds within the source video
- `transcript` is the Romanian transcription
- `gender` is the speaker's gender (M/F); replaced by `domain` for the
  `test_ood` split

Available splits:

| Split | Description | Duration | Extra columns |
| :--- | :--- | :--- | :--- |
| `trainval_annot.csv` | Manually annotated train/val | ~100h | gender |
| `trainval_auto.csv` | Auto-transcribed train/val (Whisper-large) | ~200h | gender |
| `test_seen.csv` | Test split, speakers seen during training | ~1h | gender |
| `test_unseen.csv` | Test split, unseen speakers | ~1h | gender |
| `test_ood.csv` | Out-of-domain test (different content type) | ~1h | domain |

## Reconstruction workflow

### Dependencies

```bash
pip install yt-dlp pandas "huggingface_hub[cli]"
pip install scenedetect==0.6.6 face-alignment python-speech-features
```

You'll also need `ffmpeg` available on your system PATH.

### Step 1 — Download the metadata from HuggingFace

Pick whichever splits you need.

```bash
git clone https://github.com/vsro200/vsro200.git
cd vsro200/dataset

# Download a single split
hf download vsro200/vsro200 test_seen.csv \
    --repo-type dataset --local-dir ./

# Or download all CSVs at once
hf download vsro200/vsro200 \
    --repo-type dataset --local-dir ./ \
    --include "*.csv"
```

### Step 2 — Download the source YouTube videos

```bash
python download_videos.py --ids youtube_ids.txt --output_dir videos/
```

This downloads each video listed in `youtube_ids.txt` as
`videos/<youtube_id>.mp4`. Videos that are already downloaded are
skipped.

### Step 3 — Cut the clips

For each split you downloaded:

```bash
python cut_clips.py \
    --csv test_seen.csv \
    --videos_dir videos/ \
    --output_dir clips/test_seen/
```

The output is laid out as:

```
clips/test_seen/
├── ZWNM2sZxtRg/
│   ├── 00001.mp4
│   ├── 00002.mp4
│   └── ...
├── abc123def/
│   ├── 00001.mp4
│   └── ...
```

If a source video is missing (e.g., removed from YouTube), the script
errors out by default. Pass `--skip_missing` to skip those clips
silently and continue with what's available.

### Step 4 — Extract face tracks for VSR

The clips from Step 3 still contain the full frame (with body, background,
etc.). To produce the 224×224 face crops that our VSR model expects as
input, run them through MultiVSR's preprocessing pipeline.

```bash
git clone https://github.com/Sindhu-Hegde/multivsr.git /tmp/multivsr

WORK_DIR=$(pwd)
FACE_TRACKS_ABS_PATH="${WORK_DIR}/face_tracks"

for clip_relative_path in clips/test_seen/*/*.mp4; do
    clip_absolute_path=$(realpath "$clip_relative_path")

    youtube_id=$(basename "$(dirname "$clip_absolute_path")")
    clip_index=$(basename "$clip_absolute_path" .mp4)

    (cd /tmp/multivsr/preprocess && python run_pipeline.py \
        --videofile "$clip_absolute_path" \
        --reference "${youtube_id}/${clip_index}" \
        --data_dir "$FACE_TRACKS_ABS_PATH")
done
```

The pipeline produces several intermediate folders. The face-cropped
files we want for VSR inference or training are in `pycrop/`:

```
/path/to/face_tracks/test_seen/
├── pycrop/
│   ├── ZWNM2sZxtRg/
│   │   ├── 00001/
│   │   │   └── 00000.avi
│   │   ├── 00002/
│   │   │   └── 00000.avi
│   │   └── ...
│   └── abc123def/
│       └── 00001/
│           └── 00000.avi
├── pyavi/      (intermediate — full frames)
├── pyframes/   (intermediate — extracted frames)
├── pytmp/      (intermediate — temp audio)
└── pywork/     (intermediate — face-tracking metadata)
```

The `.avi` files under `pycrop/` are the actual inputs accepted by
`inference.py`, `evaluate.py`, and the AVSR / LRRo evaluation scripts. You can safely delete the other intermediate folders.
