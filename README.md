# AI Voice Fraud Detection — Real Human Voice Dataset Builder

## What this does

Builds **100,000+ real human voice WAV clips** (3–5 s, 16 kHz mono) ready for
training a CNN spectrogram model.

```
dataset/
└── real/
    ├── real_000001.wav
    ├── real_000002.wav
    └── … (100,000+)
```

---

## Quick-start (Ubuntu / WSL / macOS)

```bash
# 1. System dependency
sudo apt install ffmpeg          # Ubuntu / WSL
# brew install ffmpeg            # macOS

# 2. Python dependencies
pip install -r requirements.txt

# 3. Run (automated sources only — gets you ~70–120k clips on its own)
python build_dataset.py
```

That single command will:
- Download **LibriSpeech train-clean-100** (~7 GB) + **TEDLIUM-3** (~9 GB) +
  **AISHELL-1** (~15 GB) automatically
- Convert all audio to 16 kHz mono WAV
- Chunk into 3–5 s segments
- Discard silent / empty chunks
- Preserve background noise (ideal for real-world fraud detection)
- Save to `dataset/real/`

---

## Reaching 100,000 samples — source by source

| Source | Clips (est.) | Auto? | How to get |
|---|---|---|---|
| LibriSpeech train-clean-100 | ~28,000 | ✅ Auto | included |
| TEDLIUM-3 | ~50,000 | ✅ Auto | included |
| AISHELL-1 | ~16,000 | ✅ Auto | included |
| **Subtotal (auto)** | **~94,000** | | |
| Mozilla Common Voice | +200,000+ | ⚠️ Free reg. | see below |
| VoxCeleb 1 | +148,000 | ⚠️ Free reg. | see below |
| CHiME-4 (noisy) | +8,000 | ⚠️ Email req. | see below |

The three auto sources alone usually exceed 100k clips. The manual sources
add millions more if needed.

---

## Manual sources — step by step

### 1. Mozilla Common Voice (free, biggest)

1. Go to <https://commonvoice.mozilla.org/en/datasets>
2. Create a free account and accept the CC-0 licence
3. Download the **English** corpus (`cv-corpus-*.tar.gz`, ~70 GB for full dataset
   — the validated TSV alone filters to ~2M quality clips)
4. Extract:
   ```bash
   mkdir -p downloads/common_voice
   tar -xf cv-corpus-*.tar.gz -C downloads/common_voice
   ```
5. Re-run the script:
   ```bash
   python build_dataset.py --cv-dir downloads/common_voice/en
   ```

### 2. VoxCeleb 1 (celebrity interviews, real-world noise)

1. Register for free at <https://mm.kaist.ac.kr/datasets/voxceleb/>
2. Download `vox1_dev_wav.zip` (~39 GB) — 148k utterances, 1,251 speakers
3. Extract:
   ```bash
   unzip vox1_dev_wav.zip -d downloads/voxceleb
   ```
4. Re-run:
   ```bash
   python build_dataset.py --vox-dir downloads/voxceleb/wav
   ```

### 3. CHiME-4 real noisy speech (optional, high value for noise diversity)

1. Visit <https://spandh.dcs.shef.ac.uk/chime_challenge/chime4/>
2. Fill in the data agreement form
3. Download and extract the **real** subset (~12 GB)
4. Re-run:
   ```bash
   python build_dataset.py --chime-dir downloads/chime4
   ```

---

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--cv-dir PATH` | — | Common Voice extracted folder |
| `--vox-dir PATH` | — | VoxCeleb extracted folder |
| `--chime-dir PATH` | — | CHiME extracted folder |
| `--all-librispeech` | off | Download all 960 h of LibriSpeech (≈500k more clips) |
| `--skip-tedlium` | off | Skip TEDLIUM-3 download |
| `--skip-aishell` | off | Skip AISHELL-1 download |
| `--output-dir PATH` | `dataset/real` | Where to save final WAVs |

Examples:
```bash
# All automated + VoxCeleb
python build_dataset.py --vox-dir downloads/voxceleb/wav

# Everything at once
python build_dataset.py \
  --cv-dir    downloads/common_voice/en \
  --vox-dir   downloads/voxceleb/wav \
  --chime-dir downloads/chime4 \
  --all-librispeech
```

---

## Audio processing pipeline

```
Raw audio (any format)
       │
       ▼
ffmpeg → 16 kHz mono WAV
       │
       ▼
pydub → 5-second sliding window chunks
       │
       ├─ chunk < 3 s        → discard
       ├─ silent (< 500 ms   → discard
       │   of speech)
       └─ has speech         → save to dataset/real/real_NNNNNN.wav
                               (background noise preserved)
```

---

## Silence gate settings

Edit these constants at the top of `build_dataset.py` if needed:

| Constant | Default | Meaning |
|---|---|---|
| `SILENCE_THRESH` | −50 dBFS | Anything quieter = silence |
| `MIN_NONSILENT_MS` | 500 ms | Minimum speech in chunk to keep |
| `CHUNK_MIN_MS` | 3,000 ms | Minimum chunk duration |
| `CHUNK_MAX_MS` | 5,000 ms | Maximum chunk duration |

---

## Licences summary

| Dataset | Licence | Commercial use |
|---|---|---|
| LibriSpeech | CC BY 4.0 | ✅ |
| TEDLIUM-3 | CC BY-NC-ND 3.0 | Research only |
| AISHELL-1 | Apache 2.0 | ✅ |
| Common Voice | CC-0 | ✅ |
| VoxCeleb | Research / VGG licence | Research only |
| CHiME-4 | Research only | Research only |

For commercial production, prefer LibriSpeech + Common Voice + AISHELL-1.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ffmpeg not found` | `sudo apt install ffmpeg` |
| Download stalls | Re-run — script resumes from already-downloaded archives |
| `pydub` import error | `pip install pydub` |
| Low sample count | Add `--all-librispeech` or add manual sources |
| Multi-channel CHiME | ffmpeg `-ac 1` handles downmix automatically |
