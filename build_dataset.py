#!/usr/bin/env python3
"""
=============================================================================
AI Voice Fraud Detection — Real Human Voice Dataset Builder
=============================================================================
Goal   : Build 100,000+ real human voice WAV clips (3–5 sec, 16kHz mono)
Sources: Mozilla Common Voice · LibriSpeech · VoxCeleb · CHiME-4/5
Output : dataset/real/real_000001.wav … real_NNNNNN.wav
=============================================================================
"""

import os
import sys
import re
import csv
import glob
import shutil
import hashlib
import tarfile
import zipfile
import logging
import argparse
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# ── third-party ─────────────────────────────────────────────────────────────
try:
    import numpy as np
    import soundfile as sf
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    from tqdm import tqdm
except ImportError:
    print("[INSTALL] Missing libraries. Run:\n"
          "  pip install numpy soundfile pydub tqdm")
    sys.exit(1)

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("dataset_build.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────
TARGET_SR       = 16_000          # 16 kHz mono
CHUNK_MIN_MS    = 3_000           # 3 s
CHUNK_MAX_MS    = 5_000           # 5 s
SILENCE_THRESH  = -50             # dBFS threshold for silence gate
MIN_NONSILENT_MS= 500             # discard chunks with <500 ms of speech
TARGET_SAMPLES  = 100_000

OUTPUT_DIR      = Path("dataset/real")
DOWNLOAD_DIR    = Path("downloads")
SCRATCH_DIR     = Path("scratch")

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

class Counter:
    """Thread-safe-ish global sample counter."""
    def __init__(self):
        self.n = 0
    def next(self) -> int:
        self.n += 1
        return self.n

COUNTER = Counter()


def ensure_dirs():
    for d in (OUTPUT_DIR, DOWNLOAD_DIR, SCRATCH_DIR):
        d.mkdir(parents=True, exist_ok=True)


def wav_output_path() -> Path:
    return OUTPUT_DIR / f"real_{COUNTER.next():06d}.wav"


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def convert_to_wav_16k(src: Path, dst: Path) -> bool:
    """Convert any audio file → 16kHz mono WAV via ffmpeg."""
    if not ffmpeg_available():
        log.error("ffmpeg not found. Install with: sudo apt install ffmpeg")
        return False
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-ac", "1",           # mono
        "-ar", str(TARGET_SR),
        "-sample_fmt", "s16",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def chunk_and_save(wav_path: Path, keep_noise: bool = True) -> int:
    """
    Split a WAV file into 3-5 s chunks, drop silent ones, save to OUTPUT_DIR.
    Returns number of chunks saved.
    """
    saved = 0
    try:
        audio = AudioSegment.from_wav(str(wav_path))
    except Exception as e:
        log.debug(f"Cannot open {wav_path}: {e}")
        return 0

    # Ensure 16kHz mono (belt-and-suspenders after ffmpeg)
    audio = audio.set_frame_rate(TARGET_SR).set_channels(1)
    total_ms = len(audio)

    chunk_ms = CHUNK_MAX_MS  # use 5-s windows; last chunk may be shorter
    pos = 0
    while pos < total_ms:
        end = min(pos + chunk_ms, total_ms)
        chunk = audio[pos:end]
        dur = len(chunk)

        if dur < CHUNK_MIN_MS:   # too short — skip
            pos += chunk_ms
            continue

        # Silence gate: keep chunk only if it has enough non-silent speech
        nonsilent = detect_nonsilent(
            chunk,
            min_silence_len=200,
            silence_thresh=SILENCE_THRESH
        )
        nonsilent_ms = sum(e - s for s, e in nonsilent)

        if nonsilent_ms < MIN_NONSILENT_MS:
            pos += chunk_ms
            continue

        # Background noise is intentionally preserved (no noise stripping)
        out_path = wav_output_path()
        chunk.export(str(out_path), format="wav")
        saved += 1
        pos += chunk_ms

    return saved


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    if dest.exists():
        log.info(f"  Already downloaded: {dest.name}")
        return True
    log.info(f"  Downloading {desc or dest.name} …")
    try:
        def _reporthook(count, block, total):
            if total > 0:
                pct = min(count * block / total * 100, 100)
                print(f"\r  {pct:5.1f}%", end="", flush=True)
        urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
        print()
        return True
    except (urllib.error.URLError, Exception) as e:
        log.warning(f"  Download failed ({e}). Try manually.")
        return False


def extract_archive(archive: Path, target_dir: Path):
    log.info(f"  Extracting {archive.name} → {target_dir} …")
    target_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix in (".tar", ".gz", ".bz2", ".xz") or ".tar" in archive.name:
        with tarfile.open(archive) as tf:
            tf.extractall(target_dir)
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target_dir)
    else:
        log.warning(f"Unknown archive format: {archive}")


def process_audio_files(source_dir: Path, ext_pattern: str = "**/*.flac",
                        label: str = "") -> int:
    files = list(source_dir.glob(ext_pattern))
    log.info(f"  [{label}] Found {len(files):,} source files in {source_dir}")
    saved = 0
    tmp = SCRATCH_DIR / "tmp_conv.wav"

    for f in tqdm(files, desc=f"  {label}", unit="file"):
        if COUNTER.n >= TARGET_SAMPLES:
            break
        if f.suffix.lower() == ".wav":
            saved += chunk_and_save(f)
        else:
            if convert_to_wav_16k(f, tmp):
                saved += chunk_and_save(tmp)

    log.info(f"  [{label}] Saved {saved:,} chunks  (total so far: {COUNTER.n:,})")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Source 1 — LibriSpeech  (fully automated)
# ─────────────────────────────────────────────────────────────────────────────
LIBRISPEECH_URLS = {
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
    "dev-clean":       "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean":      "https://www.openslr.org/resources/12/test-clean.tar.gz",
}

def build_librispeech(splits: list[str] = None) -> int:
    """Download & process LibriSpeech splits. Default: train-clean-100."""
    log.info("═" * 60)
    log.info("SOURCE 1 — LibriSpeech")
    log.info("═" * 60)

    if splits is None:
        splits = ["train-clean-100"]   # ~28 GB, ~28k speakers

    total = 0
    for split in splits:
        url  = LIBRISPEECH_URLS[split]
        dest = DOWNLOAD_DIR / f"librispeech_{split}.tar.gz"
        extr = SCRATCH_DIR  / f"librispeech_{split}"

        if not extr.exists():
            ok = download_file(url, dest, split)
            if ok:
                extract_archive(dest, extr)
            else:
                log.warning(f"  Skipping {split} (download failed)")
                continue

        total += process_audio_files(extr, "**/*.flac", f"LibriSpeech/{split}")
        if COUNTER.n >= TARGET_SAMPLES:
            break

    return total


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — Mozilla Common Voice  (automated via official dataset download)
# ─────────────────────────────────────────────────────────────────────────────

def build_common_voice(cv_dir: Optional[Path] = None) -> int:
    """
    Process an already-downloaded Mozilla Common Voice corpus.

    Mozilla Common Voice requires free registration + dataset download at:
      https://commonvoice.mozilla.org/en/datasets

    Select a language (e.g. English) and download the .tar.gz bundle.
    Then re-run this script with --cv-dir pointing to the extracted folder
    that contains clips/ and validated.tsv.

    If cv_dir is not provided, this step prints instructions and returns 0.
    """
    log.info("═" * 60)
    log.info("SOURCE 2 — Mozilla Common Voice")
    log.info("═" * 60)

    if cv_dir is None or not cv_dir.exists():
        log.warning(
            "\n"
            "  ┌─ MANUAL STEP REQUIRED ──────────────────────────────────────┐\n"
            "  │  Mozilla Common Voice cannot be downloaded automatically.   │\n"
            "  │                                                               │\n"
            "  │  1. Go to https://commonvoice.mozilla.org/en/datasets        │\n"
            "  │  2. Create a free account and accept the licence.            │\n"
            "  │  3. Download the English dataset (cv-corpus-*.tar.gz).       │\n"
            "  │  4. Extract it to a local folder, e.g.:                     │\n"
            "  │       tar -xf cv-corpus-*.tar.gz -C downloads/common_voice   │\n"
            "  │  5. Re-run with:                                              │\n"
            "  │       python build_dataset.py --cv-dir downloads/common_voice│\n"
            "  └───────────────────────────────────────────────────────────────┘\n"
        )
        return 0

    # Prefer validated.tsv for quality-filtered clips
    tsv = cv_dir / "validated.tsv"
    clips_dir = cv_dir / "clips"

    if not tsv.exists() or not clips_dir.exists():
        log.warning(f"  Expected {tsv} and {clips_dir}. Check extracted path.")
        return 0

    log.info(f"  Reading validated.tsv from {cv_dir} …")
    mp3_files = []
    with open(tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mp3_files.append(clips_dir / row["path"])

    log.info(f"  Found {len(mp3_files):,} validated clips")
    saved = 0
    tmp = SCRATCH_DIR / "cv_tmp.wav"

    for mp3 in tqdm(mp3_files, desc="  CommonVoice", unit="file"):
        if COUNTER.n >= TARGET_SAMPLES:
            break
        if not mp3.exists():
            continue
        if convert_to_wav_16k(mp3, tmp):
            saved += chunk_and_save(tmp)

    log.info(f"  [CommonVoice] Saved {saved:,} chunks  (total so far: {COUNTER.n:,})")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Source 3 — VoxCeleb 1 & 2  (requires academic/free registration)
# ─────────────────────────────────────────────────────────────────────────────

def build_voxceleb(vox_dir: Optional[Path] = None) -> int:
    """
    Process a downloaded VoxCeleb corpus.

    VoxCeleb1 & 2 require free registration at:
      http://www.robots.ox.ac.uk/~vgg/data/voxceleb/

    After registration you receive download links/credentials.
    Download vox1_dev_wav.zip / vox2_dev_aac.zip etc., extract them,
    then pass --vox-dir to this script.

    Audio is real celebrity speech from YouTube — diverse, noisy,
    real-world conditions (ideal for fraud detection training).
    """
    log.info("═" * 60)
    log.info("SOURCE 3 — VoxCeleb")
    log.info("═" * 60)

    if vox_dir is None or not vox_dir.exists():
        log.warning(
            "\n"
            "  ┌─ MANUAL STEP REQUIRED ──────────────────────────────────────┐\n"
            "  │  VoxCeleb cannot be downloaded without registration.        │\n"
            "  │                                                               │\n"
            "  │  1. Register (free) at:                                      │\n"
            "  │       https://mm.kaist.ac.kr/datasets/voxceleb/              │\n"
            "  │  2. Download vox1_dev_wav.zip  (~39 GB, 148k utterances)     │\n"
            "  │     OR vox2_dev_aac.zip        (~84 GB, 1M+ utterances)      │\n"
            "  │  3. Extract:                                                  │\n"
            "  │       unzip vox1_dev_wav.zip -d downloads/voxceleb           │\n"
            "  │  4. Re-run with:                                              │\n"
            "  │       python build_dataset.py --vox-dir downloads/voxceleb   │\n"
            "  └───────────────────────────────────────────────────────────────┘\n"
        )
        return 0

    # VoxCeleb1 wav: wav/<id>/<video>/<utterance>.wav
    # VoxCeleb2 aac: dev/aac/<id>/<video>/<utterance>.m4a
    for ext in ("**/*.wav", "**/*.m4a", "**/*.mp4"):
        files = list(vox_dir.glob(ext))
        if files:
            break

    log.info(f"  Found {len(files):,} VoxCeleb files ({ext})")
    saved = 0
    tmp = SCRATCH_DIR / "vox_tmp.wav"

    for f in tqdm(files, desc="  VoxCeleb", unit="file"):
        if COUNTER.n >= TARGET_SAMPLES:
            break
        if f.suffix == ".wav":
            saved += chunk_and_save(f)
        else:
            if convert_to_wav_16k(f, tmp):
                saved += chunk_and_save(tmp)

    log.info(f"  [VoxCeleb] Saved {saved:,} chunks  (total so far: {COUNTER.n:,})")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Source 4 — CHiME Real Noisy Speech  (CHiME-4 / CHiME-5)
# ─────────────────────────────────────────────────────────────────────────────

def build_chime(chime_dir: Optional[Path] = None) -> int:
    """
    Process a downloaded CHiME real noisy speech corpus.

    CHiME-4 (real far-field, street / café / bus noise) is ideal for
    training fraud detectors that must cope with noisy call-centre audio.

    Download via the official challenge page:
      http://spandh.dcs.shef.ac.uk/chime_challenge/

    CHiME-4 real data (CHiME3/4 data package) ~12 GB.
    CHiME-5 requires signing a data agreement form (email request).

    Pass the extracted directory with --chime-dir.
    """
    log.info("═" * 60)
    log.info("SOURCE 4 — CHiME Real Noisy Speech")
    log.info("═" * 60)

    if chime_dir is None or not chime_dir.exists():
        log.warning(
            "\n"
            "  ┌─ MANUAL STEP REQUIRED ──────────────────────────────────────┐\n"
            "  │  CHiME data requires registration / email request.          │\n"
            "  │                                                               │\n"
            "  │  CHiME-4 (recommended for noise diversity):                 │\n"
            "  │    https://spandh.dcs.shef.ac.uk/chime_challenge/chime4/    │\n"
            "  │                                                               │\n"
            "  │  CHiME-5 (dinner-party recordings, very challenging):       │\n"
            "  │    https://spandh.dcs.shef.ac.uk/chime_challenge/chime5/    │\n"
            "  │                                                               │\n"
            "  │  After download and extraction, re-run:                      │\n"
            "  │    python build_dataset.py --chime-dir downloads/chime       │\n"
            "  └───────────────────────────────────────────────────────────────┘\n"
        )
        return 0

    # CHiME typically stores WAVs as 6-channel or single-channel .wav
    files = list(chime_dir.glob("**/*.wav"))
    log.info(f"  Found {len(files):,} CHiME WAV files")
    saved = 0

    for f in tqdm(files, desc="  CHiME", unit="file"):
        if COUNTER.n >= TARGET_SAMPLES:
            break
        saved += chunk_and_save(f, keep_noise=True)

    log.info(f"  [CHiME] Saved {saved:,} chunks  (total so far: {COUNTER.n:,})")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Bonus — TEDLIUM 3  (fully automated, conference talks, diverse accents)
# ─────────────────────────────────────────────────────────────────────────────
TEDLIUM3_URL = "https://www.openslr.org/resources/51/TEDLIUM_release-3.tgz"

def build_tedlium() -> int:
    """Download & process TEDLIUM 3 (optional bonus source, ~50k utterances)."""
    log.info("═" * 60)
    log.info("BONUS — TEDLIUM 3")
    log.info("═" * 60)

    dest = DOWNLOAD_DIR / "tedlium3.tgz"
    extr = SCRATCH_DIR  / "tedlium3"

    if not extr.exists():
        ok = download_file(TEDLIUM3_URL, dest, "TEDLIUM-3 (~9 GB)")
        if ok:
            extract_archive(dest, extr)
        else:
            log.warning("  Skipping TEDLIUM-3")
            return 0

    # SPH files inside TEDLIUM — convert via ffmpeg (sph2pipe not required)
    sph_files = list(extr.glob("**/*.sph"))
    log.info(f"  Found {len(sph_files):,} SPH files")
    saved = 0
    tmp = SCRATCH_DIR / "ted_tmp.wav"

    for f in tqdm(sph_files, desc="  TEDLIUM3", unit="file"):
        if COUNTER.n >= TARGET_SAMPLES:
            break
        if convert_to_wav_16k(f, tmp):
            saved += chunk_and_save(tmp)

    log.info(f"  [TEDLIUM3] Saved {saved:,} chunks  (total so far: {COUNTER.n:,})")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Bonus — AISHELL-1  (Mandarin, adds cross-lingual diversity)
# ─────────────────────────────────────────────────────────────────────────────
AISHELL_URL = "https://www.openslr.org/resources/33/data_aishell.tgz"

def build_aishell() -> int:
    log.info("═" * 60)
    log.info("BONUS — AISHELL-1 (Mandarin, 400 speakers)")
    log.info("═" * 60)

    dest = DOWNLOAD_DIR / "aishell.tgz"
    extr = SCRATCH_DIR  / "aishell"

    if not extr.exists():
        ok = download_file(AISHELL_URL, dest, "AISHELL-1 (~15 GB)")
        if ok:
            extract_archive(dest, extr)
        else:
            log.warning("  Skipping AISHELL-1")
            return 0

    total = process_audio_files(extr, "**/*.wav", "AISHELL-1")
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    files = sorted(OUTPUT_DIR.glob("*.wav"))
    n = len(files)
    sizes_mb = sum(f.stat().st_size for f in files) / 1e6
    log.info("")
    log.info("═" * 60)
    log.info("DATASET BUILD COMPLETE")
    log.info("═" * 60)
    log.info(f"  Output directory : {OUTPUT_DIR.resolve()}")
    log.info(f"  Total samples    : {n:,}")
    log.info(f"  Total disk size  : {sizes_mb:,.1f} MB")
    if n >= TARGET_SAMPLES:
        log.info(f"  ✅ Target of {TARGET_SAMPLES:,} samples REACHED!")
    else:
        log.info(f"  ⚠️  Target of {TARGET_SAMPLES:,} NOT yet reached ({n:,} collected).")
        log.info("     Add more sources or run with --all-librispeech")
    log.info("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Build a 100k+ real human voice dataset for AI Voice Fraud Detection."
    )
    p.add_argument("--cv-dir",    type=Path, default=None,
                   help="Path to extracted Mozilla Common Voice corpus")
    p.add_argument("--vox-dir",   type=Path, default=None,
                   help="Path to extracted VoxCeleb corpus (wav/ folder)")
    p.add_argument("--chime-dir", type=Path, default=None,
                   help="Path to extracted CHiME real-noisy corpus")
    p.add_argument("--all-librispeech", action="store_true",
                   help="Download ALL LibriSpeech splits (960 h instead of 100 h)")
    p.add_argument("--skip-tedlium",    action="store_true",
                   help="Skip TEDLIUM-3 bonus download")
    p.add_argument("--skip-aishell",    action="store_true",
                   help="Skip AISHELL-1 bonus download")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                   help=f"Where to save final WAVs (default: {OUTPUT_DIR})")
    return p.parse_args()


def main():
    args = parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir
    ensure_dirs()

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║   AI Voice Fraud Detection — Dataset Builder             ║")
    log.info("║   Target: 100,000 real human voice clips @ 16kHz mono   ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    if not ffmpeg_available():
        log.error("ffmpeg is required. Install: sudo apt install ffmpeg")
        sys.exit(1)

    # ── Automated sources first ───────────────────────────────────────────
    if COUNTER.n < TARGET_SAMPLES:
        splits = (
            ["train-clean-100", "train-clean-360", "train-other-500",
             "dev-clean", "test-clean"]
            if args.all_librispeech
            else ["train-clean-100"]
        )
        build_librispeech(splits)

    if COUNTER.n < TARGET_SAMPLES and not args.skip_tedlium:
        build_tedlium()

    if COUNTER.n < TARGET_SAMPLES and not args.skip_aishell:
        build_aishell()

    # ── Manual sources (if paths provided) ───────────────────────────────
    if COUNTER.n < TARGET_SAMPLES:
        build_common_voice(args.cv_dir)

    if COUNTER.n < TARGET_SAMPLES:
        build_voxceleb(args.vox_dir)

    if COUNTER.n < TARGET_SAMPLES:
        build_chime(args.chime_dir)

    print_summary()


if __name__ == "__main__":
    main()
