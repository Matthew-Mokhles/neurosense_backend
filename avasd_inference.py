"""
inference_pipeline.py
======================
Single-clip inference pipeline reproducing the exact recipe that achieved
test F1 = 0.5240 (results/best_result/best_result.json):

  1. Extract features (CLIP ViT-L/14, Whisper Large-v3 + ASR, E5-Large-v2,
     6-d acoustic timing) — same as extract_features_v15.py, one model on
     GPU at a time to fit a 6 GB card.
  2. Run v19 (3 seeds, d_model=192) on checkpoints/v19/seed_{42,123,777}
  3. Run v18 (3 seeds, d_model=256) on checkpoints/v18_trainval/seed_{42,123,777}
  4. Cross-ensemble: probs = sigmoid(0.4 * v19_logits + 0.6 * v18_logits)
  5. Apply FIXED thresholds copied verbatim from best_result.json
     (no threshold search at inference time — they are locked constants).

IMPORTANT — feature/timing dimension note
-------------------------------------------
extract_features_v15.py produces 16-d timing (6 acoustic + 10 linguistic).
train_v18.py / train_v19.py both TRIM this to the first 6 dims at load time
(linguistic features were found to hurt at this dataset size). We replicate
that trim here: extract all 16 dims, then slice [:, :, :6].

This module is import-only (no CLI); the FastAPI app in app.py calls
`run_inference(video_path)` and gets back per-label probabilities + binary
predictions using the locked thresholds.
"""

import os
import re
import string
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — paths are auto-derived from this file's location so the pipeline
# works regardless of where the project is cloned or what env vars are set.
# Override with AVASD_PROJECT_ROOT only if you genuinely need a different root.
# ─────────────────────────────────────────────────────────────────────────────

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_env_root    = os.environ.get('AVASD_PROJECT_ROOT', '').strip()
PROJECT_ROOT = _env_root if (_env_root and os.path.isdir(_env_root)) else _THIS_DIR
V19_CKPT_DIR   = os.path.join(PROJECT_ROOT, 'checkpoints', 'v19')
V18_CKPT_DIR   = os.path.join(PROJECT_ROOT, 'checkpoints', 'v18_trainval')
SEEDS          = [42, 123, 777]
W_V19          = 0.4          # cross-ensemble weight (locked from best_result.py)
NUM_FRAMES     = 16           # frames sampled per CLIP crop
NUM_TEMP_TOKENS = 12
CLIP_DURATION_S = 30.0        # fixed clip duration assumption (same as training)
SAMPLE_RATE     = 16000

LABELS = [
    'Absence or Avoidance of Eye Contact',
    'Aggressive Behavior',
    'Hyper- or Hyporeactivity to Sensory Input',
    'Non-Responsiveness to Verbal Interaction',
    'Non-Typical Language',
    'Object Lining-Up',
    'Self-Hitting or Self-Injurious Behavior',
    'Self-Spinning or Spinning Objects',
    'Upper Limb Stereotypies',
    'Background',
]
NUM_LABELS = len(LABELS)
NTL_IDX, OLU_IDX = 4, 5

# ── FIXED THRESHOLDS — copied verbatim from results/best_result/best_result.json
# These are LOCKED constants. Do not re-derive them at inference time;
# that was exactly the leakage mistake this project spent a long time fixing.
FIXED_THRESHOLDS = {
    'Absence or Avoidance of Eye Contact':       0.14620690047740936,
    'Aggressive Behavior':                       0.3632911443710327,
    'Hyper- or Hyporeactivity to Sensory Input':  0.45822784304618835,
    'Non-Responsiveness to Verbal Interaction':   0.4392405152320862,
    'Non-Typical Language':                       0.4962025284767151,
    'Object Lining-Up':                           0.4867088496685028,
    'Self-Hitting or Self-Injurious Behavior':    0.41075947880744934,
    'Self-Spinning or Spinning Objects':          0.7620252966880798,
    'Upper Limb Stereotypies':                    0.35379746556282043,
    'Background':                                 0.3063291013240814,
}
THRESH_VEC = np.array([FIXED_THRESHOLDS[l] for l in LABELS], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSCRIPT CLEANING + LINGUISTIC FEATURES  (copied from transcript_utils.py)
# ══════════════════════════════════════════════════════════════════════════════

LING_DIM = 16
_FILLER_WORDS = {'um', 'uh', 'hmm', 'ah', 'er', 'like', 'you know'}
_ASR_ARTIFACTS = [
    r'\[.*?\]', r'\(.*?\)', r'<.*?>', r'\*+', r'\.{3,}',
    r'(?<!\w)-(?!\w)', r'\s{2,}',
]
_UNICODE_FIXUPS = {
    '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
    '\u2014': ' ', '\u2013': ' ', '\u00e2\u0080\u0099': "'",
    '\xa0': ' ', '\t': ' ',
}


def clean_transcript(text: str) -> str:
    if not text or not text.strip():
        return ''
    for src, dst in _UNICODE_FIXUPS.items():
        text = text.replace(src, dst)
    for pattern in _ASR_ARTIFACTS:
        text = re.sub(pattern, ' ', text)
    text = ''.join(c for c in text if c.isprintable())
    text = re.sub(r'([.!?])\1+', r'\1', text)
    text = re.sub(r'(?<=\w),(?=\w)', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    meaningful = re.sub(r'[^a-z0-9\s]', '', text).strip()
    if len(meaningful) < 2:
        return ''
    return text


def _compute_audio_stats(audio_np, sr=16000, silence_thresh=0.01):
    total_samples = max(len(audio_np), 1)
    clip_duration_s = total_samples / sr
    is_silent = np.abs(audio_np) < silence_thresh
    silence_frac = float(np.mean(is_silent))
    rms = float(np.sqrt(np.mean(audio_np.astype(np.float64) ** 2) + 1e-9))

    min_short_pause = int(0.1 * sr)
    min_long_pause = int(0.5 * sr)
    pause_durations, n_long_pauses, run_len = [], 0, 0
    for s in is_silent:
        if s:
            run_len += 1
        else:
            if run_len >= min_short_pause:
                pause_durations.append(run_len / sr)
            if run_len >= min_long_pause:
                n_long_pauses += 1
            run_len = 0
    if run_len >= min_short_pause:
        pause_durations.append(run_len / sr)
    if run_len >= min_long_pause:
        n_long_pauses += 1
    avg_pause_dur = float(np.mean(pause_durations)) if pause_durations else 0.0
    return silence_frac, n_long_pauses, avg_pause_dur, clip_duration_s, rms


def extract_linguistic_features(text, audio_np, sr=16000, silence_thresh=0.01):
    feats = np.zeros(LING_DIM, dtype=np.float32)
    silence_frac, n_long_pauses, avg_pause_dur, clip_duration_s, rms = \
        _compute_audio_stats(audio_np, sr, silence_thresh)
    feats[13] = np.clip(silence_frac, 0, 1)
    feats[14] = float(min(n_long_pauses, 20))
    feats[15] = np.clip(rms * 10.0, 0, 1)

    if not text or not text.strip():
        return feats

    raw_words = text.split()
    words = [w.strip(string.punctuation) for w in raw_words]
    words = [w for w in words if w]
    if not words:
        return feats

    n_words, n_unique = len(words), len(set(words))
    feats[0] = float(min(n_words, 200))
    feats[1] = float(n_unique) / float(n_words)

    if n_words >= 2:
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(n_words - 1)]
        feats[2] = float(len(set(bigrams))) / float(len(bigrams))
    else:
        feats[2] = 1.0

    rep_count = sum(1 for i in range(1, n_words) if words[i] == words[i - 1])
    feats[3] = float(min(rep_count, 20))
    feats[4] = np.clip(float(np.mean([len(w) for w in words])), 0, 15)
    if clip_duration_s > 0:
        feats[5] = np.clip((n_words / clip_duration_s) * 60.0, 0, 300)

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    n_sent = max(len(sentences), 1)
    feats[6] = float(min(n_sent, 30))
    n_excl = text.count('!') + text.count('?')
    feats[7] = np.clip(float(n_excl) / float(n_sent), 0, 5)

    if n_words >= 2:
        bi_counts = Counter(bigrams)
        echolalia = sum(c - 1 for c in bi_counts.values() if c > 1)
        feats[8] = np.clip(float(echolalia) / float(max(len(bigrams), 1)), 0, 1)
    else:
        feats[8] = 0.0

    filler_count = sum(1 for w in words if w in _FILLER_WORDS)
    feats[9] = np.clip(float(filler_count) / float(n_words), 0, 1)
    feats[10] = np.clip(avg_pause_dur, 0, 10)

    if n_sent >= 2:
        sent_reps = sum(1 for i in range(1, len(sentences))
                        if sentences[i].strip() == sentences[i - 1].strip())
        feats[11] = float(sent_reps) / float(n_sent - 1)
    else:
        feats[11] = 0.0

    word_counts = Counter(words)
    total = sum(word_counts.values())
    probs_w = np.array([c / total for c in word_counts.values()], dtype=np.float64)
    entropy = float(-np.sum(probs_w * np.log(probs_w + 1e-12)))
    max_entropy = float(np.log(n_unique + 1e-12))
    feats[12] = np.clip(entropy / max(max_entropy, 1e-6), 0, 1)
    return feats


def extract_linguistic_features_per_token(text, segments, audio_np, sr=16000,
                                          num_tokens=12, clip_duration_s=30.0,
                                          silence_thresh=0.01):
    token_feats = np.zeros((num_tokens, LING_DIM), dtype=np.float32)
    win_dur = clip_duration_s / num_tokens
    total_samp = len(audio_np)

    for t in range(num_tokens):
        w_s, w_e = t * win_dur, (t + 1) * win_dur
        s_smp, e_smp = int(w_s * sr), min(int(w_e * sr), total_samp)
        win_audio = audio_np[s_smp:e_smp] if e_smp > s_smp else np.zeros(1)

        if segments:
            win_texts = []
            for seg in segments:
                seg_s = float(seg.get('start', 0.0))
                seg_e = float(seg.get('end', seg_s + 0.1))
                seg_t = seg.get('text', '').strip()
                if seg_t and seg_s < w_e and seg_e > w_s:
                    win_texts.append(seg_t)
            win_text = clean_transcript(' '.join(win_texts))
        else:
            win_text = clean_transcript(text)

        token_feats[t] = extract_linguistic_features(win_text, win_audio, sr, silence_thresh)
    return token_feats


# ══════════════════════════════════════════════════════════════════════════════
#  RAW CLIP LOADING  (video frames + audio waveform from an arbitrary file)
# ══════════════════════════════════════════════════════════════════════════════

def _get_video_duration_s(video_path, ffmpeg_exe):
    """
    Parses 'Duration: HH:MM:SS.cc' out of ffmpeg's stderr when probing a
    file with no output specified. ffmpeg always exits non-zero in this
    mode (there's no output), which is expected — we only care about the
    stderr text, not the return code.
    """
    import subprocess
    result = subprocess.run(
        [ffmpeg_exe, '-i', video_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr_text = result.stderr.decode(errors='ignore')
    m = re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)', stderr_text)
    if not m:
        return None
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def load_video_frames(video_path, num_frames=NUM_FRAMES, frame_size=(224, 224)):
    """
    Returns a (C, T, H, W) float tensor, ImageNet-normalised, matching
    AVASDDataset._load_video()'s val/test (no-augmentation) behaviour.

    Frame extraction goes entirely through ffmpeg (one '-frames:v 1' call
    per sampled timestamp) instead of cv2.VideoCapture. On this host,
    cv2.VideoCapture — regardless of backend (MSMF or even CAP_FFMPEG) —
    has been crashing the whole process with a native access violation in
    an untracked thread (no Python traceback at all, just "Windows fatal
    exception: access violation"). ffmpeg is already a hard dependency
    here for audio extraction and has proven reliable, so frame grabs are
    routed through the same known-good subprocess path; only cv2.imread
    (plain static-image decoding, unrelated to any video/codec/hardware-
    acceleration code path) and cv2.resize are still used, on files ffmpeg
    already wrote to disk.
    """
    import cv2
    import subprocess
    import tempfile
    import shutil as _shutil
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ffmpeg_exe = _find_ffmpeg()
    duration_s = _get_video_duration_s(video_path, ffmpeg_exe)
    if not duration_s or duration_s <= 0:
        duration_s = CLIP_DURATION_S  # fallback to the training-time assumption

    # Evenly spaced timestamps across the clip — same sampling intent as
    # the old step = total_frames / num_frames logic, just in seconds
    # instead of frame indices (ffmpeg seeks by time, not frame number).
    timestamps = [duration_s * i / num_frames for i in range(num_frames)]

    tmp_dir = tempfile.mkdtemp(prefix='avasd_frames_')
    try:
        frame_paths = [None] * num_frames
        for i, ts in enumerate(timestamps):
            out_path = os.path.join(tmp_dir, f'frame_{i:03d}.png')
            cmd = [
                ffmpeg_exe, '-y', '-ss', f'{ts:.3f}', '-i', video_path,
                '-frames:v', '1', '-q:v', '2', out_path,
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0 and os.path.isfile(out_path):
                frame_paths[i] = out_path

        frames = []
        last_good = None
        for i, p in enumerate(frame_paths):
            if p is not None:
                img = cv2.imread(p)  # static-image decode only, BGR
                if img is not None:
                    last_good = img
                    frames.append(img)
                    continue
            # Extraction failed for this timestamp (e.g. seek past EOF on a
            # short/odd clip) — reuse the last successfully grabbed frame,
            # or a black frame if none have succeeded yet, matching the
            # old code's padding behaviour.
            if last_good is not None:
                frames.append(last_good)
            else:
                frames.append(np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8))

        if all(p is None for p in frame_paths):
            raise ValueError(f"Could not extract any frames from: {video_path}")
    finally:
        _shutil.rmtree(tmp_dir, ignore_errors=True)

    frames_tensor = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_size[1], frame_size[0]), interpolation=cv2.INTER_LINEAR)
        frames_tensor.append(transform(frame))

    # (T, C, H, W) -> (C, T, H, W)
    return torch.stack(frames_tensor, dim=1)


_ffmpeg_cache = None

def _find_ffmpeg():
    global _ffmpeg_cache
    if _ffmpeg_cache is not None:
        return _ffmpeg_cache

    import shutil

    # 1. Explicit override
    env_path = os.environ.get('FFMPEG_PATH', '').strip()
    if env_path and os.path.isfile(env_path):
        _ffmpeg_cache = env_path
        return env_path

    # 2. PATH lookup (no subprocess, safe)
    p = shutil.which('ffmpeg')
    if p:
        _ffmpeg_cache = p
        return p

    # 3. Common Windows install paths
    common_paths = [
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
        os.path.join(os.environ.get('USERPROFILE', ''), 'ffmpeg', 'bin', 'ffmpeg.exe'),
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'ffmpeg', 'bin', 'ffmpeg.exe'),
    ]
    for p in common_paths:
        if os.path.isfile(p):
            _ffmpeg_cache = p
            return p

    # 4. imageio-ffmpeg bundled binary — resolve the path directly without
    #    calling get_ffmpeg_exe() which spawns a validation subprocess
    #    that crashes nvdxgdmal64.dll on this host.
    try:
        from imageio_ffmpeg._utils import _get_bin_dir, FNAME_PER_PLATFORM, get_platform
        p = os.path.join(_get_bin_dir(), FNAME_PER_PLATFORM.get(get_platform(), ''))
        if p and os.path.isfile(p):
            _ffmpeg_cache = p
            return p
    except Exception:
        pass

    raise FileNotFoundError(
        "ffmpeg not found. Install it and either:\n"
        "  a) Add it to your system PATH, or\n"
        "  b) Set the FFMPEG_PATH environment variable to the ffmpeg.exe path, or\n"
        "  c) Install imageio-ffmpeg:  pip install imageio-ffmpeg"
    )


def load_audio_waveform(video_path, target_sr=SAMPLE_RATE,
                        target_length_s=CLIP_DURATION_S):
    """
    Extracts mono audio at 16kHz from a video file using ffmpeg via
    torchaudio/imageio-ffmpeg, padded/truncated to target_length_s.
    Returns a 1-D float32 tensor of shape (target_sr * target_length_s,).
    """
    import subprocess
    import tempfile
    import wave

    ffmpeg_exe = _find_ffmpeg()
    target_length = int(target_sr * target_length_s)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_wav = tmp.name

    try:
        cmd = [
            ffmpeg_exe, '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(target_sr), '-ac', '1',
            tmp_wav,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='ignore')[:500]}")

        # ffmpeg above already wrote mono pcm_s16le at target_sr, so we can
        # read the WAV directly with the stdlib instead of torchaudio.load()
        # — newer torchaudio (2.9+) routes .load() through the optional
        # torchcodec package, which needs system FFmpeg shared libraries
        # that aren't set up on this box. Reading PCM16 mono ourselves
        # avoids that dependency entirely.
        with wave.open(tmp_wav, 'rb') as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
        pcm = np.frombuffer(raw, dtype=np.int16)
        if n_channels > 1:
            pcm = pcm.reshape(-1, n_channels).mean(axis=1)
        waveform = torch.from_numpy(pcm.astype(np.float32) / 32768.0)
        if sr != target_sr:
            waveform = torch.from_numpy(
                np.interp(
                    np.linspace(0, len(pcm) - 1, int(len(pcm) * target_sr / sr)),
                    np.arange(len(pcm)), pcm.astype(np.float32) / 32768.0,
                ).astype(np.float32)
            )

        current_length = waveform.shape[0]
        if current_length > target_length:
            waveform = waveform[:target_length]
        elif current_length < target_length:
            waveform = F.pad(waveform, (0, target_length - current_length))
        return waveform.float()
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


# ══════════════════════════════════════════════════════════════════════════════
#  ENCODERS — loaded one at a time to fit 6 GB VRAM (same pattern as training)
# ══════════════════════════════════════════════════════════════════════════════

def _unload(model):
    if model is None:
        return
    model.cpu()
    del model
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL CACHE — populated once by preload_models(), kept on CPU RAM.
#  Encoders are moved to GPU one at a time at inference time (same VRAM-safe
#  pattern as before) but are no longer re-downloaded / re-built from
#  from_pretrained() or re-parsed from .pth on every single request.
# ══════════════════════════════════════════════════════════════════════════════

_CACHE = {
    'clip_model': None, 'clip_proc': None,
    'whisper_feat_ext': None, 'whisper_encoder': None,
    'whisper_asr': None,
    'e5_tok': None, 'e5_model': None,
    'v19_models': None,   # list of dicts: {'model', 'temperature', 'it_dim'} per seed
    'v18_models': None,
}


def _build_model_from_ckpt(ckpt_path):
    """Loads one checkpoint and returns (model_on_cpu, temperature, it_dim)."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['model']
    use_la, use_lc, use_aux, d_model, it_dim, ntl_hidden, olu_hidden = _detect_flags(state)

    model = TripleModalASDModel(
        v_dim=1024, a_dim=1280, t_dim=1024,
        d_model=d_model, nhead=4, num_enc_layers=2,
        proj_dropout=0.07, enc_dropout=0.15, drop_path_rate=0.15,
        use_label_attn=use_la, use_label_corr=use_lc,
        use_aux_heads=use_aux, timing_dim=it_dim,
        ntl_hidden=ntl_hidden, olu_hidden=olu_hidden,
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    temperature = ckpt.get('temperature', None)
    return model, temperature, it_dim


def _load_model_family(ckpt_dir, seeds):
    """Loads every seed checkpoint in a family once, kept on CPU until used."""
    loaded = []
    for seed in seeds:
        seed_dir = os.path.join(ckpt_dir, f'seed_{seed}')
        ckpt_path = find_checkpoint(seed_dir)
        if ckpt_path is None:
            print(f'  [WARN] No checkpoint found for seed_{seed} in {ckpt_dir} — skipping')
            continue
        model, temperature, it_dim = _build_model_from_ckpt(ckpt_path)
        loaded.append({'model': model, 'temperature': temperature, 'it_dim': it_dim})
    if not loaded:
        raise RuntimeError(f'No checkpoints found in {ckpt_dir} for seeds {seeds}')
    return loaded


def preload_models(device=None):
    """
    Call once at app startup (e.g. from app.py's lifespan handler).

    Loads every model this pipeline needs into the module-level _CACHE:
      - CLIP ViT-L/14 (vision encoder + processor)
      - Whisper Large-v3 encoder + feature extractor
      - Whisper Large-v3 ASR
      - E5-Large-v2 tokenizer + model
      - All v19 seed checkpoints (3)
      - All v18 seed checkpoints (3)

    Everything is loaded onto CPU RAM here (not GPU) — this card has 6 GB
    VRAM, not enough to hold CLIP + 2x Whisper + E5 + 6 classifier heads
    simultaneously. What this buys you instead is removing every
    from_pretrained()/torch.load() disk-and-parse cost from the request
    path: that disk I/O + deserialization is the slow part on every call
    today. Per-request inference still moves one encoder/model to GPU at
    a time, runs it, and unloads — same sequential VRAM-safe pattern as
    before, just against already-resident-in-RAM weights instead of
    re-reading from disk every time.
    """
    import time
    t0 = time.time()

    print('[preload] CLIP ViT-L/14 ...')
    from transformers import CLIPVisionModel, CLIPImageProcessor
    _CACHE['clip_model'] = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-large-patch14").eval()
    _CACHE['clip_proc'] = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14")

    print('[preload] Whisper Large-v3 encoder ...')
    from transformers import WhisperModel, WhisperFeatureExtractor
    _CACHE['whisper_feat_ext'] = WhisperFeatureExtractor.from_pretrained(
        'openai/whisper-large-v3')
    whisper_full = WhisperModel.from_pretrained('openai/whisper-large-v3')
    _CACHE['whisper_encoder'] = whisper_full.encoder.float().eval()
    del whisper_full

    print('[preload] Whisper Large-v3 ASR ...')
    import whisper as whisper_lib
    _CACHE['whisper_asr'] = whisper_lib.load_model('large-v3', device='cpu')

    print('[preload] E5-Large-v2 ...')
    from transformers import AutoTokenizer, AutoModel
    _CACHE['e5_tok'] = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
    _CACHE['e5_model'] = AutoModel.from_pretrained('intfloat/e5-large-v2').eval()

    print('[preload] v19 checkpoints (3 seeds) ...')
    _CACHE['v19_models'] = _load_model_family(V19_CKPT_DIR, SEEDS)

    print('[preload] v18 checkpoints (3 seeds) ...')
    _CACHE['v18_models'] = _load_model_family(V18_CKPT_DIR, SEEDS)

    print(f'[preload] Done in {time.time() - t0:.1f}s — all weights resident in RAM.')


@torch.no_grad()
def extract_clip_features(video_tensor, device, num_tokens=NUM_TEMP_TOKENS, topk_k=4):
    """video_tensor: (C, T, H, W) -> returns (num_tokens, 1024)."""
    import PIL

    # Use preloaded weights if available (preload_models() was called at
    # startup); otherwise fall back to loading from disk/hub (slow path).
    if _CACHE['clip_model'] is not None:
        model = _CACHE['clip_model'].to(device)
        proc = _CACHE['clip_proc']
    else:
        from transformers import CLIPVisionModel, CLIPImageProcessor
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    C, T, H, W = video_tensor.shape
    n_frames = 16
    idxs = torch.linspace(0, T - 1, n_frames).long()

    frames_np = []
    for i in idxs:
        f = video_tensor[:, i].permute(1, 2, 0).cpu().numpy()
        f = f * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        frames_np.append(np.clip(f * 255, 0, 255).astype(np.uint8))

    pil_frames = [PIL.Image.fromarray(f) for f in frames_np]
    inputs = proc(images=pil_frames, return_tensors='pt')
    out = model(pixel_values=inputs['pixel_values'].to(device))
    hidden = out.last_hidden_state  # (n_frames, patches+1, 1024)

    def pool_topk(patches, k=topk_k):
        if patches.shape[0] <= 1:
            return patches.mean(0)
        idx = patches.norm(dim=1).topk(min(k, patches.shape[0])).indices
        return patches[idx].mean(0)

    frame_tokens = torch.stack([pool_topk(hidden[fi, 1:, :]) for fi in range(n_frames)])
    if _CACHE['clip_model'] is not None:
        model.cpu()
        torch.cuda.empty_cache()
    else:
        _unload(model)

    if n_frames == num_tokens:
        return frame_tokens.cpu()
    indices = torch.linspace(0, n_frames - 1, num_tokens).long()
    return frame_tokens[indices].cpu()


@torch.no_grad()
def extract_audio_and_text_features(audio_tensor, device, num_tokens=NUM_TEMP_TOKENS):
    """
    Returns (audio_feats (T,1280), text_feats (T,1024), timing_6d (T,6), transcript str).
    Loads Whisper encoder + Whisper ASR + E5 sequentially, unloading between each.
    Uses preloaded weights from _CACHE when preload_models() has been run.
    """
    TARGET = SAMPLE_RATE * 30
    audio_np = np.pad(audio_tensor.cpu().float().numpy(),
                      (0, max(0, TARGET - len(audio_tensor))))[:TARGET]

    preloaded = _CACHE['whisper_encoder'] is not None

    # ── Whisper encoder: dense audio features ──────────────────────────────
    if preloaded:
        feat_ext = _CACHE['whisper_feat_ext']
        encoder = _CACHE['whisper_encoder'].to(device)
    else:
        from transformers import WhisperModel, WhisperFeatureExtractor
        feat_ext = WhisperFeatureExtractor.from_pretrained('openai/whisper-large-v3')
        whisper_model = WhisperModel.from_pretrained('openai/whisper-large-v3')
        encoder = whisper_model.encoder.to(device).float().eval()

    inputs = feat_ext(audio_np, sampling_rate=SAMPLE_RATE, return_tensors='pt',
                      padding='max_length')
    out = encoder(inputs.input_features.to(device).float())
    hidden = out.last_hidden_state.squeeze(0)
    seq_len = hidden.shape[0]
    win = max(1, seq_len // num_tokens)
    audio_tokens = [hidden[t * win:(t + 1) * win if t < num_tokens - 1 else seq_len].mean(0)
                    for t in range(num_tokens)]
    audio_feats = torch.stack(audio_tokens).float().cpu()
    if preloaded:
        encoder.cpu()
        torch.cuda.empty_cache()
    else:
        _unload(encoder)
        del whisper_model, feat_ext

    # ── Whisper ASR: transcription + segments ───────────────────────────────
    if preloaded:
        asr_model = _CACHE['whisper_asr'].to(device)
    else:
        import whisper as whisper_lib
        asr_model = whisper_lib.load_model('large-v3').to(device)

    result = asr_model.transcribe(audio_np, language='en', fp16=False,
                                  word_timestamps=False)
    raw_text = result.get('text', '').strip()
    segments = result.get('segments', [])
    if preloaded:
        asr_model.cpu()
        torch.cuda.empty_cache()
    else:
        _unload(asr_model)

    cleaned = clean_transcript(raw_text)

    # ── Timing features (6-d acoustic) ───────────────────────────────────
    timing_6d = _extract_timing_features_6d(audio_np, num_tokens)

    # ── E5 text features ────────────────────────────────────────────────────
    if preloaded:
        tok = _CACHE['e5_tok']
        text_model = _CACHE['e5_model'].to(device)
    else:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
        text_model = AutoModel.from_pretrained('intfloat/e5-large-v2').to(device).eval()
    t_dim = text_model.config.hidden_size

    if cleaned.strip():
        prefixed = f'passage: {cleaned}'
        enc = tok(prefixed, return_tensors='pt', padding=True,
                 truncation=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}
        emb = text_model(**enc).last_hidden_state.mean(dim=1).squeeze(0)
        vec = F.normalize(emb, dim=-1).cpu()
    else:
        vec = torch.zeros(t_dim)
    text_feats = vec.unsqueeze(0).expand(num_tokens, -1).clone()
    if preloaded:
        text_model.cpu()
        torch.cuda.empty_cache()
    else:
        _unload(text_model)

    return audio_feats, text_feats, timing_6d, cleaned, segments


def _extract_timing_features_6d(audio_np, num_tokens=NUM_TEMP_TOKENS,
                                silence_thresh=0.01):
    """Identical to extract_timing_features() in extract_features_v15.py."""
    total = len(audio_np)
    win_len = max(1, total // num_tokens)
    sr = SAMPLE_RATE
    features, prev_speech = [], False
    for t in range(num_tokens):
        s = t * win_len
        e = s + win_len if t < num_tokens - 1 else total
        w = audio_np[s:e]
        silence_ratio = float(np.mean(np.abs(w) < silence_thresh))
        rms = float(np.sqrt(np.mean(w ** 2) + 1e-9))
        frame_sz = max(1, len(w) // 8)
        frame_rms = [np.sqrt(np.mean(w[i:i + frame_sz] ** 2) + 1e-9)
                    for i in range(0, len(w) - frame_sz, frame_sz)]
        speaking_rate = float(np.std(frame_rms)) if len(frame_rms) > 1 else 0.0
        resp_latency = 0.0 if prev_speech else 1.0
        is_sil = np.abs(w) < silence_thresh
        max_sil = cur = 0
        for v in is_sil:
            if v:
                cur += 1
                max_sil = max(max_sil, cur)
            else:
                cur = 0
        pause_dur = float(max_sil) / sr
        mid = len(w) // 2
        e1 = float(np.sqrt(np.mean(w[:mid] ** 2) + 1e-9))
        e2 = float(np.sqrt(np.mean(w[mid:] ** 2) + 1e-9))
        turn_end = float(np.clip((e1 - e2) / (e1 + 1e-9), -1.0, 1.0))
        features.append(torch.tensor(
            [silence_ratio, rms, speaking_rate, resp_latency, pause_dur, turn_end],
            dtype=torch.float32))
        prev_speech = (1.0 - silence_ratio) > 0.3
    return torch.stack(features)


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def extract_features_for_clip(video_path, device):
    """
    Runs the full sequential 3-pass extraction for ONE clip:
      Pass 1 — CLIP ViT-L/14   (video, all 3 temporal crops, like training)
      Pass 2 — Whisper Large-v3 (audio + ASR)
      Pass 3 — E5-Large-v2      (text from transcript)

    Returns a dict matching the cache format expected by the models:
      video_feats_c0/c1/c2 : (1, T, 1024)
      audio_feats          : (1, T, 1280)
      text_feats           : (1, T, 1024)
      timing_feats         : (1, T, 6)      <- already trimmed to 6-d
      transcript           : str
    """
    print('  [1/3] Loading raw video + audio ...')
    video_tensor = load_video_frames(video_path, num_frames=NUM_FRAMES)
    audio_tensor = load_audio_waveform(video_path)

    print('  [2/3] CLIP ViT-L/14 — extracting center crop ...')
    T = video_tensor.shape[1]
    if T > 16:
        start = max(0, (T - 16) // 2)
        vt = video_tensor[:, start:start + 16]
    else:
        vt = video_tensor
    center_crop = extract_clip_features(vt, device)

    print('  [3/3] Whisper Large-v3 (audio+ASR) -> E5-Large-v2 (text) ...')
    audio_feats, text_feats, timing_6d, transcript, segments = \
        extract_audio_and_text_features(audio_tensor, device)

    return {
        'video_feats_c0': center_crop.unsqueeze(0),
        'audio_feats':    audio_feats.unsqueeze(0),
        'text_feats':     text_feats.unsqueeze(0),
        'timing_feats':   timing_6d.unsqueeze(0),   # already 6-d
        'transcript':     transcript,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS — v18 (d_model=256) and v19 (d_model=192)
#  Identical to train_v18.py / train_v19.py TripleModalASDModel.
# ══════════════════════════════════════════════════════════════════════════════

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        rand_tensor.floor_()
        return x.div(keep_prob) * rand_tensor


class CrossModalAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, query, key_value):
        ctx, _ = self.cross(query=query, key=key_value, value=key_value)
        return self.norm(query + self.drop_path(self.drop(ctx)))


class TemporalConv(nn.Module):
    def __init__(self, d_model, kernel=3, drop_path=0.0):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel,
                              padding=kernel // 2, groups=d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + self.drop_path(out))


class GatedFusion(nn.Module):
    def __init__(self, d_model, num_modalities=4, dropout=0.1):
        super().__init__()
        self.num_modalities = num_modalities
        self.gate = nn.Sequential(
            nn.Linear(d_model * num_modalities, d_model), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(d_model, num_modalities))

    def forward(self, *mods):
        B = mods[0].shape[0]
        stacked = torch.stack(mods, dim=1)
        flat = stacked.mean(dim=2).view(B, -1)
        gates = torch.softmax(self.gate(flat), dim=-1).view(B, self.num_modalities, 1, 1)
        return (gates * stacked).sum(dim=1)


class LabelSpecificAttentionHead(nn.Module):
    def __init__(self, d_model, num_labels):
        super().__init__()
        self.label_attn = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(),
                          nn.Linear(d_model // 2, 1))
            for _ in range(num_labels)])
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, memory):
        preds = []
        for i, scorer in enumerate(self.label_attn):
            attn = torch.softmax(scorer(memory), dim=1)
            pooled = (memory * attn).sum(dim=1)
            preds.append(self.classifier.weight[i] @ pooled.T + self.classifier.bias[i])
        return torch.stack(preds, dim=1)


class AuxAttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.score = nn.Linear(d_model, 1, bias=False)

    def forward(self, x):
        attn = torch.softmax(self.score(x), dim=1)
        return (x * attn).sum(dim=1)


class AuxHead(nn.Module):
    def __init__(self, n_modalities, d_model, hidden_dim=96, dropout=0.2):
        super().__init__()
        self.pools = nn.ModuleList([AuxAttentionPool(d_model) for _ in range(n_modalities)])
        self.mlp = nn.Sequential(
            nn.Linear(d_model * n_modalities, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, *streams):
        pooled = [p(s) for p, s in zip(self.pools, streams)]
        return self.mlp(torch.cat(pooled, dim=-1)).squeeze(-1)


class TripleModalASDModel(nn.Module):
    """Used for BOTH v18 (d_model=256) and v19 (d_model=192) — same architecture,
    only width differs. Matches train_v18.py / train_v19.py exactly."""

    def __init__(self, v_dim, a_dim, t_dim, d_model=256, nhead=4,
                num_enc_layers=2, proj_dropout=0.07, enc_dropout=0.15,
                num_labels=NUM_LABELS, num_tokens=NUM_TEMP_TOKENS,
                drop_path_rate=0.15, timing_dim=6,
                use_label_attn=True, use_label_corr=True, use_aux_heads=True,
                ntl_hidden=96, olu_hidden=48):
        super().__init__()
        self.use_label_corr = use_label_corr
        self.use_aux_heads = use_aux_heads

        self.v_proj = nn.Sequential(nn.Linear(v_dim, d_model), nn.GELU(), nn.Dropout(proj_dropout))
        self.a_proj = nn.Sequential(nn.Linear(a_dim, d_model), nn.GELU(), nn.Dropout(proj_dropout))
        self.t_proj = nn.Sequential(nn.Linear(t_dim, d_model), nn.GELU(), nn.Dropout(proj_dropout))
        self.it_proj = nn.Sequential(nn.Linear(timing_dim, d_model), nn.GELU(), nn.Dropout(proj_dropout))

        self.pos_enc = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)
        self.v_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.a_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.t_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.it_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.v2a = CrossModalAttention(d_model, nhead, proj_dropout, drop_path_rate)
        self.v2t = CrossModalAttention(d_model, nhead, proj_dropout, drop_path_rate)
        self.a2t = CrossModalAttention(d_model, nhead, proj_dropout, drop_path_rate)

        self.temp_conv = TemporalConv(d_model, kernel=3, drop_path=drop_path_rate)
        self.gated_fusion = GatedFusion(d_model, num_modalities=4, dropout=enc_dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=enc_dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        self.head = (LabelSpecificAttentionHead(d_model, num_labels) if use_label_attn
                    else None)

        if use_label_corr:
            self.label_corr = nn.Parameter(torch.zeros(num_labels, num_labels))

        if use_aux_heads:
            self.ntl_head = AuxHead(n_modalities=3, d_model=d_model, hidden_dim=ntl_hidden)
            self.olu_head = AuxHead(n_modalities=2, d_model=d_model, hidden_dim=olu_hidden)

    def forward(self, vf, af, tf, itf):
        vp = self.v_proj(vf) + self.pos_enc + self.v_type
        ap = self.a_proj(af) + self.pos_enc + self.a_type
        tp = self.t_proj(tf) + self.pos_enc + self.t_type
        itp = self.it_proj(itf) + self.pos_enc + self.it_type

        vp = self.v2a(vp, ap); ap = self.v2a(ap, vp)
        vp = self.v2t(vp, tp); tp = self.v2t(tp, vp)
        ap = self.a2t(ap, tp); tp = self.a2t(tp, ap)

        fused = self.gated_fusion(vp, ap, tp, itp)
        memory = self.encoder(self.temp_conv(fused))
        logits = self.head(memory)

        if self.use_label_corr:
            logits = logits + (self.label_corr @ torch.sigmoid(logits.detach()).T).T

        aux_ntl, aux_olu = None, None
        if self.use_aux_heads and self.training:
            aux_ntl = self.ntl_head(ap, tp, itp)
            aux_olu = self.olu_head(vp, itp)

        return logits, aux_ntl, aux_olu


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT LOADING — auto-detect architecture flags from state_dict keys
# ══════════════════════════════════════════════════════════════════════════════

def _detect_flags(state):
    keys = list(state.keys())
    use_la = any('head.label_attn' in k for k in keys)
    use_lc = any(k == 'label_corr' for k in keys)
    use_aux = any(k.startswith('ntl_head.') or k.startswith('olu_head.') for k in keys)
    d_model = state['v_proj.0.weight'].shape[0]
    it_dim = state['it_proj.0.weight'].shape[1]
    # Auto-detect aux head hidden dims from checkpoint weights so v18
    # (ntl=128, olu=64) and v19 (ntl=96, olu=48) both load correctly.
    ntl_hidden = int(state['ntl_head.mlp.0.weight'].shape[0]) if use_aux and 'ntl_head.mlp.0.weight' in state else 96
    olu_hidden = int(state['olu_head.mlp.0.weight'].shape[0]) if use_aux and 'olu_head.mlp.0.weight' in state else 48
    return use_la, use_lc, use_aux, d_model, it_dim, ntl_hidden, olu_hidden


def find_checkpoint(seed_dir):
    """Priority order matching final_test_eval.py."""
    for name in ['best_f1.pth', 'best_auc.pth', 'best_train_loss.pth', 'final.pth']:
        p = os.path.join(seed_dir, name)
        if os.path.exists(p):
            return p
    import glob
    pts = glob.glob(os.path.join(seed_dir, '*.pth'))
    return pts[0] if pts else None


@torch.no_grad()
def run_single_checkpoint(ckpt_path, vf, af, tf, itf, device):
    """vf/af/tf/itf: (1, T, D) tensors already on CPU. Returns (1, NUM_LABELS) logits.
    Slow path (no preload): builds the model fresh from ckpt_path every call.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt['model']
    use_la, use_lc, use_aux, d_model, it_dim, ntl_hidden, olu_hidden = _detect_flags(state)

    model = TripleModalASDModel(
        v_dim=1024, a_dim=1280, t_dim=1024,
        d_model=d_model, nhead=4, num_enc_layers=2,
        proj_dropout=0.07, enc_dropout=0.15, drop_path_rate=0.15,
        use_label_attn=use_la, use_label_corr=use_lc,
        use_aux_heads=use_aux, timing_dim=it_dim,
        ntl_hidden=ntl_hidden, olu_hidden=olu_hidden,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    itf_use = itf[:, :, :it_dim] if itf.shape[-1] > it_dim else itf
    out = model(vf.to(device), af.to(device), tf.to(device), itf_use.to(device))
    logits = out[0] if isinstance(out, tuple) else out

    T = ckpt.get('temperature', None)
    if T is not None:
        Tt = torch.tensor(T, dtype=torch.float32, device=device)
        logits = logits / Tt.unsqueeze(0)

    result = logits.cpu().numpy()
    _unload(model)
    return result


@torch.no_grad()
def run_preloaded_checkpoint(entry, vf, af, tf, itf, device):
    """Fast path: entry is one of the dicts cached by preload_models()
    ({'model', 'temperature', 'it_dim'}). Moves the already-built model to
    GPU, runs it, moves it back to CPU (no re-parsing of the .pth file).
    """
    model = entry['model'].to(device)
    it_dim = entry['it_dim']

    itf_use = itf[:, :, :it_dim] if itf.shape[-1] > it_dim else itf
    out = model(vf.to(device), af.to(device), tf.to(device), itf_use.to(device))
    logits = out[0] if isinstance(out, tuple) else out

    T = entry['temperature']
    if T is not None:
        Tt = torch.tensor(T, dtype=torch.float32, device=device)
        logits = logits / Tt.unsqueeze(0)

    result = logits.cpu().numpy()
    model.cpu()
    torch.cuda.empty_cache()
    return result


def run_model_family(ckpt_dir, seeds, vf, af, tf, itf, device, cache_key=None):
    """Average logits across all seeds in a checkpoint family (v18 or v19).
    If cache_key ('v19_models' / 'v18_models') is given and populated by
    preload_models(), reuses cached in-RAM models instead of re-reading
    .pth files from disk on every request.
    """
    cached = _CACHE.get(cache_key) if cache_key else None
    if cached is not None:
        all_logits = [run_preloaded_checkpoint(entry, vf, af, tf, itf, device)
                     for entry in cached]
        return np.stack(all_logits).mean(0)

    all_logits = []
    for seed in seeds:
        seed_dir = os.path.join(ckpt_dir, f'seed_{seed}')
        ckpt_path = find_checkpoint(seed_dir)
        if ckpt_path is None:
            print(f'  [WARN] No checkpoint found for seed_{seed} in {ckpt_dir} — skipping')
            continue
        logits = run_single_checkpoint(ckpt_path, vf, af, tf, itf, device)
        all_logits.append(logits)
    if not all_logits:
        raise RuntimeError(f'No checkpoints found in {ckpt_dir} for seeds {seeds}')
    return np.stack(all_logits).mean(0)   # (1, NUM_LABELS)


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(video_path, device=None):
    """
    Full pipeline: raw video file -> per-label probabilities + binary predictions.

    Returns:
        {
          'transcript': str,
          'probabilities': {label: float, ...},
          'predictions':   {label: bool, ...},
          'thresholds':    {label: float, ...},
        }
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'[1/4] Extracting features from {video_path} ...')
    feats = extract_features_for_clip(video_path, device)

    vf = feats['video_feats_c0']   # use center crop for inference (no TTA in this version)
    af = feats['audio_feats']
    tf = feats['text_feats']
    itf = feats['timing_feats']

    print('[2/4] Running v19 ensemble (3 seeds, d_model=192) ...')
    v19_logits = run_model_family(V19_CKPT_DIR, SEEDS, vf, af, tf, itf, device, cache_key='v19_models')

    print('[3/4] Running v18 ensemble (3 seeds, d_model=256) ...')
    v18_logits = run_model_family(V18_CKPT_DIR, SEEDS, vf, af, tf, itf, device, cache_key='v18_models')

    print('[4/4] Cross-ensembling + applying fixed thresholds ...')
    final_logits = W_V19 * v19_logits + (1 - W_V19) * v18_logits
    probs = 1.0 / (1.0 + np.exp(-final_logits))   # (1, NUM_LABELS)
    probs = probs[0]

    predictions = (probs >= THRESH_VEC).astype(bool)

    return {
        'transcript':    feats['transcript'],
        'probabilities': {LABELS[i]: float(probs[i]) for i in range(NUM_LABELS)},
        'predictions':   {LABELS[i]: bool(predictions[i]) for i in range(NUM_LABELS)},
        'thresholds':    dict(FIXED_THRESHOLDS),
    }
