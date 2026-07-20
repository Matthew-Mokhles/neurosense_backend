# NeuroSense Backend

FastAPI backend for the NeuroSense screening platform. It exposes five ML models behind HTTP: sentence-level dysgraphia detection, Q-CHAT-10 toddler autism screening, stereotypical-movement analysis (ST-GCN), facial expression recognition (FERAC), and audio-visual ASD behavior detection (AV-ASD).

All models can run from a **single unified server** (`main.py`) or as **standalone services** (one process per model) when native runtime collisions or memory pressure make a single process unreliable.

Interactive API docs are available at `/docs` on whichever server you start.

---

## Models

| Prefix | Model | Input | Output |
|--------|-------|-------|--------|
| `/graphomotor` | Keras CNN (MobileNetV2-based) | Handwriting image (JPEG/PNG) | Dysgraphia likelihood + confidence |
| `/qchat` | XGBoost + Random Forest ensemble | Q-CHAT-10 answers + demographics | ASD traits risk classification |
| `/asd-motion` | ST-GCN on MediaPipe skeletons | Child activity video(s) or `.npz` skeleton | Per-activity and session-level ASD movement report |
| `/ferac` | ConvNeXt-Small + MediaPipe face pipeline | Webcam/camera frame (JPEG/PNG) | Emotion label (`anger`, `fear`, `joy`, `neutral`) |
| `/avasd` | CLIP + Whisper + E5 + v19/v18 ensemble | ~30 s video clip | Per-label behavior probabilities and binary predictions |

**Clinical note:** These are research/screening tools. They are not a substitute for professional diagnosis.

---

## Quick start

### 1. Prerequisites

- **Python 3.10+** (3.10 recommended; the project is developed and tested on Windows)
- **ffmpeg** on `PATH` (required for AV-ASD video/audio extraction). Set `FFMPEG_PATH` if it is not on `PATH`.
- Enough disk/RAM for model weights (several GB when all models are loaded; AV-ASD also downloads Hugging Face weights on first use)

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements_ferac.txt
```

The unified server additionally needs packages used by individual routers but not listed in those files. Install them if imports fail:

```bash
pip install tensorflow torch torchvision timm transformers openai-whisper python-multipart imageio-ffmpeg
```

### 4. Place model artifacts

Large weights are gitignored. Ensure these paths exist relative to the repo root:

| Path | Used by |
|------|---------|
| `sentence_dysgraphia_model_final.h5` | Graphomotor |
| `xgboost_random_forest_model.pkl` | Q-CHAT |
| `checkpoints/best_model.pth` | ASD motion (ST-GCN) |
| `saved_models_v2/convnext_small_best.pth` | FERAC |
| `checkpoints/v19/seed_{42,123,777}/` | AV-ASD (v19 family) |
| `checkpoints/v18_trainval/seed_{42,123,777}/` | AV-ASD (v18 family) |

MediaPipe face/pose models and Hugging Face weights (CLIP, Whisper, E5) are downloaded automatically on first use.

### 5. Run the unified API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

Or use the self-healing supervisor (recommended on Windows hosts with intermittent native crashes):

```bash
python run_server.py
```

`run_server.py` starts uvicorn as a child process and restarts it after unexpected exits. Use **one worker** when models share a GPU; CUDA contexts do not share well across worker processes.

- Root: `http://localhost:8000/`
- Swagger UI: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

At startup, graphomotor, Q-CHAT, ASD motion, and FERAC are **preloaded**. AV-ASD loads **lazily** on the first `/avasd` request (it is the heaviest stack: CLIP, Whisper, E5, and twelve classifier checkpoints).

---

## Standalone servers

Running every framework (TensorFlow, PyTorch, MediaPipe, scikit-learn/XGBoost) in one process can cause native runtime collisions on some hosts. Each model also has a dedicated entry point:

| App | Port | Command |
|-----|------|---------|
| Graphomotor | 8001 | `uvicorn app_graphomotor:app --host 0.0.0.0 --port 8001` |
| Q-CHAT | 8002 | `uvicorn app_qchat:app --host 0.0.0.0 --port 8002` |
| ASD motion | 8003 | `uvicorn app_asd_motion:app --host 0.0.0.0 --port 8003` |
| FERAC | 8004 | `uvicorn app_ferac:app --host 0.0.0.0 --port 8004` |
| AV-ASD | 8005 | `uvicorn app_avasd:app --host 0.0.0.0 --port 8005` |

Standalone apps preload their own model at startup. Route prefixes are unchanged (`/graphomotor`, `/qchat`, etc.).

---

## API reference

### Meta (unified server only)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service status and model index |
| `GET` | `/health` | Liveness check (200 after startup preload completes) |

### Graphomotor — `/graphomotor`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/graphomotor/` | Health; reports whether the Keras model is loaded |
| `POST` | `/graphomotor/predict` | `multipart/form-data`, field `file` — handwriting image |

Response fields: `label`, `confidence`, `dysgraphia_score`, `low_potential_score`. Uses test-time augmentation (small rotations) before averaging predictions.

### Q-CHAT — `/qchat`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/qchat/questions` | Full Q-CHAT-10 questionnaire, scoring notes, and option lists |
| `POST` | `/qchat/classify` | JSON body with `A1`–`A10` (letters `A`–`E`), `age_months`, `gender`, `jaundice`, `family_asd` |

Scoring follows official Q-CHAT-10 rules: items 1–9 score 1 for C/D/E; item 10 scores 1 for A/B/C. The backend maps letters to binary features before running the ensemble.

Response fields: `classification`, `label`, `confidence`, `risk_level`, `probabilities`.

### ASD motion — `/asd-motion`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/asd-motion/health` | Health check |
| `GET` | `/asd-motion/activities` | List of 11 supported activities |
| `POST` | `/asd-motion/predict/video` | Video + `activity` form field; optional `child_name`, `child_age` |
| `POST` | `/asd-motion/predict/skeleton` | Precomputed `.npz` skeleton + `activity` |
| `POST` | `/asd-motion/predict/session` | Multiple videos + JSON `activities` list for a full session report |

Supported activities: `arm_swing`, `body_swing`, `chest_expansion`, `squat`, `drumming`, `maracas_forward`, `maracas_shaking`, `sing_and_clap`, `frog_pose`, `tree_pose`, `twist_pose`.

Video uploads are written to a temp file, skeletons are extracted with MediaPipe, then the ST-GCN model produces a JSON report.

### FERAC — `/ferac`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/ferac/health` | Health check |
| `POST` | `/ferac/predict` | Single frame, **no** temporal smoothing (good for quick tests) |
| `POST` | `/ferac/session/start` | Start a session; returns `session_id` |
| `POST` | `/ferac/session/{session_id}/predict` | Frame with per-session temporal smoothing (preferred for live apps) |
| `POST` | `/ferac/session/{session_id}/end` | End session and free smoothing state |

Upload field name: `frame` (`multipart/form-data`). Sessions expire after 30 minutes of inactivity.

### AV-ASD — `/avasd`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/avasd/health` | Health; reports whether ensemble weights are loaded and CUDA availability |
| `GET` | `/avasd/labels` | Ten behavior labels and locked decision thresholds |
| `POST` | `/avasd/analyze` | Synchronous analysis (blocks until inference finishes) |
| `POST` | `/avasd/analyze/start` | Upload video; returns `job_id` immediately |
| `GET` | `/avasd/analyze/status/{job_id}` | Poll async job (`pending` / `done` / `error`) |

Accepted video extensions: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm` (max 500 MB). Clips are normalized to ~30 seconds (training assumption). Mobile clients should prefer `/analyze/start` + status polling because inference can take minutes on CPU.

The locked recipe (test F1 ≈ 0.524) cross-ensembles v19 and v18 checkpoints with `w_v19=0.4` and fixed per-label thresholds from `avasd_best_result.json`.

Override checkpoint root with the `AVASD_PROJECT_ROOT` environment variable if weights live outside the repo.

---

## Example requests

**Graphomotor**

```bash
curl -X POST "http://localhost:8000/graphomotor/predict" \
  -F "file=@handwriting.jpg"
```

**Q-CHAT**

```bash
curl -X POST "http://localhost:8000/qchat/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "A1":"A","A2":"B","A3":"C","A4":"C","A5":"C",
    "A6":"C","A7":"B","A8":"C","A9":"C","A10":"D",
    "age_months": 36,
    "gender": "male",
    "jaundice": "no",
    "family_asd": "no"
  }'
```

**ASD motion**

```bash
curl -X POST "http://localhost:8000/asd-motion/predict/video" \
  -F "video=@clip.mp4" \
  -F "activity=arm_swing"
```

**FERAC (session flow)**

```bash
curl -X POST "http://localhost:8000/ferac/session/start"
# → {"session_id":"..."}

curl -X POST "http://localhost:8000/ferac/session/<session_id>/predict" \
  -F "frame=@frame.jpg"
```

**AV-ASD (async)**

```bash
curl -X POST "http://localhost:8000/avasd/analyze/start" -F "file=@clip.mp4"
# → {"job_id":"..."}

curl "http://localhost:8000/avasd/analyze/status/<job_id>"
```

---

## Project layout

```
main.py                  # Unified FastAPI app (all routers)
run_server.py            # Uvicorn supervisor with auto-restart
graphomotorapi.py        # Sentence dysgraphia router
qchat_api.py             # Q-CHAT-10 router
stereotypical_api.py     # ASD motion router
feracapi.py              # FERAC router
avasdapi.py              # AV-ASD router
app_*.py                 # Standalone per-model servers
inference.py             # ST-GCN inference engine + CLI
skeleton_extractor.py    # MediaPipe pose → skeleton
avasd_inference.py       # AV-ASD feature extraction + ensemble
ferac_*.py               # FERAC detection, alignment, inference, smoothing
xgboost_random_forest_model.py  # Q-CHAT training/inference class
requirements.txt         # Core Python dependencies
requirements_ferac.txt   # FERAC / vision extras
```

---

## Operational notes

**Blocking work off the event loop:** TensorFlow, PyTorch, OpenCV, and MediaPipe inference run in Starlette's thread pool so one slow request does not block other endpoints.

**CORS:** All apps allow all origins (`*`) for mobile client development. Tighten this in production.

**GPU vs CPU:** The codebase runs on CPU when no GPU is available. AV-ASD serializes GPU inference with a lock so only one video is processed at a time (designed for ~6 GB VRAM). Use `--workers 1` with CUDA.

**Windows:** `main.py` sets several environment variables (`KMP_DUPLICATE_LIB_OK`, OpenCV/OpenCL/ANGLE overrides, UTF-8 stdout) to reduce DLL and encoding issues when multiple numeric libraries share one process.

**Retraining Q-CHAT:** See `xgboost_random_forest_model.py` and `Readme.txt` for training metrics and feature engineering details. The served model expects the features listed in `/qchat/questions`.

---

## License

Not specified in this repository. Add a license file before public distribution.
