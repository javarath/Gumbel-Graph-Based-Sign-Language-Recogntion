# Gumbel-Graph-Based-Sign-Language-Recogntion

Graph-based Indian Sign Language recognition using Gumbel Attention Pooling over skeletal landmarks.  
Runs on Windows / Linux / macOS — Python 3.10+.

---

## Table of Contents

1. [How it Works](#how-it-works)
2. [System Setup (Fresh Machine)](#system-setup-fresh-machine)
3. [Kaggle Dataset Setup](#kaggle-dataset-setup)
4. [Step-by-Step: Data → Train → Run](#step-by-step-data--train--run)
5. [Running the Streamlit App](#running-the-streamlit-app)
6. [File & Folder Reference](#file--folder-reference)
7. [Config Reference](#config-reference)
8. [Troubleshooting](#troubleshooting)

---

## How it Works

### Pipeline

```
Raw Video (.mp4)
      │
      ▼
┌──────────────────────┐
│  MediaPipe Holistic   │  keypoints/extractor.py
│  75 landmarks/frame   │  33 pose + 21 left hand + 21 right hand
│  shape: (T, 75, 4)    │  each node: [x, y, z, visibility]
└──────────────────────┘
      │
      ▼
┌──────────────────────┐
│    Graph Builder      │  keypoints/graph_builder.py
│  Anatomical edges     │  wrist→finger, elbow→shoulder, etc.
│  Adj: D⁻½ A D⁻½      │  symmetric normalisation (Kipf & Welling)
└──────────────────────┘
      │  (T, N=75, F=4)  +  (75, 75) adjacency
      ▼
┌──────────────────────┐
│    Stacked GCN        │  models/gcn.py
│  3 layers: 64→128→256│  H_out = act( A_norm @ H @ W + b )
│  BatchNorm + Residual │
└──────────────────────┘
      │  (B, T, 75, 256)
      ▼
┌──────────────────────┐
│  Gumbel Attention     │  models/pooling.py  ← CORE COMPONENT
│     Pooling           │
│  score each of 75     │  s_i = W₂·tanh(W₁·hᵢ + b₁) + b₂
│  nodes → hard select  │  α = one_hot(argmax(s + gumbel_noise))
│  straight-through     │  z = Σ αᵢ · hᵢ
└──────────────────────┘
      │  (B, T, 256)
      ▼
┌──────────────────────┐
│  Temporal Encoder     │  models/temporal.py
│  Bidirectional GRU    │  2 layers: 256 + 128 units
│   (B, 512)            │  BiGRU captures anticipatory + follow-through motion
└──────────────────────┘
      │
      ▼
┌──────────────────────┐
│   Embedding Head      │  models/embedding.py
│  Dense → LayerNorm    │
│  → 384-dim L2-norm    │  projects into SBERT semantic space
└──────────────────────┘
      │
      ▼
  Cosine similarity vs precomputed SBERT gloss embeddings
      │
      ▼
  Top-k (gloss, confidence) predictions
```

### Why Gumbel Attention Pooling

Sign language is spatially **sparse** — any given sign activates only 2–5 of the 75 joints.
Mean pooling treats all 75 nodes equally and drowns the discriminative signal in noise.
Gumbel Attention Pooling learns *which* joints matter per sign using:

- **Learned scoring MLP** — assigns relevance scores to each joint
- **Gumbel noise** (training only) — adds controlled stochasticity for exploration
- **Hard one-hot selection** — true sparsity at forward pass (only 1 joint selected)
- **Straight-through estimator** — gradients flow through the soft version so backprop works
- **Temperature annealing** — τ starts at 1.0 (soft, explorative) and decays to 0.1 (hard, discriminative)

Result: the model is interpretable — you can see exactly which joint it relied on.

---

## System Setup (Fresh Machine)

### 1. Install Python 3.10+

Download from https://www.python.org/downloads/  
**Windows:** tick "Add Python to PATH" during install.

Verify:
```bash
python --version   # must show 3.10.x or higher
```

### 2. Clone / copy the project

```bash
# If you have git:
git clone <your-repo-url>
cd isl_recognition

# Or just copy the project folder and cd into it
cd isl_recognition
```

### 3. (Recommended) Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | What it does |
|---|---|
| `tensorflow>=2.13` | Model training and inference |
| `mediapipe>=0.10` | 75-landmark body skeleton extraction |
| `opencv-python>=4.8` | Video reading, webcam, drawing |
| `sentence-transformers>=2.2` | SBERT gloss embeddings (needs internet once) |
| `scipy>=1.11` | Adjacency matrix normalisation |
| `streamlit>=1.28` | Web app UI |
| `tqdm>=4.65` | Progress bars during preprocessing |
| `matplotlib>=3.7` | Training curve plots |
| `pyyaml>=6.0` | Config file parsing |
| `pytest>=7.4` | Test suite |

### 5. Verify installation

```bash
python -m pytest tests/ -v
# Expected: 41 passed
```

---

## Kaggle Dataset Setup

### Step 1 — Get Kaggle API credentials

1. Log in at https://www.kaggle.com
2. Click your profile picture → **Settings** → **API** → **Create New Token**
3. This downloads `kaggle.json`
4. Place it at:
   - **Windows:** `C:\Users\<YourName>\.kaggle\kaggle.json`
   - **Linux/macOS:** `~/.kaggle/kaggle.json`

```bash
pip install kaggle
```

### Step 2 — Find and download the ISL dataset

Go to https://www.kaggle.com and search for:
```
indian sign language dataset
```

Popular ones to look for:

| Dataset name | Type | Notes |
|---|---|---|
| Indian Sign Language (ISL) | Images (jpg) | Alphabet + common words |
| ISL Recognition Dataset | Video clips (mp4) | Word-level signs |
| INCLUDE Dataset | Video clips | 263 ISL signs |

**Download the dataset** (replace `username/dataset-name` with what you find):

```bash
# Creates data/kaggle_raw/ and unzips inside it
kaggle datasets download -d username/dataset-name -p data/kaggle_raw --unzip
```

Check what you got:
```bash
# Windows
dir data\kaggle_raw

# Linux/macOS
ls data/kaggle_raw/
```

### Step 3 — Convert to project format

The project expects:
```
data/raw/
├── hello/
│   ├── 0001.mp4
│   └── 0002.mp4
├── water/
│   └── 0001.mp4
└── ...
```

Run the conversion script — it **auto-detects** whether your dataset has images or videos:

```bash
python scripts/prepare_kaggle_data.py --kaggle_dir data/kaggle_raw
```

**If your dataset has image frames** (most common — folders of .jpg files per sign):
```bash
python scripts/prepare_kaggle_data.py \
    --kaggle_dir data/kaggle_raw \
    --format image \
    --fps 30 \
    --frames_per_sample 60
```
`--frames_per_sample 60` means every 60 images → 1 video clip (= 2 seconds at 30 fps).

**If your dataset already has .mp4 / .avi video clips per sign:**
```bash
python scripts/prepare_kaggle_data.py \
    --kaggle_dir data/kaggle_raw \
    --format video
```

**Verify the output:**
```python
from data import VideoCollector
print(VideoCollector.list_all())
# {'hello': 12, 'water': 10, 'namaste': 8, ...}
```

---

## Step-by-Step: Data → Train → Run

### Step 1 — Preprocess (video → keypoint arrays)

```bash
python scripts/preprocess.py
```

What it does:
- Opens each `.mp4` in `data/raw/{gloss}/`
- Runs MediaPipe Holistic to extract 75 landmarks per frame
- Saves `(60, 75, 4)` numpy arrays to `data/processed/{gloss}/{idx:04d}.npy`

> **First run needs internet** — MediaPipe downloads its model weights (~30 MB).

Re-process everything (if you changed the dataset):
```bash
python scripts/preprocess.py --force
```

Expected output:
```
── Preprocessed Samples ─────────────────────────
  hello                          12 samples
  namaste                        10 samples
  water                           8 samples
  TOTAL                          30 samples
─────────────────────────────────────────────────
```

### Step 2 — Train

```bash
python scripts/train.py
```

With options:
```bash
python scripts/train.py --epochs 50 --batch_size 8 --plot
```

What happens during training:
1. Loads all `data/processed/**/*.npy` files
2. Computes SBERT embeddings for each gloss label — **needs internet once**, then cached to `data/gloss_embeddings.npy`
3. Splits data → train / val / test (70% / 15% / 15%)
4. Trains for up to 100 epochs with:
   - **Cosine decay** learning rate schedule
   - **Gumbel temperature annealing** (τ: 1.0 → 0.1 over epochs)
   - **Early stopping** if `val_top1_accuracy` doesn't improve for 15 epochs
   - **Checkpoint** saves best weights to `checkpoints/best_model.weights.h5`
   - **TensorBoard** logs to `logs/`

Typical console output:
```
Epoch 1/100
  loss: 0.842  top1_accuracy: 0.12  val_top1_accuracy: 0.08  τ: 1.0000 → 0.9500
Epoch 2/100
  loss: 0.731  top1_accuracy: 0.24  val_top1_accuracy: 0.19  τ: 0.9500 → 0.9025
...
Epoch 23/100  ← early stop triggers here if val plateaus
  loss: 0.124  top1_accuracy: 0.91  val_top1_accuracy: 0.87
```

Monitor with TensorBoard (optional):
```bash
tensorboard --logdir logs
# open http://localhost:6006
```

### Step 3 — Confirm training succeeded

```bash
python -c "
from pathlib import Path
p = Path('checkpoints/best_model.weights.h5')
print('Weights exist:', p.exists(), '| Size:', p.stat().st_size // 1024, 'KB')
"
```

---

## Running the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Opens at **http://localhost:8501**

### Tab 1 — Upload Video

1. Click **Browse files** → choose a `.mp4 / .avi / .mov` file
2. The app runs the full pipeline:
   - Extracts keypoints with MediaPipe
   - Builds graph features
   - Runs through GCN → Gumbel Pooling → BiGRU → EmbeddingHead
3. Shows:
   - **Skeleton overlay** on the first frame (green = pose, blue = left hand, red = right hand)
   - **Top-3 predictions** as confidence progress bars
   - **Gumbel attention map** on the middle frame — larger red circles = joints the model focused on

### Tab 2 — Live Webcam

1. Click **Start**
2. The app opens your webcam and shows:
   - Live skeleton overlay on every frame
   - Prediction cards updating every ~30 frames
   - **Gumbel τ** value (shows how sharp/soft the attention currently is)
3. Click **Stop** to end

### Single-file inference (no app)

```python
from inference import ISLPredictor

p = ISLPredictor()
results = p.predict("my_video.mp4")
print(results)
# [('hello', 0.93), ('namaste', 0.71)]
```

### Real-time webcam (terminal / OpenCV window)

```python
from inference import RealtimePredictor

rt = RealtimePredictor()
rt.run()   # press Q to quit
```

---

## File & Folder Reference

```
isl_recognition/
│
├── config/
│   ├── config.yaml          All hyperparameters and file paths — edit this to tune
│   └── loader.py            Loads config.yaml → cfg namespace (singleton)
│
├── keypoints/
│   ├── extractor.py         MediaPipe wrapper. KeypointExtractor + KeypointFrame
│   │                        Output: (75, 4) per frame  [x, y, z, visibility]
│   └── graph_builder.py     Builds (T,75,4) node features + (75,75) adjacency
│                            Defines anatomical edge lists (_POSE_EDGES, _HAND_EDGES)
│
├── models/
│   ├── gcn.py               GCNLayer: H_out = act(A @ H @ W + b)
│   │                        StackedGCN: 3 layers with BatchNorm + Dropout + Residual
│   ├── pooling.py           GumbelAttentionPooling — the core novel component
│   │                        Implements hard sparse joint selection with ST estimator
│   ├── temporal.py          TemporalEncoder: Bidirectional GRU (2 stacked layers)
│   ├── embedding.py         EmbeddingHead: projects to 384-dim L2-normalised space
│   │                        GlossEmbeddingStore: SBERT embedding cache + cosine lookup
│   └── isl_model.py         ISLModel: assembles the full pipeline end-to-end
│
├── training/
│   ├── losses.py            CosineSimilarityLoss  (main), TripletCosineLoss (optional)
│   ├── metrics.py           MeanCosineSimilarity, TopKAccuracy (top-1 and top-5)
│   └── trainer.py           Trainer class + GumbelTemperatureCallback (τ annealing)
│
├── data/
│   ├── collector.py         VideoCollector: record webcam clips or copy video files
│   ├── preprocessor.py      Preprocessor: video → .npy keypoint arrays
│   └── dataset.py           ISLDataset: load .npy files → tf.data pipeline
│
├── inference/
│   ├── predictor.py         ISLPredictor: load weights + predict from a video file
│   └── realtime.py          RealtimePredictor: webcam loop with threaded prediction
│
├── utils/
│   ├── logger.py            get_logger(__name__) — structured logging
│   │                        Set log level via env var ISL_LOG_LEVEL=DEBUG
│   └── visualization.py     draw_skeleton()      — coloured edge overlay on frame
│                            draw_attention_map() — circle size = attention weight
│                            plot_training_history() — loss + accuracy curves
│
├── app/
│   └── streamlit_app.py     Web UI: two tabs (upload video + live webcam)
│
├── scripts/
│   ├── prepare_kaggle_data.py  Convert Kaggle ISL dataset → data/raw/{gloss}/*.mp4
│   ├── preprocess.py           Run MediaPipe on all raw videos → data/processed/
│   └── train.py                Full training script with CLI args
│
├── tests/                   41 pytest tests covering config, graph, and model
│
├── data/
│   ├── raw/                 Raw .mp4 clips  (created by prepare_kaggle_data.py)
│   ├── processed/           .npy keypoint arrays  (created by preprocess.py)
│   ├── gloss_embeddings.npy SBERT embeddings cache (created on first train)
│   └── label_map.json       gloss → index mapping  (created on first train)
│
├── checkpoints/
│   └── best_model.weights.h5   Best model weights (created by train.py)
│
├── logs/                    TensorBoard logs + training.csv
├── requirements.txt
└── README.md
```

---

## Config Reference

All settings are in [config/config.yaml](config/config.yaml). You never need to touch Python files to tune the model.

| Section | Key | Default | What to change |
|---|---|---|---|
| `video` | `max_frames` | 60 | Increase for longer signs |
| `video` | `min_frames` | 15 | Clips shorter than this are skipped |
| `video` | `fps` | 30 | Match your camera / dataset FPS |
| `gcn` | `hidden_dims` | [64,128,256] | Deeper = more capacity |
| `gcn` | `dropout_rate` | 0.3 | Increase if overfitting |
| `pooling` | `temperature` | 1.0 | Starting Gumbel τ |
| `pooling` | `anneal_rate` | 0.95 | How fast τ shrinks each epoch |
| `pooling` | `min_temperature` | 0.1 | τ floor (hard attention) |
| `temporal` | `gru_units` | [256,128] | GRU hidden sizes |
| `training` | `epochs` | 100 | Max epochs (early stop active) |
| `training` | `batch_size` | 16 | Reduce to 8 if GPU OOM |
| `training` | `learning_rate` | 1e-3 | Starting LR (cosine decayed) |
| `training` | `early_stopping_patience` | 15 | Epochs to wait before stopping |
| `inference` | `similarity_threshold` | 0.6 | Raise to filter weak predictions |
| `inference` | `top_k` | 5 | Number of candidates returned |
| `app` | `webcam_index` | 0 | Change to 1/2 for external camera |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python
```

### `ModuleNotFoundError: No module named 'mediapipe'`
```bash
pip install mediapipe
```

### `FileNotFoundError: best_model.weights.h5`
You haven't trained yet. Run:
```bash
python scripts/train.py
```

### `RuntimeError: Call build() first` (GlossEmbeddingStore)
The SBERT cache doesn't exist yet — training creates it. Needs internet on first run.

### `No preprocessed .npy files found`
Run preprocessing before training:
```bash
python scripts/preprocess.py
```

### `No video data found in data/raw`
Run the Kaggle data preparation script first:
```bash
python scripts/prepare_kaggle_data.py --kaggle_dir data/kaggle_raw
```

### Streamlit: `OSError: [Errno 98] Address already in use`
```bash
streamlit run app/streamlit_app.py --server.port 8502
```

### Low accuracy
- More data per gloss (aim for 10+ clips each)
- Increase `epochs` in config.yaml
- Lower `similarity_threshold` to see more candidates
- Check `logs/training.csv` to see if training converged

### `ISL_LOG_LEVEL` — verbose debug output
```bash
# Windows
set ISL_LOG_LEVEL=DEBUG
python scripts/train.py

# Linux/macOS
ISL_LOG_LEVEL=DEBUG python scripts/train.py
```

---

## Quick Reference Card

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download Kaggle dataset
kaggle datasets download -d username/dataset-name -p data/kaggle_raw --unzip

# 3. Convert to project format
python scripts/prepare_kaggle_data.py --kaggle_dir data/kaggle_raw

# 4. Extract keypoints
python scripts/preprocess.py

# 5. Train
python scripts/train.py --plot

# 6. Run web app
streamlit run app/streamlit_app.py

# 7. Run tests
python -m pytest tests/ -v
```

---

## Acknowledgements

- **MediaPipe** (Google) — real-time 75-landmark holistic body tracking
- **Sentence-Transformers** — `all-MiniLM-L6-v2` for semantic gloss embeddings
- **Jang et al., 2017** — "Categorical Reparameterization with Gumbel-Softmax", ICLR 2017
- **Kipf & Welling, 2017** — "Semi-Supervised Classification with Graph Convolutional Networks"
