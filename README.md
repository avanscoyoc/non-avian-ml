# Artificial and Abiotic Noise Classification with Frozen Audio Embeddings

How well do frozen pretrained audio models transfer to artificial noise (engines, traffic, generators, power tools,
device static, …) and abiotic sound (wind, thunder, water) in passive acoustic recordings? For 14 noise classes and
6 pretrained models, a small classifier head is trained on each model's embeddings. The findings, figures and
methods are in [docs/brief.md](docs/brief.md) ([PDF](docs/brief.pdf)); this README covers the code and how to
reproduce them.

## Repository layout

```
config.yaml     settings shared by every experiment
core/           shared machinery, runs nothing itself: data split and training draws, encoders, embedding
                cache, classifier heads, plotting
experiments/    one module per experiment in the brief; each saves results/<name>/ and draws its figures
scripts/        data and setup utilities
data/           metadata.csv, window manifests and splits/ (tracked); audio clips from Zenodo (not in git)
cache/          model files and embeddings (not in git; rebuilt on demand, safe to delete)
results/        every number behind the brief
figs/           every figure in the brief
docs/           brief.md (+ PDF) and slides
old/            superseded material, kept locally (not in git)
```

## Experiments

| Experiment | In the brief | Run | Figures (`figs/`) |
|---|---|---|---|
| Training size (saturation) and zero-shot | Figures 1, 2, S1 | `pixi run saturation` | `saturation_main`, `saturation_summary`, `saturation_low_sample` |
| Linear probe vs MLP head | Figure S2 | `pixi run linear-probe` | `head_comparison` |
| Binary cross-detection | Figures 3, S3, cutoff note | `pixi run cross-detection` | `cross_detection`, `cross_detection_all_data` |
| One multiclass head | Figures 4–6 | `pixi run multiclass` | `confusion_matrices`, `tsne_embeddings`, `binary_vs_multiclass_auc` |
| Input window length (BirdNET 3.0) | Figure 7, Table S3 | `pixi run window-length` | `window_size` |

Each experiment writes to `results/<experiment>/` (outputs are listed at the top of its module), skips work whose
results are already saved (delete the file it names to redo it) and then draws its figures; `--plot` only redraws,
e.g. `pixi run python -m experiments.multiclass --plot`. The linear probe and multiclass experiments compare
against the saturation results, so run saturation first.

```bash
pixi run experiments    # all five, in order
pixi run figures        # redraw every figure from results/
pixi run brief          # build docs/brief.pdf
```

Shared design (`core/`):

- **One split by source recording**, used by every experiment (`core/data.py`, `core/study.py`). Clips cut from the
  same recording sound alike, so every recording is either a test or a training recording, never both. Each class
  has the same test clips in every experiment, and the same training draw for a given N and seed. The exact
  training and test clips of every experiment are in [`data/splits/`](data/splits/README.md), checked against
  the code before each run (`pixi run check-splits`).
- **Frozen encoders** (`core/encoders.py`). Each clip is embedded once per model and cached (`core/embeddings.py`).
- **Heads** (`core/heads.py`). MLP Linear → 256 → 64 → 2, dropout 0.3, Adam, 20 epochs, batch 32. Each training
  size N is repeated with 10 random training draws; results are means with 95% CIs across draws.
- **Zero-shot (N = 0)**: a model's own output label for the class, where it has one
  (`zero_shot_labels` in `config.yaml`; review with `pixi run zero-shot-labels`, which uses training clips only).

## Classes

The 14 target classes are listed in `config.yaml` (`classes`, grouped by `class_types`):

- **Artificial noise:** device static, generator, traffic, engine, power tools, human vocal, metal clanging,
  airplane, gun, fireworks, human non-vocal
- **Abiotic:** wind, thunder, water

The other classes in `data/metadata.csv` (frogs, coyote, field cricket, nutria, dog, branches) are used only as
negatives, so the negatives resemble what a noise detector hears in the field.

## Models

| Model | Embedding | Input | Runs on | Zero-shot labels used |
|-------|-----------|-------|---------|-----------------------|
| BirdNET 2.4 (TFLite) | 1024-D (`GLOBAL_AVG_POOL`) | 3 s @ 48 kHz | CPU | Engine, Power tools, Human vocal, Human non-vocal, Gun, Fireworks |
| BirdNET 3.0 (preview3.1) | 1280-D | 3 s @ 32 kHz | GPU if available | Human vocal |
| Perch v8 | 1280-D | 5 s @ 32 kHz (3 s clip centre-padded) | CPU | none (bird labels only) |
| Perch 2.0 | 1536-D | 5 s @ 32 kHz (3 s clip centre-padded) | CPU | FSD50K sound classes (engine, traffic, wind, thunder, water, …) |
| PANNs CNN14 (AudioSet) | 2048-D | 3 s @ 32 kHz | GPU if available | AudioSet classes (static, engine, traffic, clang, …) |
| ResNet-18 (ImageNet) | 512-D | 64-mel spectrogram, 3 s @ 16 kHz | GPU if available | none |

Classifier heads run on the GPU when one is available. TensorFlow/TFLite models always run on the CPU, so the GPU
environment needs CUDA PyTorch only.

## Installation

```bash
pixi install                # laptop / devcontainer (CPU)
pixi install -e gpu         # Linux x86_64 server with an NVIDIA GPU (CUDA 12)
pixi run download-models    # optional: all six models -> cache/models/ (~1.5 GB); otherwise each downloads on first use
```

## Data

Raw clips: Zenodo, DOI [10.5281/zenodo.20534256](https://doi.org/10.5281/zenodo.20534256) (the deposit should be
updated to the layout below).

```
data/
├── metadata.csv                      # one row per clip: common_name, source_audio_path, start/end times, filename
├── clips/{class}/*.wav               # 3 s clips, filename as in metadata.csv
└── windows/                          # window-length experiment, see data/windows/README.md
    ├── manifest.csv                  # kept clips: window bounds and source recording
    ├── feasibility_all_clips.csv     # every metadata.csv row and whether it could be cut at 10 s
    └── {3s,5s,10s}/{class}/*.wav     # the same events cut at three lengths (scripts/make_window_clips.py)
```

Only these clips are needed: negatives are drawn in code from the other classes' clips (equal quota per class),
and Perch pads clips to 5 s in code. Check the folder against the metadata (this also prints the table below):

```bash
pixi run check-data
```

### Test and training split (shared by every experiment)

- Each class's **test set** is 50 clips from test recordings, and each binary test adds the same number of
  negatives drawn equally from the other classes' test clips. The same test clips are used for every model,
  training size, random draw and experiment (binary, multiclass, zero-shot).
- **Training clips** come only from training recordings. A recording goes to test only if every class in it keeps
  at least 40 training clips.
- **Low-sample classes (\*)** have fewer than 90 clips (50 test + 40 training). They use the same rule with a smaller
  target: about half of their clips (whole recordings) are held out for test and the rest are kept for training.
  Their test sets are small (3–30 clips from 1–13 recordings), so their AUCs are provisional. Everything follows
  from `data/metadata.csv`: when more clips are added, these classes move to the full design (50 test clips,
  larger N) with no code changes.
- About 20% of the negative-only classes' clips are held out as test negatives.

| Class | Type | Recordings / clips | Test recordings / clips | Training recordings / clips | Max N |
|---|---|---|---|---|---|
| Device static | artificial | 168 / 1416 | 9 / 50 | 159 / 1366 | 160 |
| Generator | artificial | 32 / 437 | 10 / 50 | 22 / 384 | 160 |
| Traffic | artificial | 53 / 297 | 13 / 50 | 40 / 242 | 160 |
| Engine | artificial | 115 / 170 | 38 / 50 | 76 / 117 | 80 |
| Power tools | artificial | 11 / 162 | 9 / 29 | 2 / 133 | 80 |
| Human vocal | artificial | 16 / 117 | 14 / 50 | 2 / 64 | 40 |
| Metal clanging * | artificial | 28 / 60 | 13 / 30 | 15 / 30 | 20 |
| Airplane * | artificial | 12 / 52 | 6 / 25 | 6 / 27 | 20 |
| Gun * | artificial | 7 / 19 | 5 / 9 | 2 / 10 | 10 |
| Fireworks * | artificial | 4 / 13 | 2 / 6 | 2 / 7 | 5 |
| Human non vocal * | artificial | 2 / 8 | 1 / 3 | 1 / 5 | 5 |
| Wind | abiotic | 106 / 508 | 11 / 50 | 94 / 453 | 160 |
| Thunder * | abiotic | 5 / 21 | 2 / 8 | 3 / 13 | 10 |
| Water * | abiotic | 3 / 11 | 2 / 5 | 1 / 6 | 5 |

\* low sample. 163 of 788 recordings are test; the other 625 are training (all classes, including negative-only
ones). Some classes have few training recordings (power tools: 2, human vocal: 2), so their training clips are
less independent than the clip counts suggest.

## Running on the cirrus GPU server

```bash
nvidia-smi && uname -m                     # GPU visible, x86_64
git clone https://github.com/avanscoyoc/non-avian-ml.git && cd non-avian-ml
curl -fsSL https://pixi.sh/install.sh | bash && source ~/.bashrc
pixi install -e gpu
pixi run -e gpu python -c "import torch; print(torch.cuda.is_available())"   # True
pixi run -e gpu download-models
# copy the audio to the server (data/clips/, data/windows/{3s,5s,10s}/), e.g. zip on the laptop, then unzip
pixi run -e gpu check-data
# TensorFlow models use the CPU and PyTorch models the GPU, so run the saturation experiment as two parallel jobs
nohup pixi run -e gpu python -m experiments.saturation --models birdnet perch perch2 > run_cpu.log 2>&1 &
nohup pixi run -e gpu python -m experiments.saturation --models birdnet3 cnn14 resnet > run_gpu.log 2>&1 &
tail -f run_cpu.log run_gpu.log
pixi run -e gpu experiments                # the rest (completed saturation runs are skipped)
```

Whichever saturation job finishes last writes the complete `results/saturation/auc_by_n.csv`. Completed
class × model runs are skipped on restart, and cached embeddings are reused. On a CPU-only machine run a single
job: parallel jobs there just compete for the same cores. `--classes` runs a subset of classes on the same split.

## Citation

```bibtex
@software{non_avian_ml_2025,
  title = {Non-Avian ML Audio Classification Framework},
  author = {Van Scoyoc, Amy},
  year = {2025},
  url = {https://github.com/avanscoyoc/non-avian-ml}
}
```
