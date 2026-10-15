# data/splits

The exact train/test splits behind `results/` and the figures in `docs/brief.md`, as two plain tables. They need no
code from this repository. `scripts/export_splits.py` wrote them from `data/metadata.csv`,
`data/windows/manifest.csv` and `config.yaml`. Retraining heads on these draws reproduces the saved test AUCs exactly.

## Files

| File | One row per | Columns |
| --- | --- | --- |
| `partition.csv` | clip | `study`, `filename`, `class`, `source`, `split` |
| `sets.csv.gz` | clip in a test set or a training draw | `study`, `target`, `set`, `n`, `seed`, `filename`, `label` |

- `study`: `main` uses the clips in `data/metadata.csv`, at `data/clips/{class}/{filename}`. `windows` uses the
  window-length experiment's clips (`data/windows/manifest.csv`), at `data/windows/{3s,5s,10s}/{class}/{filename}`.
  The same filename is used at all three lengths.
- `source`: the source recording. The split is by recording: no recording has clips in both train and test.
- `split`: `train`, `test`, or `unused`. An unused clip comes from a test recording but wasn't sampled into its
  class's test set, so it's never used.
- `target`: the class a binary detector is trained for. `label`: 1 = the target class, 0 = a negative (another class).
- `set = test`: the target's test set. It stays the same for every model, N and seed, so `n` and `seed` are empty.
- `set = train`: one training draw, `n` target clips plus `n` negatives, for `seed` 1–10. Rows are in the order
  drawn (shuffled).

## Using it

```python
import pandas as pd

sets = pd.read_csv("data/splits/sets.csv.gz")
main = sets[sets.study == "main"]
train = main[(main.target == "gun") & (main.set == "train") & (main.n == 10) & (main.seed == 3)]
test = main[(main.target == "gun") & (main.set == "test")]
```

| Experiment | Training | Test |
| --- | --- | --- |
| Saturation (Figures 1, 2, S1), linear probe (S2) | `main` draws, N = 5–160 as far as the class allows, seeds 1–10 | the target's test set |
| Multiclass (Figures 4–6) | for seed s: the `label == 1` rows of each class's N = 40, seed s draw; classes with ≥ 40 `train` clips in `partition.csv` | those classes' test clips (`set == test`, `label == 1`) |
| Cross-detection (Figure 3) | per class: N = 40, or its largest N if it can't reach 40 | every target class's test clips (`label == 1`) |
| Cross-detection, all data (Figure S3) | per class: its largest `n` (its whole training pool) | as above |
| Window length (Figure 7, Table S3) | `windows` draws, N = 5, 10, 20 | the target's test set |

## Background

- How the split was made: whole recordings were held out once (seed 1) to give each class 50 test clips. Each
  class keeps at least 40 training clips. Classes with fewer than 90 clips hold out about half. Negatives come
  equally from every other class. See the main README, "Test and training split".
- Heads: an MLP (embedding → 256 → 64 → 2, ReLU, dropout 0.3), trained with Adam (lr 1e-3) for 20 epochs with
  batch size 32, on frozen embeddings. Their torch seeds don't carry over to other code. Heads you train yourself
  on these draws will differ slightly in AUC; the clips are identical.
- Updating: after adding a training size to `config.yaml`, run `pixi run python -m scripts.export_splits` again.
  It adds that size's draws, and existing rows stay the same. New clips in `metadata.csv` or different split
  settings change the whole split. `pixi run python -m scripts.export_splits --check` reports whether these files
  still match.
