# data/splits

Exactly which clips every experiment in `docs/brief.md` trained and tested on, as plain tables. Use them to
reproduce the results, or to test your own model on the same clips, without any code from this repository.

| File | What it is |
| --- | --- |
| `experiments.csv.gz` | **Start here.** One row per clip per training or test set, for every experiment. |
| `partition.csv` | The split itself: every clip's source recording and whether it is `train`, `test` or `unused`. |
| `cutoff_folds.csv.gz` | Only for the detection-cutoff check in the brief's supplementary note. The figures don't use it. |

Clips live at `data/clips/{class}/{filename}`, except for `window_length`, whose clips are at
`data/windows/{3s,5s,10s}/{class}/{filename}` (the same filename at all three lengths). Both come from Zenodo
(see the main README).

## experiments.csv.gz

| Column | Meaning |
| --- | --- |
| `experiment` | `saturation`, `cross_detection`, `cross_detection_all_data`, `multiclass` or `window_length` |
| `target` | The class a binary detector is trained to find. Empty for `multiclass`, and for the cross-detection test set, which every detector shares. |
| `n` | Training clips per class (N). Empty on test rows. |
| `seed` | Random training draw, 1–10. Empty on test rows. |
| `set` | `train` or `test` |
| `filename`, `class` | The clip and its class |
| `label` | Binary experiments: 1 = the target class, 0 = a negative (another class). Empty for `multiclass` and for the cross-detection test set: use `class` there. |

Each training set is one `(experiment, target, n, seed)` group, or `(experiment, seed)` for `multiclass`. Its rows
are in the order this repository trained on them. Order only matters if you want this code's exact numbers.

| `experiment` | Figures | Training sets | Test set |
| --- | --- | --- | --- |
| `saturation` | 1, 2, S1; also S2 (linear probe) and the zero-shot points (no training) | Per target, N = 5–160 (as far as the class allows) × 10 seeds: N clips of the class + N negatives. 510 sets. | Per target: its test clips + as many negatives |
| `cross_detection` | 3 | Per target, N = 40 (or the largest N a class reaches: 5–20) × 10 seeds. 140 sets. | One shared set: the test clips of all 14 classes (415). Every detector is scored on every class. |
| `cross_detection_all_data` | S3 | Per target, all of its training clips (N = 5–1,366) × 10 seeds. 140 sets. | As `cross_detection` |
| `multiclass` | 4, 5, 6 | Per seed: 40 clips of each of the 7 classes with at least 40 training clips. These are the positives of `saturation`'s N = 40 draw for the same seed. 10 sets. | Those 7 classes' test clips (329) |
| `window_length` | 7, Table S3 | Per target, N = 5, 10, 20 × 10 seeds. 210 sets. | Per target: its test clips + as many negatives |

```python
import pandas as pd

e = pd.read_csv("data/splits/experiments.csv.gz")

# Figure 1: the gun detector, N = 10, seed 3
sat = e[(e.experiment == "saturation") & (e.target == "gun")]
train = sat[(sat.set == "train") & (sat.n == 10) & (sat.seed == 3)]
test = sat[sat.set == "test"]

# Figure 4: the multiclass head, seed 1
mc = e[e.experiment == "multiclass"]
train, test = mc[(mc.set == "train") & (mc.seed == 1)], mc[mc.set == "test"]
```

## partition.csv

Every clip in both clip sets (`clip_set` = `main` or `windows`), with its `class` and `source` recording.
`split` is:

- `train`: from a training recording. Training sets only ever use these.
- `test`: in its class's test set.
- `unused`: from a test recording but not sampled into a test set. These clips are never used, so no recording
  contributes to both training and testing.

Use this file to build training sets of your own (another N, for example) on the same split. The split was made
once (seed 1) by holding out whole recordings: 50 test clips per class in `main` and 25 in `windows`, while each
class keeps at least 40 (`main`) or 20 (`windows`) training clips. Classes too small for that hold out about half
of their clips. The main README, "Test and training split", has the details.

## cutoff_folds.csv.gz

**You only need this file to reproduce one robustness check. No figure uses it.**

Figures 3 and S3 count a clip as detected when the detector's score is at least 0.5, the natural cutoff for a
detector trained on equal numbers of positives and negatives. As a check, each detector also got its own cutoff,
chosen from its training clips alone:

1. Split the training set into up to 5 folds, by recording.
2. Score each fold with a detector trained on the other folds.
3. Choose the cutoff that best balances detections and false alarms (F1) on those scores.

The tuned cutoffs didn't improve that balance, so the figures keep 0.5. The brief's supplementary note reports the
check, and `results/cross_detection/` holds it in its `*_tuned` columns.

This file gives each training clip's `fold` (0–4), for every training set in `cross_detection` and
`cross_detection_all_data`. A training set has no rows here when its positives or its negatives all come from one
recording, since those can't be split by recording; its cutoff stays at 0.5.

## Keeping these in sync

The experiments compute their clips from `data/metadata.csv`, `data/windows/manifest.csv` and `config.yaml` with
the same functions that wrote these files. `pixi run check-splits` confirms that the two still match, and every
`pixi run` experiment task runs it first. After changing the data or the split settings on purpose, run
`pixi run splits` and commit the new files with the new results.

Retraining with this repository's code on these sets reproduces the saved results exactly. Heads trained with other
code will give slightly different AUCs on the same clips. The heads are MLPs (embedding → 256 → 64 → classes,
ReLU, dropout 0.3) trained on frozen embeddings with Adam (lr 1e-3), 20 epochs, batch size 32.
