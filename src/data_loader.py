import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

CLIP_SECONDS = 3.0


# ── Manifest and data checks ──────────────────────────────────────────────────

def load_manifest(csv_path) -> pd.DataFrame:
    """One row per clip: class folder name, filename and source recording."""
    df = pd.read_csv(csv_path)
    return pd.DataFrame({
        "cls": df["common_name"],
        "filename": df["filename"],
        "source": df["source_audio_path"],
    })


def clip_path(data_path, cls: str, filename: str) -> Path:
    return Path(data_path) / cls / filename


def check_data(manifest: pd.DataFrame, data_path) -> tuple[list[str], list[str]]:
    """Every manifest clip exists, and nothing else is in the class folders.

    Returns (problems, notes). Clips that are not exactly 3.000 s are only a note: the loaders
    centre-crop or zero-pad every clip to 3 s (a few clips end early because the source recording does).
    """
    data_path = Path(data_path)
    problems, notes = [], []
    if manifest.filename.duplicated().any():
        problems.append(f"duplicate filenames in manifest: {manifest.filename[manifest.filename.duplicated()].tolist()[:5]}")

    expected = set(zip(manifest.cls, manifest.filename))
    missing = Counter(cls for cls, fn in expected if not clip_path(data_path, cls, fn).is_file())
    if missing:
        problems.append(f"{sum(missing.values())} clips missing: " + ", ".join(f"{c} {n}" for c, n in sorted(missing.items())))

    extra = Counter(
        d.name for d in data_path.iterdir() if d.is_dir()
        for f in d.glob("*.wav") if (d.name, f.name) not in expected
    )
    if extra:
        problems.append(f"{sum(extra.values())} wav files not in manifest: " + ", ".join(f"{c} {n}" for c, n in sorted(extra.items())))

    off_length = []
    for cls, fn in sorted(expected):
        p = clip_path(data_path, cls, fn)
        if p.is_file():
            info = sf.info(str(p))
            if abs(info.frames - CLIP_SECONDS * info.samplerate) > 1:
                off_length.append(f"{cls}/{fn} {info.duration:.3f}s")
    if off_length:
        notes.append(f"{len(off_length)} clips not exactly {CLIP_SECONDS:.0f} s (padded/cropped to 3 s): {', '.join(off_length)}")
    return problems, notes


# ── Audio ─────────────────────────────────────────────────────────────────────

def load_clip(path, sr: int, seconds: float = CLIP_SECONDS) -> np.ndarray:
    """Mono float32 at `sr`, centre-cropped or end-padded to exactly `seconds`."""
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    n = int(round(sr * seconds))
    if len(y) > n:
        start = (len(y) - n) // 2
        y = y[start:start + n]
    elif len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y.astype(np.float32)


def centre_pad(y: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """Pad with equal silence on both sides (3 s -> 5 s adds 1 s each side, as data_5s did)."""
    extra = int(round(sr * seconds)) - len(y)
    if extra <= 0:
        return y
    return np.pad(y, (extra // 2, extra - extra // 2))


# ── One shared train/test partition by source recording ───────────────────────

@dataclass
class Partition:
    """Every source recording is either test or training, for every class and every experiment."""
    test_recordings: set
    test_clips: dict[str, list[str]]   # per class: the fixed held-out clips (<= test_size for experiment classes)
    train_clips: dict[str, list[str]]  # per class: all clips from training recordings
    source: dict[str, str]             # filename -> source recording
    low_sample: set                    # target classes too small for the full design (about half held out)


def make_partition(manifest: pd.DataFrame, classes: list[str], test_size: int = 50, seed: int = 1,
                   max_test_recording_clips: int = 25, min_train: dict[str, int] | None = None,
                   donor_test_fraction: float = 0.2, low_sample_test_fraction: float = 0.5) -> Partition:
    """Hold out whole recordings once, shared by the binary and multiclass experiments.

    Experiment classes (scarcest first) get recordings moved to the test set, in seeded random order,
    until they have `test_size` test clips: recordings with <= `max_test_recording_clips` clips first,
    larger ones only if still short. A recording is only moved if every experiment class it contains
    keeps at least `min_train[cls]` training clips (recordings can hold several classes).

    Low-sample classes (fewer than `test_size + min_train[cls]` clips) use the same rule with a smaller
    target: about `low_sample_test_fraction` of their clips go to test and the rest are kept for training.
    Everything follows from the manifest, so a class moves to the full design once it has enough clips.
    Donor-only classes with >= 2 recordings get ~`donor_test_fraction` of their clips held out, so test
    negatives stay balanced across donors.
    """
    rng = random.Random(seed)
    min_train = min_train or {}
    size = manifest.groupby("source").size()
    per_rec = {rec: g.cls.value_counts().to_dict() for rec, g in manifest.groupby("source")}
    total = manifest.cls.value_counts().to_dict()
    n_test = {c: 0 for c in total}
    n_train = dict(total)
    test = set()

    target, floor = {}, {}
    for c in classes:
        if total[c] >= test_size + min_train.get(c, 0):
            target[c], floor[c] = test_size, min_train.get(c, 0)
        else:
            target[c] = max(1, int(total[c] * low_sample_test_fraction))
            floor[c] = total[c] - target[c]
    low_sample = {c for c in classes if target[c] < test_size}

    def safe(rec):
        return all(n_train[c] - k >= min(floor[c], n_train[c])
                   for c, k in per_rec[rec].items() if c in classes)

    def hold_out(rec):
        test.add(rec)
        for c, k in per_rec[rec].items():
            n_test[c] += k
            n_train[c] -= k

    for c in sorted(classes, key=lambda c: (total[c], c)):
        recs = sorted(manifest[manifest.cls == c].source.unique())
        rng.shuffle(recs)
        for rec in recs:  # small recordings first
            if n_test[c] >= target[c]:
                break
            if rec not in test and size[rec] <= max_test_recording_clips and safe(rec):
                hold_out(rec)
        for rec in sorted(recs, key=lambda r: (size[r], r)):  # larger ones only if still short
            if n_test[c] >= target[c]:
                break
            if rec not in test and safe(rec):
                hold_out(rec)

    for c in sorted(set(total) - set(classes)):
        recs = sorted(manifest[manifest.cls == c].source.unique())
        if len(recs) < 2:
            continue
        rng.shuffle(recs)
        for rec in recs:
            if n_test[c] >= donor_test_fraction * total[c]:
                break
            if rec not in test and safe(rec):
                hold_out(rec)

    in_test = manifest.source.isin(test)
    test_clips, train_clips = {}, {}
    for c, g in manifest.groupby("cls"):
        held = sorted(g.filename[in_test[g.index]])
        test_clips[c] = sorted(rng.sample(held, min(test_size, len(held)))) if c in classes else held
        train_clips[c] = sorted(g.filename[~in_test[g.index]])
    return Partition(test, test_clips, train_clips, dict(zip(manifest.filename, manifest.source)), low_sample)


@dataclass
class Split:
    """Binary task for one target class, derived from the shared partition."""
    target: str
    train_pos: list[str]
    train_neg: list[str]
    test_pos: list[str]
    test_neg: list[str]
    n_test_recordings: int

    @property
    def max_n(self) -> int:
        return min(len(self.train_pos), len(self.train_neg))


def _quotas(available: dict[str, int], n: int) -> dict[str, int]:
    """Equal quota per donor class; shortfalls go to the largest pools (build_negatives rule)."""
    quota = {d: 0 for d in available}
    remaining = n
    active = [d for d in sorted(available) if available[d] > 0]
    while remaining > 0 and active:
        share, extra = divmod(remaining, len(active))
        for i, d in enumerate(sorted(active, key=lambda d: (-available[d], d))):
            take = min(share + (i < extra), available[d] - quota[d])
            quota[d] += take
            remaining -= take
        active = [d for d in active if quota[d] < available[d]]
    return quota


def _balanced_sample(pools: dict[str, list[str]], n: int, rng: random.Random) -> list[str]:
    quota = _quotas({d: len(v) for d, v in pools.items()}, n)
    return sorted(f for d in sorted(quota) for f in rng.sample(pools[d], quota[d]))


def binary_split(partition: Partition, target: str, seed: int = 1) -> Split:
    """1:1 binary task: the target's held-out clips vs an equal number of other classes' held-out clips
    (equal quota per class); training pools from training recordings only, negatives as many as positives."""
    rng = random.Random(f"{seed}-{target}")
    test_pos = partition.test_clips[target]
    test_neg = _balanced_sample({d: v for d, v in partition.test_clips.items() if d != target}, len(test_pos), rng)
    train_pos = partition.train_clips[target]
    train_neg = _balanced_sample({d: v for d, v in partition.train_clips.items() if d != target}, len(train_pos), rng)

    test_recs = {partition.source[f] for f in test_pos + test_neg}
    assert not test_recs & {partition.source[f] for f in train_pos + train_neg}, f"{target}: recording overlap"
    assert not test_recs - partition.test_recordings, f"{target}: test clip outside the test recordings"
    return Split(target, train_pos, train_neg, test_pos, test_neg, len(test_recs))


def print_split_audit(partition: Partition, splits: dict[str, Split], training_sizes: list[int]) -> None:
    n_recs = len(set(partition.source.values()))
    print(f"Shared partition: {len(partition.test_recordings)} test / {n_recs - len(partition.test_recordings)} "
          f"training recordings (of {n_recs})")
    print(f"{'class':22s} {'test pos/neg':>12s} {'test recs':>9s} {'train pos/neg':>13s} {'max N':>6s}")
    for sp, s in splits.items():
        feasible = [n for n in training_sizes if n <= s.max_n]
        print(f"{sp:22s} {len(s.test_pos):>6d}/{len(s.test_neg):<5d} {s.n_test_recordings:>9d} "
              f"{len(s.train_pos):>6d}/{len(s.train_neg):<6d} {max(feasible, default=0):>6d}")


# ── Per-seed training draws (unchanged logic) ──────────────────────────────────

def load_audio_files(training_size, train_pool_pos, train_pool_neg, random_seed=42):
    """Sample balanced training data from train pool."""
    if len(train_pool_pos) < training_size:
        raise ValueError(
            f"Train pool has {len(train_pool_pos)} pos files, need {training_size}"
        )
    if len(train_pool_neg) < training_size:
        raise ValueError(
            f"Train pool has {len(train_pool_neg)} neg files, need {training_size}"
        )

    rng = random.Random(random_seed)
    pos_sample = rng.sample(train_pool_pos, training_size)
    neg_sample = rng.sample(train_pool_neg, training_size)

    files = pos_sample + neg_sample
    labels = [1] * training_size + [0] * training_size

    combined = list(zip(files, labels))
    rng.shuffle(combined)
    files, labels = zip(*combined)

    return list(files), list(labels)
