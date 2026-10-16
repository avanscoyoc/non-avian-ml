"""Flags clips whose embeddings sit apart from their class, for listening review (mislabels, a second sound on top,
odd recordings). Writes results/clip_review.csv: one row per clip of the main set, most confused first.

For each model, a clip is compared only with clips from OTHER source recordings (clips cut from the same recording
are near-copies and would vouch for each other):
  disagree   how few of its k nearest clips are its own class: 1 - same-class neighbours / the most it could have
             (k, or fewer if its class has fewer clips in other recordings). 0 = all its own class, 1 = none.
             Empty if no other recording of its class exists (branches).
  neighbour  the class most of those k neighbours belong to (its own class wins ties)
  atypical   within-class percentile of its cosine distance to the centroid of its class's other recordings
             (1 = farthest clip in its class), whether or not another class is close

Summary columns: `disagree` and `atypical` averaged over models; `models_confused` = how many models' neighbour is
another class (k-nearest-neighbours would misclassify the clip); `confused_with` = the other class that turns up most
among its neighbours over all models. The per-model columns follow ({model}_disagree, _neighbour, _atypical).

Every clip is included (train, test and unused; target and negative-only classes): filter on `split` or `class`.
Embeddings come from the cache (and fill it if needed).

Usage:
    pixi run python -m scripts.clip_review
    pixi run python -m scripts.clip_review --models perch2 cnn14 --k 15
"""
import argparse

import numpy as np
import pandas as pd

from core.binary import save_csv
from core.config import load_config
from core.encoders import MODEL_NAMES
from core.paths import CONFIG
from core.study import load_study
from scripts.export_splits import partition_rows


def model_scores(emb: np.ndarray, cls: np.ndarray, rec: np.ndarray, k: int, block: int = 1024):
    """One model's scores per clip against clips from other recordings only. cls, rec: integer codes.

    Returns (disagree, neighbour class votes (clips x classes), cosine distance to own-class centroid).
    """
    n, n_cls = len(emb), cls.max() + 1
    pair = np.unique(cls * (rec.max() + 1) + rec, return_inverse=True)[1]  # (class, recording) group
    avail = np.bincount(cls)[cls] - np.bincount(pair)[pair]  # same-class clips in other recordings

    votes = np.zeros((n, n_cls))
    for start in range(0, n, block):  # in blocks: the full similarity matrix's argpartition needs ~0.5 GB
        stop = min(start + block, n)
        sim = emb[start:stop] @ emb.T  # embeddings are L2-normalised: cosine similarity
        sim[rec[start:stop, None] == rec[None, :]] = -np.inf  # same recording, including the clip itself
        nn = np.argpartition(-sim, k, axis=1)[:, :k]
        np.add.at(votes, (np.arange(start, stop)[:, None], cls[nn]), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        disagree = np.where(avail > 0, 1 - votes[np.arange(n), cls] / np.minimum(k, avail), np.nan)

        class_sum, pair_sum = np.zeros((n_cls, emb.shape[1])), np.zeros((pair.max() + 1, emb.shape[1]))
        np.add.at(class_sum, cls, emb)
        np.add.at(pair_sum, pair, emb)
        centroid = class_sum[cls] - pair_sum[pair]  # own class without own recording
        dist = np.where(avail > 0, 1 - (emb * centroid).sum(1) / np.linalg.norm(centroid, axis=1), np.nan)
    return disagree, votes, dist


def review(clips: pd.DataFrame, embs: dict[str, np.ndarray], k: int) -> pd.DataFrame:
    """Scores for each clip (rows: filename, class, source, split) from each model's embeddings (same row order)."""
    clips = clips.reset_index(drop=True)
    cls, names = pd.factorize(clips["class"])
    names = np.asarray(names)
    rec = pd.factorize(clips["source"])[0]
    rows = np.arange(len(clips))
    out = clips.assign(class_clips=clips.groupby("class")["filename"].transform("size"),
                       class_recordings=clips.groupby("class")["source"].transform("nunique"))

    per_model, pooled = {}, np.zeros((len(clips), len(names)))
    for model, emb in embs.items():
        disagree, votes, dist = model_scores(emb, cls, rec, k)
        tie_break = votes.copy()
        tie_break[rows, cls] += 0.5
        per_model[f"{model}_disagree"] = disagree
        per_model[f"{model}_neighbour"] = names[tie_break.argmax(1)]
        per_model[f"{model}_atypical"] = pd.Series(dist).groupby(cls).rank(pct=True).to_numpy()
        votes[rows, cls] = 0
        pooled += votes
    per_model = pd.DataFrame(per_model)

    neighbours = per_model[[f"{m}_neighbour" for m in embs]].to_numpy()
    out["disagree"] = per_model[[f"{m}_disagree" for m in embs]].mean(axis=1)
    out["models_confused"] = (neighbours != clips["class"].to_numpy()[:, None]).sum(1)
    out["confused_with"] = np.where(pooled.max(1) > 0, names[pooled.argmax(1)], "")
    out["atypical"] = per_model[[f"{m}_atypical" for m in embs]].mean(axis=1)
    out = pd.concat([out, per_model], axis=1)
    return out.sort_values(["disagree", "atypical"], ascending=False, na_position="last").round(3)


def main():
    parser = argparse.ArgumentParser(description="Score every clip by how far its embedding sits from its class.")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, help="Default: all models in the config.")
    parser.add_argument("--k", type=int, default=10, help="Nearest neighbours per clip (default 10).")
    args = parser.parse_args()
    cfg = load_config(args.config)
    study = load_study(cfg)
    clips = pd.DataFrame(partition_rows("main", study)).drop(columns="clip_set")

    embs = {}
    for model in args.models or cfg.models:
        table, _ = study.embed(model, clips.filename)
        embs[model] = table.emb[table.rows(clips.filename)]
    out = review(clips, embs, args.k)

    path = save_csv(out, cfg.results_path / "clip_review.csv")
    print(out.head(20)[["filename", "class", "split", "disagree", "models_confused", "confused_with",
                        "atypical"]].to_string(index=False))
    print(f"Written: {path} ({len(out):,} clips, {len(embs)} models, k = {args.k})")


if __name__ == "__main__":
    main()
