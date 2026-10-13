"""
saturation_summary.py

Summarises the training-size (saturation) results per class x model, separating classes a model already
knows (it has a built-in output label for the class, see src/zero_shot_labels.yaml) from new ones.

Writes results/saturation_summary.csv and .md with, per class x model:
  label_status      known (built-in label -> zero-shot point) or new
  zero_shot_auc     AUC of the built-in label (known only)
  auc_at_40, ci_40  test AUC at the headline training size N = 40 (if the class has enough training clips)
  best_auc, best_n  best mean test AUC over training sizes and where it occurs
  plateau_n         smallest N whose mean is within 0.02 of best_auc
  break_even_n      smallest N whose 95% CI lies entirely above the zero-shot AUC (known only)
  outcome           known only: adaptation beats label / tie at ceiling (both >= 0.97) / label better
  low_sample, train_recordings, train_clips
and prints the headline numbers used in the text.

Usage:
    pixi run python scripts/saturation_summary.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import ROOT  # noqa: E402
from src.config import build_partition, load_config  # noqa: E402
from src.data_loader import load_manifest  # noqa: E402
from src.plot_style import MODELS, class_name, model_name  # noqa: E402
from src.zero_shot import load_label_map  # noqa: E402

HEADLINE_N = 40
PLATEAU_TOL = 0.02
CEILING = 0.97


def summarise(cfg) -> pd.DataFrame:
    results = pd.concat(pd.read_csv(cfg.results_path / f"results_{c}.csv") for c in cfg.species
                        if (cfg.results_path / f"results_{c}.csv").exists())
    labels = load_label_map()
    manifest = load_manifest(cfg.clips_csv)
    partition = build_partition(manifest, cfg)

    rows = []
    for c in cfg.species:
        train_recs = len({partition.source[f] for f in partition.train_clips[c]})
        for m in [m for m in MODELS if m in cfg.models]:
            d = results[(results.species == c) & (results.model == m)].sort_values("training_size")
            if d.empty:
                continue
            trained = d[d.training_size > 0]
            known = bool(labels.get(c, {}).get(m))
            zs = d[d.training_size == 0].test_auc_mean
            zs = float(zs.iloc[0]) if known and len(zs) else np.nan
            best = trained.loc[trained.test_auc_mean.idxmax()]
            at40 = trained[trained.training_size == HEADLINE_N]
            above = trained[trained.test_auc_mean - trained.test_auc_ci_95 > zs] if known else trained.iloc[:0]
            if not known:
                outcome = ""
            elif zs >= CEILING and best.test_auc_mean >= CEILING:
                outcome = "tie at ceiling"
            elif len(above):
                outcome = "adaptation beats label"
            elif best.test_auc_mean > zs:
                outcome = "adaptation >= label (CI overlaps)"
            else:
                outcome = "label better"
            rows.append({
                "class": c, "model": m, "label_status": "known" if known else "new", "zero_shot_auc": zs,
                "auc_at_40": float(at40.test_auc_mean.iloc[0]) if len(at40) else np.nan,
                "ci_40": float(at40.test_auc_ci_95.iloc[0]) if len(at40) else np.nan,
                "best_auc": float(best.test_auc_mean), "best_n": int(best.training_size),
                "plateau_n": int(trained[trained.test_auc_mean >= best.test_auc_mean - PLATEAU_TOL].training_size.min()),
                "break_even_n": int(above.training_size.min()) if len(above) else np.nan,
                "outcome": outcome, "low_sample": bool(d.low_sample.iloc[0]),
                "train_recordings": train_recs, "train_clips": len(partition.train_clips[c]),
            })
    return pd.DataFrame(rows)


def main():
    cfg = load_config(ROOT / "src" / "config.yaml")
    s = summarise(cfg)
    out = cfg.results_path / "saturation_summary.csv"
    s.round(4).to_csv(out, index=False)

    md = s.copy()
    md["class"] = md["class"].map(class_name)
    md["model"] = md["model"].map(model_name)
    md = md.round(3).astype(object).where(md.notna(), "")
    lines = ["| " + " | ".join(md.columns) + " |", "|" + "---|" * len(md.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in md.itertuples(index=False)]
    (cfg.results_path / "saturation_summary.md").write_text("\n".join(lines) + "\n")
    print(f"Written: {out} (+ .md)")

    known, full = s[s.label_status == "known"], s[~s.low_sample]
    new_everywhere = [c for c in cfg.species if (s[s["class"] == c].label_status == "new").all()]
    print(f"\nclass x model pairs: {len(s)} | with a built-in label (known): {len(known)} | new: {len(s) - len(known)}")
    print(f"classes no model knows: {new_everywhere}")
    print("known pairs by outcome:", known.outcome.value_counts().to_dict())
    print(f"median break-even N (where adaptation clears the label): {known.break_even_n.median()}")
    print(f"well-sampled classes: median plateau N {full.plateau_n.median()} (within {PLATEAU_TOL} of best)")
    print(f"AUC at N={HEADLINE_N}, well-sampled classes, mean by model:")
    print(full.groupby("model").auc_at_40.mean().reindex([m for m in MODELS if m in cfg.models]).round(3).to_string())


if __name__ == "__main__":
    main()
