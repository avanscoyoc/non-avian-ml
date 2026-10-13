"""Per-model embedding cache: each clip is encoded once per model, however many classes use it.

Layout: {embeddings_path}/{model}/names.npy, emb.npy, scores.npy (model's own label scores, if any),
labels.json, info.json (which model file produced them). Only clips missing from the cache are encoded,
so the encoder is loaded only when needed; if the model file changed, the cache is rebuilt.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data_loader import clip_path
from src.model_loader import MODEL_SOURCES, load_model


@dataclass
class EmbeddingTable:
    names: np.ndarray
    emb: np.ndarray
    scores: np.ndarray | None
    labels: list[str] | None

    def __post_init__(self):
        self.index = {n: i for i, n in enumerate(self.names)}

    def rows(self, filenames) -> list[int]:
        return [self.index[f] for f in filenames]


def _load_cache(cache_dir: Path) -> EmbeddingTable | None:
    if not (cache_dir / "emb.npy").exists():
        return None
    scores_file = cache_dir / "scores.npy"
    labels_file = cache_dir / "labels.json"
    return EmbeddingTable(
        names=np.load(cache_dir / "names.npy"),
        emb=np.load(cache_dir / "emb.npy"),
        scores=np.load(scores_file, mmap_mode="r") if scores_file.exists() else None,
        labels=json.loads(labels_file.read_text()) if labels_file.exists() else None,
    )


def _save_cache(cache_dir: Path, table: EmbeddingTable) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"names": table.names, "emb": table.emb}
    if table.scores is not None:
        arrays["scores"] = np.asarray(table.scores)
    for key, arr in arrays.items():  # write-then-rename so an interrupted run never leaves a torn cache
        tmp = cache_dir / f"{key}.tmp.npy"
        np.save(tmp, arr)
        tmp.replace(cache_dir / f"{key}.npy")
    if table.labels is not None:
        (cache_dir / "labels.json").write_text(json.dumps(table.labels))


def _check_source(cache_dir: Path, source: str) -> bool:
    """True if the cache was built with the current model file (an unstamped cache is stamped as current)."""
    info_file = cache_dir / "info.json"
    if not info_file.exists():
        if (cache_dir / "emb.npy").exists():
            info_file.write_text(json.dumps({"source": source}))
        return True
    return json.loads(info_file.read_text()).get("source") == source


def get_embeddings(model_name: str, clips: list[tuple[str, str]], data_path, embeddings_path,
                   clip_seconds: float | None = None) -> EmbeddingTable:
    """Embeddings (and label scores) for (class, filename) clips, encoding only the uncached ones.

    clip_seconds: input window length if not the default 3 s (recorded in the cache stamp).
    """
    cache_dir = Path(embeddings_path) / model_name
    source = MODEL_SOURCES[model_name] + (f"; {clip_seconds:g} s input" if clip_seconds else "")
    if not _check_source(cache_dir, source):
        print(f"[{model_name}] cached embeddings come from a different model file; re-encoding all clips")
        for f in cache_dir.glob("*"):
            f.unlink()
    table = _load_cache(cache_dir)
    cached = table.index if table is not None else {}
    missing = sorted({c for c in clips if c[1] not in cached}, key=lambda c: c[1])
    if not missing:
        print(f"[{model_name}] all {len(set(clips))} clips cached")
        return table

    print(f"[{model_name}] encoding {len(missing)} clips ({len(cached)} cached)")
    encoder, _ = load_model(model_name, clip_seconds)
    emb, scores = encoder.encode([clip_path(data_path, cls, fn) for cls, fn in missing])
    labels = encoder.labels
    encoder.cleanup()

    names = np.array([fn for _, fn in missing])
    if table is not None:
        names = np.concatenate([table.names, names])
        emb = np.concatenate([table.emb, emb])
        if scores is not None:
            scores = np.concatenate([np.asarray(table.scores), scores])
    _save_cache(cache_dir, EmbeddingTable(names, emb, scores, labels))
    (cache_dir / "info.json").write_text(json.dumps({"source": source}))
    return _load_cache(cache_dir)
