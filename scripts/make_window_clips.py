#!/usr/bin/env python3
"""
Build a 3s / 5s / 10s window subset for BirdNET+ V3.0.

All three windows share ONE center: the midpoint of the annotated event
(start_time, end_time) in the full recording. A clip is kept only if the
10s window fits inside the recording. Because the windows are nested,
that guarantees the 3s and 5s windows fit too, so every clip in the
subset has all three sizes. Clips whose 10s windows overlap another
kept clip from the same recording are dropped (status overlaps_10s).

Produces data/windows/ for experiments/window_length.py. Needs ffmpeg/ffprobe
and the source recordings (not in the repo). Run from the folder that contains
audio/ (source_audio_path in metadata.csv is relative to it):
    python3 <repo>/scripts/make_window_clips.py --csv <repo>/data/metadata.csv --out <repo>/data/windows --list-only
    python3 <repo>/scripts/make_window_clips.py --csv <repo>/data/metadata.csv --out <repo>/data/windows

--list-only writes just the two CSVs. Outputs (3s/, 5s/, 10s/, manifest.csv,
feasibility_all_clips.csv) go in --out. Nothing existing is overwritten: ffmpeg
runs with -n (skip if the file already exists), so re-runs fill gaps only.
"""
import argparse, csv, os, subprocess, sys

# source_audio_path is tried relative to each root, in order:
#   audio/frog/x.flac           -> <run folder>/audio/frog/x.flac
#   ExtremeSSD/Mojave/x.flac    -> /Volumes/ExtremeSSD/Mojave/x.flac
ROOTS = [os.getcwd(), "/Volumes"]
WINDOWS = [3, 5, 10]
EPS = 1e-3  # tolerance for float rounding at recording edges


def resolve(p):
    for root in ROOTS:
        c = os.path.join(root, p)
        if os.path.isfile(c):
            return c
    return None


_dur_cache = {}
def duration(path):
    """Recording length in seconds via ffprobe (cached per recording)."""
    if path not in _dur_cache:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True)
        try:
            _dur_cache[path] = float(out.stdout.strip())
        except ValueError:
            _dur_cache[path] = None
    return _dur_cache[path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="metadata.csv")
    ap.add_argument("--out", default=".", help="where 3s/ 5s/ 10s/ and the CSVs go")
    ap.add_argument("--list-only", action="store_true",
                    help="only write the feasibility + manifest CSVs")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, newline="")))
    os.makedirs(args.out, exist_ok=True)
    feas, manifest = [], []

    for i, r in enumerate(rows, 1):
        clip_id = os.path.splitext(r["filename"])[0]
        species = r["common_name"].replace("/", "-")
        src = resolve(r["source_audio_path"])
        dur = duration(src) if src else None
        ev_s, ev_e = float(r["start_time"]), float(r["end_time"])
        center = (ev_s + ev_e) / 2
        rec = {"clip_id": clip_id, "common_name": species,
               "source_audio_path": r["source_audio_path"],
               "recording_duration_s": dur, "event_start_s": ev_s,
               "event_end_s": ev_e, "event_len_s": round(ev_e - ev_s, 3),
               "center_s": round(center, 3)}

        if src is None:
            rec["status"] = "source_missing"
        elif dur is None:
            rec["status"] = "unreadable_duration"
        for w in WINDOWS:
            s, e = center - w / 2, center + w / 2
            fits = dur is not None and s >= -EPS and e <= dur + EPS
            rec[f"start_{w}s"] = round(max(s, 0.0), 3)
            rec[f"end_{w}s"] = round(e, 3)
            rec[f"fits_{w}s"] = fits
        rec.setdefault("status", "ok" if rec["fits_10s"] else "too_close_to_edge")
        # how far the new centered 3s window is from the existing 3s clip
        old_c = (float(r["start_time_3s"]) + float(r["end_time_3s"])) / 2
        rec["shift_vs_existing_3s_s"] = round(center - old_c, 3)
        rec["event_longer_than_3s"] = (ev_e - ev_s) > 3
        rec["overlaps_with"] = ""
        rec["_src"] = src
        feas.append(rec)
        if i % 200 == 0:
            print(f"checked {i}/{len(rows)}", file=sys.stderr)

    # --- Drop overlapping 10s windows within each recording ---------------
    # Windows that don't overlap at 10s can't overlap at 3s or 5s (same center).
    # Go through each recording in time order and keep a clip only if its
    # 10s window starts at/after the end of the last kept one. Since all
    # windows are the same width, this keeps the maximum number of clips.
    by_rec = {}
    for rec in feas:
        if rec["status"] == "ok":
            by_rec.setdefault(rec["_src"], []).append(rec)
    for recs in by_rec.values():
        recs.sort(key=lambda r: r["center_s"])
        last = None
        for rec in recs:
            if last is not None and rec["center_s"] - 5 < last["center_s"] + 5 - EPS:
                rec["status"] = "overlaps_10s"
                rec["overlaps_with"] = last["clip_id"]
            else:
                last = rec

    for rec in feas:
        if rec["status"] == "ok":
            m = dict(rec, resolved_path=rec["_src"])
            for w in WINDOWS:
                m[f"path_{w}s"] = f"{w}s/{rec['common_name']}/{rec['clip_id']}.wav"
            manifest.append(m)
    for rec in feas:
        del rec["_src"]
    for m in manifest:
        m.pop("_src", None)

    def write(name, data, drop=()):
        cols = [k for k in data[0] if k not in drop]
        with open(os.path.join(args.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(data)

    write("feasibility_all_clips.csv", feas)
    if manifest:
        write("manifest.csv", manifest, drop=("resolved_path",))
    counts = {}
    for f in feas:
        counts[f["status"]] = counts.get(f["status"], 0) + 1
    print(f"{len(manifest)}/{len(rows)} clips kept (fit 3s+5s+10s, no overlapping 10s windows)")
    print("status counts:", counts)

    if args.list_only or not manifest:
        return
    for i, m in enumerate(manifest, 1):
        for w in WINDOWS:
            out = os.path.join(args.out, m[f"path_{w}s"])
            os.makedirs(os.path.dirname(out), exist_ok=True)
            subprocess.run(
                ["ffmpeg", "-nostdin", "-loglevel", "error", "-n",
                 "-i", m["resolved_path"], "-ss", str(m[f"start_{w}s"]),
                 "-t", str(w), out])
        if i % 50 == 0:
            print(f"extracted {i}/{len(manifest)}", file=sys.stderr)
    print("done")


if __name__ == "__main__":
    main()
