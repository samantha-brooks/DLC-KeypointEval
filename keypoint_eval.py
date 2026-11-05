"""
DLC-KeypointEval: per-keypoint accuracy statistics for a DeepLabCut model.

DeepLabCut's ``evaluate_network`` reports a single train error and a single test
error averaged over *all* keypoints. A model can look fine overall while a few
keypoints are poor. This script re-uses the files ``evaluate_network`` already
wrote (the predictions .h5) and the labeled ground truth, and reports the same
kind of statistics *for each keypoint*, separately for the train and test images.

NOTE ON "RMSE": the number DeepLabCut prints as "Train/Test error (px)" is the
mean Euclidean distance between prediction and label (it is computed as a mean
of pixel errors, not a root-mean-square). To make your per-keypoint numbers
directly comparable, this tool reports that quantity as ``Mean_error_px`` and
also reports a true root-mean-square as ``RMSE_px``. The ALL row of
``Mean_error_px`` should reproduce DLC's own numbers.

Usage (after running deeplabcut.evaluate_network on your project):

    python keypoint_eval.py --config path/to/config.yaml
    python keypoint_eval.py --config path/to/config.yaml --shuffle 1 --pcutoff 0.6

Outputs are written to  <evaluation folder>/keypoint-eval/  (or --out):
    keypoint_eval.xlsx          one sheet per split + worst predictions
    per_keypoint_<split>.csv    same tables as CSV (train / test / all)
    worst_predictions.csv       the largest individual errors, per keypoint
    per_keypoint_errors.png     bar + box plot of error per keypoint
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS = (5.0, 10.0, 20.0)


# --------------------------------------------------------------------------- #
# Locating DLC files
# --------------------------------------------------------------------------- #
def load_config(config_path: Path) -> dict:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("Task", "date", "scorer", "TrainingFraction"):
        if key not in cfg:
            raise ValueError(f"'{key}' not found in {config_path}; is this a DLC config.yaml?")
    return cfg


def training_set_dir(project: Path, cfg: dict) -> Path:
    return (
        project
        / "training-datasets"
        / f"iteration-{cfg.get('iteration', 0)}"
        / f"UnaugmentedDataSet_{cfg['Task']}{cfg['date']}"
    )


def evaluation_dir(project: Path, cfg: dict, train_frac: float, shuffle: int) -> Path | None:
    """The folder evaluate_network wrote into (PyTorch or TensorFlow engine)."""
    sub = (
        Path(f"iteration-{cfg.get('iteration', 0)}")
        / f"{cfg['Task']}{cfg['date']}-trainset{int(round(train_frac * 100))}shuffle{shuffle}"
    )
    candidates = []
    if cfg.get("eval_prefix"):
        candidates.append(project / cfg["eval_prefix"] / sub)
    candidates += [project / "evaluation-results-pytorch" / sub, project / "evaluation-results" / sub]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def _epoch_number(path: Path) -> int:
    """Trailing number of a predictions file name, e.g. '..._snapshot_best-120.h5' -> 120."""
    m = re.search(r"(\d+)(?!.*\d)", path.stem)
    return int(m.group(1)) if m else -1


def choose_predictions_file(files: list[Path], snapshot: str | None = None) -> Path:
    """
    Pick which evaluated snapshot to report on.

    Mirrors DeepLabCut: with the default ``snapshotindex: -1`` (what video analysis
    uses), the "best" snapshot is chosen if one was saved; otherwise the snapshot
    trained for the most epochs. ``snapshot`` (a text fragment) overrides this.
    """
    files = list(files)
    if snapshot:
        files = [f for f in files if snapshot in f.name]
    if not files:
        raise FileNotFoundError(
            "No predictions .h5 found"
            + (f" matching '{snapshot}'" if snapshot else "")
            + ". Run deeplabcut.evaluate_network first."
        )
    if snapshot:
        chosen = max(files, key=lambda p: (_epoch_number(p), p.stat().st_mtime))
    else:
        best = [f for f in files if "best" in f.name.lower()]
        if best:
            chosen = max(best, key=lambda p: p.stat().st_mtime)
            print(f"Using the BEST snapshot (what DLC uses to analyze videos): {chosen.name}")
        else:
            chosen = max(files, key=lambda p: (_epoch_number(p), p.stat().st_mtime))
            print(f"No 'best' snapshot was evaluated; using the highest-epoch one: {chosen.name}")
    others = [f.name for f in files if f != chosen]
    if others:
        print(f"  (other evaluated snapshots ignored: {', '.join(others)}; use --snapshot <text> to choose)")
    return chosen


def find_predictions_file(eval_dir: Path, snapshot: str | None) -> Path:
    return choose_predictions_file(sorted(eval_dir.glob("*.h5")), snapshot)


def find_ground_truth(project: Path, cfg: dict) -> Path:
    name = f"CollectedData_{cfg['scorer']}.h5"
    p = training_set_dir(project, cfg) / name
    if p.exists():
        return p
    raise FileNotFoundError(f"Ground-truth file not found: {p}")


def load_split_ids(project: Path, cfg: dict, train_frac: float, shuffle: int):
    """Returns (train_ids, test_ids) as row positions in the ground-truth table, or None."""
    doc = training_set_dir(project, cfg) / (
        f"Documentation_data-{cfg['Task']}_{int(round(train_frac * 100))}shuffle{shuffle}.pickle"
    )
    if not doc.exists():
        return None
    with open(doc, "rb") as f:
        meta = pickle.load(f)
    return [int(i) for i in meta[1]], [int(i) for i in meta[2]]


# --------------------------------------------------------------------------- #
# Error computation
# --------------------------------------------------------------------------- #
def _index_keys(idx: pd.Index) -> list[str]:
    if isinstance(idx, pd.MultiIndex):
        return ["/".join(str(p) for p in t) for t in idx]
    return [str(t).replace("\\", "/") for t in idx]


def _keypoint_labels(columns: pd.MultiIndex) -> pd.Index:
    """Per-column keypoint label: 'bodypart', or 'individual/bodypart' for multi-animal."""
    nlev = columns.nlevels
    if nlev == 3:  # scorer, bodyparts, coords
        return pd.Index([str(c[1]) for c in columns])
    if nlev == 4:  # scorer, individuals, bodyparts, coords
        return pd.Index([f"{c[1]}/{c[2]}" for c in columns])
    raise ValueError(f"Unexpected number of column levels ({nlev}) in DLC table.")


def to_wide(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """DLC table -> {keypoint: DataFrame(index=image key, columns=[x, y, (likelihood)])}."""
    labels = _keypoint_labels(df.columns)
    coords = [str(c[-1]) for c in df.columns]
    keys = _index_keys(df.index)
    out: dict[str, pd.DataFrame] = {}
    order: list[str] = []
    for kp in labels:
        if kp not in out:
            out[kp] = None  # type: ignore[assignment]
            order.append(kp)
    for kp in order:
        cols = {}
        for j, (lab, co) in enumerate(zip(labels, coords)):
            if lab == kp:
                cols[co] = df.iloc[:, j].to_numpy(dtype=float)
        out[kp] = pd.DataFrame(cols, index=keys)
    return out


def compute_errors(
    gt: pd.DataFrame, pred: pd.DataFrame, split_by_key: dict[str, str] | None = None
) -> pd.DataFrame:
    """
    Long table with one row per (image, keypoint) that has a ground-truth label.

    Columns: image, keypoint, split, gt_x, gt_y, pred_x, pred_y, likelihood,
             dx, dy, error (Euclidean px; NaN when no prediction was made).
    """
    gt_w, pred_w = to_wide(gt), to_wide(pred)
    missing = [k for k in gt_w if k not in pred_w]
    if missing:
        raise ValueError(f"Keypoints in ground truth but not in predictions: {missing}")

    frames = []
    for kp, g in gt_w.items():
        g = g[~g["x"].isna() & ~g["y"].isna()]
        g = g[~g.index.duplicated(keep="first")]
        p = pred_w[kp]
        p = p[~p.index.duplicated(keep="first")].reindex(g.index)
        dx = p["x"].to_numpy() - g["x"].to_numpy()
        dy = p["y"].to_numpy() - g["y"].to_numpy()
        frames.append(
            pd.DataFrame(
                {
                    "image": g.index,
                    "keypoint": kp,
                    "gt_x": g["x"].to_numpy(),
                    "gt_y": g["y"].to_numpy(),
                    "pred_x": p["x"].to_numpy(),
                    "pred_y": p["y"].to_numpy(),
                    "likelihood": p["likelihood"].to_numpy() if "likelihood" in p else np.nan,
                    "dx": dx,
                    "dy": dy,
                    "error": np.hypot(dx, dy),
                }
            )
        )
    errs = pd.concat(frames, ignore_index=True)
    if split_by_key is not None:
        errs["split"] = errs["image"].map(split_by_key).fillna("unassigned")
    else:
        errs["split"] = "all"
    return errs


def summarize(errs: pd.DataFrame, pcutoff: float, thresholds=DEFAULT_THRESHOLDS) -> pd.Series:
    """Summary statistics for one group of (image, keypoint) rows."""
    n_labeled = len(errs)
    e = errs["error"].to_numpy()
    valid = ~np.isnan(e)
    ev = e[valid]
    row: dict[str, float] = {"N_labeled": n_labeled, "N_predicted": int(valid.sum())}

    def stat(fn, arr):
        return float(fn(arr)) if arr.size else np.nan

    row["Mean_error_px"] = stat(np.mean, ev)  # == DLC's reported "RMSE"
    row["RMSE_px"] = stat(lambda a: np.sqrt(np.mean(a**2)), ev)
    row["Median_error_px"] = stat(np.median, ev)
    row["SD_error_px"] = stat(lambda a: np.std(a, ddof=1) if a.size > 1 else np.nan, ev)
    row["P90_error_px"] = stat(lambda a: np.percentile(a, 90), ev)
    row["P95_error_px"] = stat(lambda a: np.percentile(a, 95), ev)
    row["Max_error_px"] = stat(np.max, ev)
    row["Bias_x_px"] = stat(np.mean, errs["dx"].to_numpy()[valid])
    row["Bias_y_px"] = stat(np.mean, errs["dy"].to_numpy()[valid])
    for t in thresholds:
        row[f"Pct_within_{t:g}px"] = 100.0 * float((ev <= t).sum()) / n_labeled if n_labeled else np.nan

    lik = errs["likelihood"].to_numpy()
    row["Mean_likelihood"] = stat(np.mean, lik[~np.isnan(lik)])
    conf = valid & (lik >= pcutoff)
    row["N_above_pcutoff"] = int(conf.sum())
    row["Pct_above_pcutoff"] = 100.0 * conf.sum() / n_labeled if n_labeled else np.nan
    ec = e[conf]
    row["Mean_error_px_pcutoff"] = stat(np.mean, ec)
    row["RMSE_px_pcutoff"] = stat(lambda a: np.sqrt(np.mean(a**2)), ec)
    return pd.Series(row)


def per_keypoint_table(
    errs: pd.DataFrame, keypoint_order: list[str], pcutoff: float, thresholds=DEFAULT_THRESHOLDS
) -> pd.DataFrame:
    rows = {}
    for kp in keypoint_order:
        sub = errs[errs["keypoint"] == kp]
        rows[kp] = summarize(sub, pcutoff, thresholds)
    rows["ALL"] = summarize(errs, pcutoff, thresholds)
    table = pd.DataFrame(rows).T
    table.index.name = "Keypoint"
    for c in ("N_labeled", "N_predicted", "N_above_pcutoff"):
        table[c] = table[c].astype(int)
    ranks = table.loc[keypoint_order, "Mean_error_px"].rank(ascending=False, method="min")
    table["Rank_worst_first"] = ranks.reindex(table.index)
    return table


def worst_predictions(errs: pd.DataFrame, keypoint_order: list[str], n: int) -> pd.DataFrame:
    parts = []
    for kp in keypoint_order:
        sub = errs[(errs["keypoint"] == kp) & errs["error"].notna()]
        parts.append(sub.nlargest(n, "error"))
    cols = ["keypoint", "split", "image", "error", "likelihood", "gt_x", "gt_y", "pred_x", "pred_y"]
    return pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def plot_errors(errs: pd.DataFrame, tables: dict[str, pd.DataFrame], keypoint_order, out: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return False

    ref = "test" if "test" in tables else "all"
    order = (
        tables[ref].loc[keypoint_order, "Mean_error_px"].sort_values(ascending=True, na_position="first").index.tolist()
    )
    n = len(order)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(4, 0.38 * n + 1.5)), sharey=True)
    y = np.arange(n)
    splits = [s for s in ("train", "test") if s in tables] or ["all"]
    h = 0.8 / len(splits)
    colors = {"train": "#4C78A8", "test": "#E4572E", "all": "#4C78A8"}
    for i, s in enumerate(splits):
        vals = tables[s].loc[order, "Mean_error_px"].to_numpy()
        ax1.barh(y + (i - (len(splits) - 1) / 2) * h, vals, height=h, label=s, color=colors[s])
    ax1.set_yticks(y)
    ax1.set_yticklabels(order)
    ax1.set_xlabel("Mean error (px)  [DLC's reported 'RMSE']")
    ax1.set_title("Mean error per keypoint")
    ax1.legend()
    ax1.grid(axis="x", alpha=0.3)

    sub = errs if ref == "all" else errs[errs["split"] == ref]
    data = [sub.loc[sub["keypoint"] == kp, "error"].dropna().to_numpy() for kp in order]
    ax2.boxplot(data, vert=False, positions=y, widths=0.6, showfliers=True, flierprops={"markersize": 3})
    ax2.set_xlabel("Error (px)")
    ax2.set_title(f"Error distribution per keypoint ({ref} images)")
    ax2.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def write_outputs(out_dir: Path, tables, worst, errs, keypoint_order) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for s, t in tables.items():
        p = out_dir / f"per_keypoint_{s}.csv"
        t.to_csv(p, float_format="%.3f")
        written.append(p)
    p = out_dir / "worst_predictions.csv"
    worst.to_csv(p, index=False, float_format="%.3f")
    written.append(p)
    try:
        xlsx = out_dir / "keypoint_eval.xlsx"
        with pd.ExcelWriter(xlsx) as xw:
            for s, t in tables.items():
                t.round(3).to_excel(xw, sheet_name=s)
            worst.round(3).to_excel(xw, sheet_name="worst_predictions", index=False)
        written.append(xlsx)
    except ImportError:
        print("openpyxl not installed; skipping .xlsx (CSVs were written).")
    png = out_dir / "per_keypoint_errors.png"
    if plot_errors(errs, tables, keypoint_order, png):
        written.append(png)
    return written


def print_table(name: str, table: pd.DataFrame, pcutoff: float) -> None:
    cols = ["N_labeled", "Mean_error_px", "RMSE_px", "Median_error_px", "P95_error_px", "Max_error_px",
            "Pct_above_pcutoff", "Mean_error_px_pcutoff"]
    body = table.drop(index="ALL").sort_values("Mean_error_px", ascending=False)
    show = pd.concat([body, table.loc[["ALL"]]])[cols]
    show = show.rename(columns={"Pct_above_pcutoff": f"%>=p{pcutoff:g}", "Mean_error_px_pcutoff": f"Mean_px@p{pcutoff:g}"})
    print(f"\n=== {name.upper()} images (worst keypoint first; mean error in px) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.2f}".format):
        print(show.to_string())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(
    config: Path,
    shuffle: int = 1,
    trainingsetindex: int = 0,
    snapshot: str | None = None,
    predictions: Path | None = None,
    pcutoff: float | None = None,
    thresholds=DEFAULT_THRESHOLDS,
    worst: int = 5,
    out: Path | None = None,
    quiet: bool = False,
) -> dict:
    config = Path(config)
    project = config.parent
    cfg = load_config(config)
    train_frac = cfg["TrainingFraction"][trainingsetindex]
    if pcutoff is None:
        pcutoff = float(cfg.get("pcutoff", 0.6))

    if predictions is None:
        ev = evaluation_dir(project, cfg, train_frac, shuffle)
        if ev is None:
            raise FileNotFoundError(
                f"No evaluation folder found for shuffle {shuffle} / training fraction {train_frac}. "
                "Run deeplabcut.evaluate_network first, or pass --predictions."
            )
        predictions = find_predictions_file(ev, snapshot)
    else:
        predictions = Path(predictions)
        ev = predictions.parent
    print(f"Predictions : {predictions}")

    gt_path = find_ground_truth(project, cfg)
    print(f"Ground truth: {gt_path}")
    gt = pd.read_hdf(gt_path)
    pred = pd.read_hdf(predictions)

    split_by_key = None
    ids = load_split_ids(project, cfg, train_frac, shuffle)
    if ids is None:
        print("WARNING: train/test split file not found; reporting all images together.")
    else:
        gt_keys = _index_keys(gt.index)
        split_by_key = {}
        for i in ids[0]:
            split_by_key[gt_keys[i]] = "train"
        for i in ids[1]:
            split_by_key[gt_keys[i]] = "test"

    errs = compute_errors(gt, pred, split_by_key)
    keypoint_order = list(dict.fromkeys(errs["keypoint"]))
    n_pred_missing = int(errs["error"].isna().sum())
    if n_pred_missing:
        print(f"NOTE: {n_pred_missing} labeled points had no prediction (counted as undetected).")

    tables = {}
    for s in ("train", "test"):
        sub = errs[errs["split"] == s]
        if len(sub):
            tables[s] = per_keypoint_table(sub, keypoint_order, pcutoff, thresholds)
    tables["all"] = per_keypoint_table(errs, keypoint_order, pcutoff, thresholds)

    if not quiet:
        for s in ("test", "train") if "test" in tables else ("all",):
            if s in tables:
                print_table(s, tables[s], pcutoff)

    wp = worst_predictions(errs, keypoint_order, worst)
    out_dir = Path(out) if out else ev / "keypoint-eval"
    written = write_outputs(out_dir, tables, wp, errs, keypoint_order)
    print("\nWrote:")
    for w in written:
        print(f"  {w}")
    return {"tables": tables, "errors": errs, "worst": wp, "out_dir": out_dir}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Per-keypoint accuracy statistics for a DeepLabCut model.")
    ap.add_argument("--config", required=True, type=Path, help="Path to the project's config.yaml")
    ap.add_argument("--shuffle", type=int, default=1, help="Shuffle number (default 1)")
    ap.add_argument("--trainingsetindex", type=int, default=0, help="Index into TrainingFraction (default 0)")
    ap.add_argument("--snapshot", help="Only use predictions files whose name contains this text")
    ap.add_argument("--predictions", type=Path, help="Predictions .h5 written by evaluate_network (overrides auto-search)")
    ap.add_argument("--pcutoff", type=float, help="Likelihood cutoff (default: pcutoff from config.yaml)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS),
                    help="Pixel thresholds for '% within' columns (default 5 10 20)")
    ap.add_argument("--worst", type=int, default=5, help="Worst predictions to list per keypoint (default 5)")
    ap.add_argument("--out", type=Path, help="Output folder (default: <evaluation folder>/keypoint-eval)")
    args = ap.parse_args(argv)
    try:
        run(args.config, args.shuffle, args.trainingsetindex, args.snapshot, args.predictions,
            args.pcutoff, tuple(args.thresholds), args.worst, args.out)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
