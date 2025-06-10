"""Leaderboard rendering (CSV + Markdown) from per-model summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from ..data.classes import DETECTION_CLASSES

_MAIN_COLS = [
    ("mAP50", "mAP@50"),
    ("mAP50_95", "mAP@50:95"),
    ("mAP75", "mAP@75"),
    ("occupancy/seat_accuracy", "Seat acc."),
    ("occupancy/child_presence_recall", "Child recall"),
    ("latency_ms_mean", "Latency (ms)"),
]


def _fmt(value, pct: bool) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{100 * value:.2f}" if pct else f"{value:.1f}"


def leaderboard_frame(results: Sequence[Dict], split: str = "test") -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"Model": r["model"], "Family": r["family"], "Params (M)": r.get("params_m")}
        for key, label in _MAIN_COLS:
            row[label] = r.get(f"{split}/{key}")
        row["Train (min)"] = r.get("train_time_min")
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("mAP@50", ascending=False, na_position="last").reset_index(drop=True)


def per_class_frame(results: Sequence[Dict], split: str = "test") -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"Model": r["model"]}
        for name in DETECTION_CLASSES:
            row[name] = r.get(f"{split}/AP50/{name}")
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, pct_cols: Sequence[str]) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for _, row in df.iterrows():
        cells = []
        for col in headers:
            v = row[col]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                cells.append(_fmt(v, col in pct_cols))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_leaderboard(results: List[Dict], out_dir: Path, split: str = "test") -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    board = leaderboard_frame(results, split)
    classes = per_class_frame(results, split)
    pct = [label for key, label in _MAIN_COLS if "latency" not in key]

    csv_path = out_dir / "leaderboard.csv"
    board.to_csv(csv_path, index=False)
    md = [
        f"# CabinSense leaderboard ({split} split)",
        "",
        "Metrics in percent; sorted by mAP@50.",
        "",
        to_markdown(board, pct),
        "",
        "## AP@50 per class",
        "",
        to_markdown(classes, list(DETECTION_CLASSES)),
        "",
    ]
    md_path = out_dir / "leaderboard.md"
    md_path.write_text("\n".join(md))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    return {"csv": csv_path, "markdown": md_path}


def load_summaries(output_dir: Path) -> List[Dict]:
    """Rebuild the leaderboard from ``<output_dir>/*/summary.json`` (e.g. after SageMaker jobs)."""
    return [json.loads(p.read_text()) for p in sorted(Path(output_dir).glob("*/summary.json"))]
