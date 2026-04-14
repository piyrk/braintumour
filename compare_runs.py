from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def format_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_rows(baseline: dict, augmented: dict):
    keys = sorted(set(baseline.keys()) | set(augmented.keys()))
    rows = []
    for key in keys:
        base_value = baseline.get(key, "-")
        aug_value = augmented.get(key, "-")
        rows.append((key, format_value(base_value), format_value(aug_value)))
    return rows


def render_markdown_table(rows):
    lines = ["| Metric | Baseline | Augmented |", "| --- | --- | --- |"]
    for metric, baseline, augmented in rows:
        lines.append(f"| {metric} | {baseline} | {augmented} |")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare baseline and augmented metric JSON files.")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline metrics JSON file.")
    parser.add_argument("--augmented", type=Path, required=True, help="Augmented metrics JSON file.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/comparison_table.md"), help="Output markdown table.")
    return parser.parse_args()


def main():
    args = parse_args()
    baseline = load_json(args.baseline)
    augmented = load_json(args.augmented)

    rows = build_rows(baseline, augmented)
    table = render_markdown_table(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")

    print(table)
    print(f"Saved comparison table to {args.output}")


if __name__ == "__main__":
    main()
