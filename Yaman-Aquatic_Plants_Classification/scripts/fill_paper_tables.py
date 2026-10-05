"""Write aggregated results into the placeholder rows of Paper/main_v3.tex.

    python scripts/fill_paper_tables.py            # fill rows whose seeds are complete
    python scripts/fill_paper_tables.py --partial  # also fill rows with fewer seeds
    python scripts/fill_paper_tables.py --dry-run  # show what would change

Rows of ``tab:datagrid`` (Training data & Head & 6 cells) and ``tab:backbones``
(Encoder & Head & 6 cells) are matched by their two label cells to the groups that
``aggregate_results.py`` builds. Cells are mean $\\pm$ std over seeds (one decimal) or the
single value when there is one seed. Rows are only replaced when every one of their six
cells is still ``\\tbd`` unless ``--overwrite`` is given, so hand edits survive re-runs.
``tab:datagrid`` expects seeds 8-12; ``tab:backbones`` uses seeds 8-10 (the ConvNeXt-V2
Tiny rows there are the seed 8-10 subset of the data suite).
"""

from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.aggregate_results import (  # noqa: E402
    BACKBONE_LABELS, DATA_LABELS, HEAD_LABELS, SPLITS, VARIANT_LABELS, collect, fmt, load_runs,
)

PAPER = os.path.join(os.path.dirname(ROOT), "Paper", "main_v3.tex")
FIELDS = ("accuracy", "binary_FNR")
ROW_RE = re.compile(r"^(\s*)([^&%\\]+?)\s*&\s*([^&\\]+?)\s*&\s*(.+?)\s*\\\\\s*$")
TABLES = {
    # label: (first-column kind, seeds)
    "tab:datagrid": ("data", (8, 9, 10, 11, 12)),
    "tab:backbones": ("backbone", (8, 9, 10)),
}


def table_rows(groups, kind, seeds):
    """{(col1 label, col2 label): (cells, n_seeds)} for one table."""
    rows = {}
    for (data, backbone, head, variant, fraction), bucket in groups.items():
        if fraction < 1.0:
            continue
        if kind == "data" and backbone != "convnextv2":
            continue
        if kind == "backbone" and data != "lab+web":
            continue
        used = [s for s in bucket["seeds"] if s in seeds]
        if not used:
            continue
        first = DATA_LABELS.get(data, data) if kind == "data" else BACKBONE_LABELS.get(backbone, backbone)
        second = HEAD_LABELS.get(head, head) + VARIANT_LABELS[variant]
        cells = []
        for split, _ in SPLITS:
            for field in FIELDS:
                values = [v for v, s in zip(bucket.get((split, field), []), bucket["seeds"]) if s in seeds]
                cells.append(fmt(values).replace("+/-", r"$\pm$"))
        rows[(first, second)] = (cells, len(used))
    return rows


def fill(lines, groups, partial, overwrite):
    changed, current, expected = 0, None, None
    for index, line in enumerate(lines):
        label = re.search(r"\\label\{(tab:[^}]+)\}", line)
        if label:
            current = label.group(1) if label.group(1) in TABLES else None
            if current:
                kind, seeds = TABLES[current]
                expected = (table_rows(groups, kind, seeds), len(seeds))
            continue
        if current is None:
            continue
        if r"\end{tabular}" in line:
            current = None
            continue
        match = ROW_RE.match(line)
        if not match:
            continue
        indent, first, second, rest = match.groups()
        cells = [c.strip() for c in rest.split("&")]
        if len(cells) != 6:
            continue
        if not overwrite and any(c != r"\tbd" for c in cells):
            continue
        rows, n_expected = expected
        entry = rows.get((first.strip(), second.strip()))
        if entry is None:
            continue
        new_cells, n_seeds = entry
        if n_seeds < n_expected and not partial:
            print(f"  skip {first.strip()} / {second.strip()}: {n_seeds}/{n_expected} seeds")
            continue
        newline = "\r\n" if line.endswith("\r\n") else "\n"
        lines[index] = f"{indent}{first.strip()} & {second.strip()} & " + " & ".join(new_cells) + r" \\" + newline
        print(f"  {first.strip()} & {second.strip()} & {' & '.join(new_cells)}  ({n_seeds} seeds)")
        changed += 1
    return changed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=os.path.join(ROOT, "runs", "models"))
    parser.add_argument("--paper", default=PAPER)
    parser.add_argument("--partial", action="store_true", help="fill rows even if some seeds are missing")
    parser.add_argument("--overwrite", action="store_true", help="replace rows that already have numbers")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    groups = collect(load_runs(args.runs), show_original=False)
    with open(args.paper, encoding="utf-8") as handle:
        lines = handle.readlines()
    changed = fill(lines, groups, args.partial, args.overwrite)
    remaining = sum(line.count(r"\tbd") for line in lines)
    print(f"{changed} row(s) filled, {remaining} \\tbd cell(s) remain in {os.path.relpath(args.paper, ROOT)}")
    if changed and not args.dry_run:
        with open(args.paper, "w", encoding="utf-8", newline="") as handle:
            handle.writelines(lines)


if __name__ == "__main__":
    main()
