"""Per-atom FEREBUS dataset splitting for the live training step.

ICHOR writes one feature+property csv per atom from the committed training set
(f1..fN columns, then property columns like iqa). FEREBUS reads three csvs per
atom instead:

    <system>_<atom>_TRAINING_SET.csv
    <system>_<atom>_INT_VALIDATION_SET.csv
    <system>_<atom>_EXT_VALIDATION_SET.csv

this module does that split. which rows go to train / internal-val / external-val
is chosen by POLUS's RS sampler (the canonical random sampler), but the csvs are
written here rather than through RS.write_data_set -- that method slices the last
column to strip a trailing newline, which is fragile when the property is the
last column, so we serialise the selected rows ourselves and keep the data
verbatim.

the header is normalised so feature columns are bare f1..fN (FEREBUS only treats
a column as a feature if it is "f" followed by a number). ichor already writes
bare names so this is a no-op there; a polus-style f1_O3 header gets the suffix
trimmed. property names (iqa, integration_error, ...) are left untouched.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from .import_utils import quiet_import_module

# feature column written with an alf suffix, e.g. f1_O3 -> we want f1. anything
# that is not f-then-digits-then-underscore (iqa, integration_error, q00) is left
# exactly as-is.
_FEATURE_SUFFIXED = re.compile(r"^f\d+_")


def _bare_header(header_line: str) -> str:
    cols = header_line.rstrip("\n").rstrip("\r").split(",")
    out = []
    for c in cols:
        c = c.strip()
        if _FEATURE_SUFFIXED.match(c):
            out.append(c.split("_")[0])
        else:
            out.append(c)
    return ",".join(out) + "\n"


def _read_csv(source_csv: Path) -> Tuple[str, List[str]]:
    with open(source_csv, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        raise ValueError("empty feature csv: " + str(source_csv))
    return lines[0], lines[1:]


def _row_count(source_csv: Path) -> int:
    _, body = _read_csv(source_csv)
    # ignore a possible blank trailing line
    return sum(1 for ln in body if ln.strip())


MIN_ROWS_PER_FEREBUS_SET = 2
MIN_ROWS_FOR_FEREBUS = 3 * MIN_ROWS_PER_FEREBUS_SET


def plan_sizes(n_rows: int, fractions: Sequence[float]) -> Tuple[int, int, int]:
    """Turn (train, int_val, ext_val) fractions into row counts for n_rows.

    FEREBUS reads all three files before much of its train/validation branching, and its CSV reader
    rejects nrows < 2. Treat a too-small pool as a staging failure rather than emitting malformed
    or header-only validation files.
    """
    f_tr, f_iv, f_ev = (float(x) for x in fractions)
    if n_rows < MIN_ROWS_FOR_FEREBUS:
        raise ValueError(
            "FEREBUS requires at least "
            + str(MIN_ROWS_FOR_FEREBUS)
            + " rows to write train/internal/external CSVs with >= "
            + str(MIN_ROWS_PER_FEREBUS_SET)
            + " rows each; got "
            + str(int(n_rows))
        )
    n_iv = max(MIN_ROWS_PER_FEREBUS_SET, int(round(f_iv * n_rows)))
    n_ev = max(MIN_ROWS_PER_FEREBUS_SET, int(round(f_ev * n_rows)))
    n_tr = n_rows - n_iv - n_ev
    if n_tr < MIN_ROWS_PER_FEREBUS_SET:
        deficit = MIN_ROWS_PER_FEREBUS_SET - n_tr
        while deficit > 0 and n_ev > MIN_ROWS_PER_FEREBUS_SET:
            n_ev -= 1
            deficit -= 1
        while deficit > 0 and n_iv > MIN_ROWS_PER_FEREBUS_SET:
            n_iv -= 1
            deficit -= 1
        n_tr = n_rows - n_iv - n_ev
    if n_tr < MIN_ROWS_PER_FEREBUS_SET:
        raise ValueError("FEREBUS split could not allocate at least two rows to every set")
    return n_tr, n_iv, n_ev


def split_indices(
    source_csv: Path, prop: str, fractions: Sequence[float],
) -> Tuple[List[int], List[int], List[int]]:
    """Disjoint train / int-val / ext-val row ids chosen by POLUS RS.

    Determinism is the caller's job: seed `random` before calling if a
    reproducible split is wanted (RS uses the stdlib `random` module).
    """
    RS = quiet_import_module("polus.samplers.RS.randomSampling").RS

    n = _row_count(source_csv)
    n_tr, n_iv, n_ev = plan_sizes(n, fractions)
    rs = RS(str(source_csv), prop)
    tr = [int(i) for i in rs.get_training_point_IDs(n_tr)]
    iv = [int(i) for i in rs.get_validation_point_IDs(n_iv)] if n_iv else []
    ev = [int(i) for i in rs.get_test_point_IDs(n_ev)] if n_ev else []
    return tr, iv, ev


def _write_subset(header_bare: str, body: List[str], row_ids: Sequence[int], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="\n") as f:
        f.write(header_bare)
        for i in row_ids:
            if not (0 <= i < len(body)):
                raise ValueError(
                    "POLUS RS returned out-of-range row id "
                    + str(i)
                    + " for body length "
                    + str(len(body))
                )
            line = body[i].rstrip("\n").rstrip("\r")
            f.write(line + "\n")


def validate_ferebus_csv(csv_path: Path, prop: str) -> Dict[str, int]:
    """Validate the exact CSV shape FEREBUS_CPU reads.

    FEREBUS classifies headers named f<number> as features and then looks for the requested
    property column. It also rejects files with fewer than four columns or fewer than two data
    rows. Keep those checks on the daemon side so bad staging halts before cluster time is spent.
    """
    p = Path(csv_path)
    header, body = _read_csv(p)
    cols = [c.strip() for c in header.rstrip("\n").rstrip("\r").split(",")]
    if len(cols) < 4:
        raise ValueError("FEREBUS CSV has fewer than four columns: " + str(p))
    features = [i for i, c in enumerate(cols) if re.fullmatch(r"f\d+", c)]
    if not features:
        raise ValueError("FEREBUS CSV has no f<number> feature columns: " + str(p))
    if prop not in cols:
        raise ValueError("FEREBUS CSV missing property " + repr(prop) + ": " + str(p))
    prop_idx = cols.index(prop)
    rows = [ln.rstrip("\n").rstrip("\r") for ln in body if ln.strip()]
    if len(rows) < MIN_ROWS_PER_FEREBUS_SET:
        raise ValueError(
            "FEREBUS CSV "
            + str(p)
            + " has "
            + str(len(rows))
            + " data rows; need at least "
            + str(MIN_ROWS_PER_FEREBUS_SET)
        )
    for row_no, row in enumerate(rows, start=2):
        parts = [x.strip() for x in row.split(",")]
        if len(parts) != len(cols):
            raise ValueError(
                "FEREBUS CSV row "
                + str(row_no)
                + " has "
                + str(len(parts))
                + " columns; expected "
                + str(len(cols))
                + ": "
                + str(p)
            )
        for idx in features + [prop_idx]:
            if parts[idx] == "":
                raise ValueError("FEREBUS CSV blank numeric cell at row " + str(row_no))
            try:
                float(parts[idx])
            except ValueError as exc:
                raise ValueError(
                    "FEREBUS CSV non-numeric value "
                    + repr(parts[idx])
                    + " at row "
                    + str(row_no)
                    + ", column "
                    + cols[idx]
                    + ": "
                    + str(p)
                ) from exc
    return {"rows": len(rows), "columns": len(cols), "features": len(features)}


def split_atom_csv_to_property_dirs(
    source_csv: Path,
    out_dirs_by_prop: Mapping[str, Path],
    system: str,
    atom: str,
    properties: Sequence[str],
    fractions: Sequence[float],
    *,
    row_ids: Mapping[str, Sequence[int]] = None,
) -> Dict[str, object]:
    """Split one atom CSV once and write the same row partition for every property.

    The CSV may contain multiple target columns. FEREBUS selects the active one through `-P` /
    `properties = [...]`, so every property-specific copy keeps the full header and all target
    columns.
    """
    props = [str(p) for p in properties]
    if not props:
        raise ValueError("at least one FEREBUS property is required")
    source_csv = Path(source_csv)
    header_raw, body = _read_csv(source_csv)
    body = [ln for ln in body if ln.strip()]
    header_bare = _bare_header(header_raw)

    # Validate requested properties before asking POLUS RS to read the file; its error path exits.
    cols = [c.strip() for c in header_bare.rstrip("\n").rstrip("\r").split(",")]
    for prop in props:
        if prop not in cols:
            raise ValueError(
                "requested FEREBUS property "
                + repr(prop)
                + " is missing from "
                + str(source_csv)
            )

    if row_ids is None:
        out_root = Path(next(iter(out_dirs_by_prop.values()))).parent
        out_root.mkdir(parents=True, exist_ok=True)
        norm_csv = out_root / (atom + "_normalised_for_split.csv")
        norm_csv.write_text(
            header_raw.rstrip("\n").rstrip("\r") + "\n"
            + "".join(ln.rstrip("\n").rstrip("\r") + "\n" for ln in body),
            encoding="utf-8",
            newline="\n",
        )
        try:
            tr, iv, ev = split_indices(norm_csv, props[0], fractions)
        finally:
            try:
                norm_csv.unlink()
            except OSError:
                pass
    else:
        try:
            tr = [int(i) for i in row_ids["train"]]
            iv = [int(i) for i in row_ids["int_val"]]
            ev = [int(i) for i in row_ids["ext_val"]]
        except Exception as exc:
            raise ValueError("explicit FEREBUS row_ids must contain train/int_val/ext_val") from exc
        all_ids = tr + iv + ev
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("explicit FEREBUS row_ids contain duplicates")
        expected = set(range(len(body)))
        if set(all_ids) != expected:
            raise ValueError(
                "explicit FEREBUS row_ids must partition every row exactly once"
            )

    targets = (
        (system + "_" + atom + "_TRAINING_SET.csv", tr),
        (system + "_" + atom + "_INT_VALIDATION_SET.csv", iv),
        (system + "_" + atom + "_EXT_VALIDATION_SET.csv", ev),
    )
    counts = {"train": len(tr), "int_val": len(iv), "ext_val": len(ev)}
    row_ids = {
        "train": [int(i) for i in tr],
        "int_val": [int(i) for i in iv],
        "ext_val": [int(i) for i in ev],
    }
    validation: Dict[str, Dict[str, int]] = {}
    for prop in props:
        out_dir = Path(out_dirs_by_prop[prop])
        for name, ids in targets:
            _write_subset(header_bare, body, ids, out_dir / name)
        for name, _ids in targets:
            info = validate_ferebus_csv(out_dir / name, prop)
            validation[prop + ":" + name] = info
    return {"counts": counts, "row_ids": row_ids, "validation": validation}


def split_atom_csv(
    source_csv: Path,
    out_dir: Path,
    system: str,
    atom: str,
    prop: str,
    fractions: Sequence[float],
) -> Dict[str, int]:
    """Split one per-atom feature csv into the three FEREBUS set csvs.

    Returns the row count written to each set. Seed `random` before calling for
    a reproducible split.
    """
    result = split_atom_csv_to_property_dirs(
        source_csv,
        {str(prop): Path(out_dir)},
        system,
        atom,
        [str(prop)],
        fractions,
    )
    return dict(result["counts"])


def prop_stats(csv_path, prop: str = "iqa") -> Dict[str, float]:
    """per-property scaling stats FEREBUS wants in its config (the target_prop_* keys), worked out
    from the property column of a written set csv: min / max / range / mean / median / std / cv.

    these differ a LOT between atoms (an oxygen iqa energy is nowhere near a hydrogen one) which is
    exactly why we hand each atom its own ferebus toml rather than one shared config. returns an
    empty dict when the column is missing or there are no rows, and the caller just omits the keys.
    """
    import math as _math

    p = Path(csv_path)
    header, body = _read_csv(p)
    cols = [c.strip() for c in header.rstrip("\n").rstrip("\r").split(",")]
    if prop not in cols:
        return {}
    idx = cols.index(prop)
    vals: List[float] = []
    for ln in body:
        if not ln.strip():
            continue
        parts = ln.rstrip("\n").rstrip("\r").split(",")
        if idx >= len(parts):
            continue
        try:
            vals.append(float(parts[idx]))
        except ValueError:
            continue
    if not vals:
        return {}
    n = len(vals)
    mean = sum(vals) / n
    srt = sorted(vals)
    mid = n // 2
    median = srt[mid] if n % 2 else 0.5 * (srt[mid - 1] + srt[mid])
    # population variance -- these are a scaling anchor, not an inference, so dividing by n is fine.
    # guard the all-equal/single-row case so we dont sqrt a tiny negative rounding crumb.
    var = sum((v - mean) ** 2 for v in vals) / n
    std = _math.sqrt(var) if var > 0.0 else 0.0
    prop_range = srt[-1] - srt[0]
    degenerate = (
        not _math.isfinite(std)
        or not _math.isfinite(prop_range)
        or std <= 0.0
        or prop_range <= 0.0
    )
    floor = 1.0e-12
    if not _math.isfinite(std) or std <= 0.0:
        std = floor
    if not _math.isfinite(prop_range) or prop_range <= 0.0:
        prop_range = floor
    # coefficient of variation is std/|mean|; some properties sit near zero mean so guard it.
    cv = std / abs(mean) if mean != 0.0 else 0.0
    return {
        "min": srt[0], "max": srt[-1], "range": prop_range,
        "mean": mean, "median": median, "std": std, "cv": cv,
        "degenerate_property_stats": bool(degenerate),
    }
