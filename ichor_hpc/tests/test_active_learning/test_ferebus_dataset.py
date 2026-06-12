"""The per-atom FEREBUS dataset split (csv -> 3 FEREBUS set csvs).

Verified off-cluster against synthetic ichor-format and polus-suffixed feature
csvs. The upstream ichor pointdir -> feature-csv export needs real AIMAll IQA
and is exercised on the cluster, not here.
"""
import random

import pytest

from ichor.hpc.active_learning.daemon.ferebus_dataset import (
    _bare_header,
    plan_sizes,
    split_atom_csv,
    split_atom_csv_to_property_dirs,
    validate_ferebus_csv,
)


def _write_csv(path, header, n_rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        for i in range(n_rows):
            # 3 features + iqa + integration_error
            f.write(f"{i+0.1},{i+0.2},{i+0.3},{-75.0-i*0.01},{i*1e-5}\n")


def test_bare_header_trims_feature_suffix_only():
    # polus-style suffixed feature cols -> bare; property names left intact
    out = _bare_header("f1_O3,f2_C2,f3_O3-C2,iqa,integration_error\n")
    assert out == "f1,f2,f3,iqa,integration_error\n"


def test_bare_header_noop_on_ichor_format():
    out = _bare_header("f1,f2,f3,iqa,integration_error\n")
    assert out == "f1,f2,f3,iqa,integration_error\n"


def test_plan_sizes_fractions_and_slack():
    # 100 rows, 0.8/0.1/0.1 -> 80/10/10
    assert plan_sizes(100, (0.8, 0.1, 0.1)) == (80, 10, 10)
    # training absorbs rounding so the three never exceed the pool
    n_tr, n_iv, n_ev = plan_sizes(7, (0.8, 0.1, 0.1))
    assert n_tr + n_iv + n_ev == 7 and n_tr >= 1
    # tiny pool -> controlled failure, because FEREBUS reads all three CSVs.
    with pytest.raises(ValueError):
        plan_sizes(2, (0.8, 0.1, 0.1))


@pytest.mark.parametrize("header", [
    "f1,f2,f3,iqa,integration_error",            # ichor format
    "f1_O3,f2_C2,f3_O3C2,iqa,integration_error",  # polus-suffixed format
])
def test_split_atom_csv_end_to_end(tmp_path, header):
    src = tmp_path / "WATER_O1.csv"
    _write_csv(src, header, 20)
    out = tmp_path / "staging"
    random.seed(0)
    counts = split_atom_csv(src, out, "WATER", "O1", "iqa", (0.8, 0.1, 0.1))

    # right counts, summing to the pool
    assert counts == {"train": 16, "int_val": 2, "ext_val": 2}
    assert sum(counts.values()) == 20

    train = out / "WATER_O1_TRAINING_SET.csv"
    intv = out / "WATER_O1_INT_VALIDATION_SET.csv"
    extv = out / "WATER_O1_EXT_VALIDATION_SET.csv"
    for p in (train, intv, extv):
        assert p.is_file()

    # header normalised to bare f-names, property names intact
    htxt = train.read_text(encoding="utf-8").splitlines()[0]
    assert htxt == "f1,f2,f3,iqa,integration_error"

    # LF line endings (a stray CR could break the Fortran last-column match)
    assert b"\r" not in train.read_bytes()

    # rows are disjoint across the three sets and cover the whole pool
    def _data_rows(p):
        return p.read_text(encoding="utf-8").splitlines()[1:]
    all_rows = _data_rows(train) + _data_rows(intv) + _data_rows(extv)
    assert len(all_rows) == 20
    assert len(set(all_rows)) == 20  # no row duplicated across sets

    # data preserved verbatim (first feature column parses back to the source)
    f1_vals = sorted(float(r.split(",")[0]) for r in all_rows)
    assert f1_vals == [round(i + 0.1, 1) for i in range(20)]


def test_plan_sizes_tiny_pool_raises():
    for n in (3, 4, 5):
        with pytest.raises(ValueError):
            plan_sizes(n, (0.8, 0.1, 0.1))
    assert plan_sizes(6, (0.8, 0.1, 0.1)) == (2, 2, 2)


def test_split_atom_csv_to_property_dirs_reuses_same_rows(tmp_path):
    src = tmp_path / "WATER_O1.csv"
    with open(src, "w", encoding="utf-8", newline="\n") as f:
        f.write("f1,f2,f3,iqa,q00\n")
        for i in range(20):
            f.write(f"{i+0.1},{i+0.2},{i+0.3},{-75.0-i*0.01},{0.2+i*0.001}\n")
    out_iqa = tmp_path / "iqa"
    out_q00 = tmp_path / "q00"
    random.seed(0)
    result = split_atom_csv_to_property_dirs(
        src,
        {"iqa": out_iqa, "q00": out_q00},
        "WATER",
        "O1",
        ["iqa", "q00"],
        (0.8, 0.1, 0.1),
    )
    assert result["counts"] == {"train": 16, "int_val": 2, "ext_val": 2}
    for prop_dir in (out_iqa, out_q00):
        for kind in ("TRAINING", "INT_VALIDATION", "EXT_VALIDATION"):
            p = prop_dir / f"WATER_O1_{kind}_SET.csv"
            assert p.is_file()
            assert p.read_text(encoding="utf-8").splitlines()[0] == "f1,f2,f3,iqa,q00"
    assert (
        out_iqa / "WATER_O1_TRAINING_SET.csv"
    ).read_text(encoding="utf-8") == (
        out_q00 / "WATER_O1_TRAINING_SET.csv"
    ).read_text(encoding="utf-8")


def test_split_atom_csv_to_property_dirs_honours_explicit_row_ids(tmp_path):
    src = tmp_path / "WATER_O1.csv"
    with open(src, "w", encoding="utf-8", newline="\n") as f:
        f.write("f1,f2,f3,iqa,q00\n")
        for i in range(8):
            f.write(f"{i},{i+0.2},{i+0.3},{-75.0-i*0.01},{0.2+i*0.001}\n")
    out_iqa = tmp_path / "iqa"
    out_q00 = tmp_path / "q00"
    row_ids = {"train": [7, 0, 1, 2], "int_val": [3, 4], "ext_val": [5, 6]}

    result = split_atom_csv_to_property_dirs(
        src,
        {"iqa": out_iqa, "q00": out_q00},
        "WATER",
        "O1",
        ["iqa", "q00"],
        (0.8, 0.1, 0.1),
        row_ids=row_ids,
    )

    assert result["counts"] == {"train": 4, "int_val": 2, "ext_val": 2}
    train_rows = (
        out_iqa / "WATER_O1_TRAINING_SET.csv"
    ).read_text(encoding="utf-8").splitlines()[1:]
    assert [int(row.split(",")[0]) for row in train_rows] == [7, 0, 1, 2]
    assert (
        out_iqa / "WATER_O1_EXT_VALIDATION_SET.csv"
    ).read_text(encoding="utf-8") == (
        out_q00 / "WATER_O1_EXT_VALIDATION_SET.csv"
    ).read_text(encoding="utf-8")


def test_split_atom_csv_to_property_dirs_rejects_incomplete_explicit_partition(tmp_path):
    src = tmp_path / "WATER_O1.csv"
    _write_csv(src, "f1,f2,f3,iqa,integration_error", 8)
    with pytest.raises(ValueError, match="partition every row"):
        split_atom_csv_to_property_dirs(
            src,
            {"iqa": tmp_path / "iqa"},
            "WATER",
            "O1",
            ["iqa"],
            (0.8, 0.1, 0.1),
            row_ids={"train": [0, 1, 2], "int_val": [3, 4], "ext_val": [5]},
        )


def test_validate_ferebus_csv_rejects_missing_and_bad_property(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("f1,f2,f3,iqa\n0.1,0.2,0.3,\n0.4,0.5,0.6,-1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="blank numeric"):
        validate_ferebus_csv(bad, "iqa")
    with pytest.raises(ValueError, match="missing property"):
        validate_ferebus_csv(bad, "q00")


def test_prop_stats_from_iqa_column(tmp_path):
    from ichor.hpc.active_learning.daemon.ferebus_dataset import prop_stats
    src = tmp_path / "set.csv"
    # iqa is the 4th column; values -1,-2,-3,-4 -> mean -2.5, population std sqrt(1.25)
    with open(src, "w", encoding="utf-8", newline="\n") as f:
        f.write("f1,f2,f3,iqa,integration_error\n")
        for v in (-1.0, -2.0, -3.0, -4.0):
            f.write(f"0.1,0.2,0.3,{v},1e-5\n")
    s = prop_stats(src, "iqa")
    assert s["min"] == -4.0 and s["max"] == -1.0 and s["range"] == 3.0
    assert s["mean"] == -2.5 and s["median"] == -2.5
    assert abs(s["std"] - 1.1180339887498949) < 1e-9
    assert abs(s["cv"] - (s["std"] / 2.5)) < 1e-12


def test_prop_stats_floors_degenerate_property_column(tmp_path):
    from ichor.hpc.active_learning.daemon.ferebus_dataset import prop_stats
    src = tmp_path / "constant.csv"
    src.write_text(
        "f1,f2,f3,iqa\n"
        "0.1,0.2,0.3,-75.0\n"
        "0.4,0.5,0.6,-75.0\n",
        encoding="utf-8",
        newline="\n",
    )
    s = prop_stats(src, "iqa")
    assert s["degenerate_property_stats"] is True
    assert s["std"] > 0.0
    assert s["range"] > 0.0


def test_prop_stats_empty_or_missing(tmp_path):
    from ichor.hpc.active_learning.daemon.ferebus_dataset import prop_stats
    # header only, no data rows -> empty dict, caller just omits the keys
    src = tmp_path / "h.csv"
    src.write_text("f1,f2,iqa\n", encoding="utf-8", newline="\n")
    assert prop_stats(src, "iqa") == {}
    # property column not present at all
    src2 = tmp_path / "h2.csv"
    src2.write_text("f1,f2\n0.1,0.2\n", encoding="utf-8", newline="\n")
    assert prop_stats(src2, "iqa") == {}


def test_split_survives_trailing_blank_line(tmp_path):
    # a stray trailing blank used to desync our row count from RS's, which could empty a set.
    # the normalised-copy fix should make the split robust to it -- all three sets non-empty.
    src = tmp_path / "WATER_O1.csv"
    with open(src, "w", encoding="utf-8", newline="\n") as f:
        f.write("f1,f2,f3,iqa,integration_error\n")
        for i in range(8):
            f.write(f"{i+0.1},{i+0.2},{i+0.3},{-75.0-i*0.01},{i*1e-5}\n")
        f.write("\n")  # the troublesome trailing blank
    out = tmp_path / "staging"
    random.seed(0)
    counts = split_atom_csv(src, out, "WATER", "O1", "iqa", (0.8, 0.1, 0.1))
    assert sum(counts.values()) == 8
    assert counts["train"] >= 1 and counts["int_val"] >= 1 and counts["ext_val"] >= 1
    # and the normalised scratch file is cleaned up, not left lying in staging
    assert not (out / "O1_normalised_for_split.csv").exists()
