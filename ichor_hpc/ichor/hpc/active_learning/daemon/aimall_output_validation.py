"""Shared structural completion contract for one AIMAll point directory."""
from __future__ import annotations

import argparse
import hashlib
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

from ..strict_json import strict_json as json


AIMALL_STRUCTURAL_INVALID_EXIT_CODE = 86

AIMALL_OUTPUT_VALID = "valid"
AIMALL_OUTPUT_INVALID = "output_invalid"
AIMALL_AUTHORITY_INVALID = "authority_invalid"


@dataclass(frozen=True)
class AIMAllOutputAssessment:
    """One non-mutating structural assessment and its stability fingerprint."""

    category: str
    reason: str
    fingerprint_sha256: str

    @property
    def valid(self) -> bool:
        return self.category == AIMALL_OUTPUT_VALID


def _stat_record(path: Path) -> dict[str, Any]:
    try:
        observed = path.lstat()
    except OSError as exc:
        return {
            "path": path.name,
            "missing": True,
            "error": type(exc).__name__,
        }
    record = {
        "path": path.name,
        "mode": int(observed.st_mode),
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "size": int(observed.st_size),
        "mtime_ns": int(observed.st_mtime_ns),
    }
    if stat.S_ISREG(observed.st_mode):
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            after = path.lstat()
        except OSError as exc:
            record["content_error"] = type(exc).__name__
        else:
            after_identity = (
                int(after.st_dev),
                int(after.st_ino),
                int(after.st_size),
                int(after.st_mtime_ns),
            )
            before_identity = (
                int(observed.st_dev),
                int(observed.st_ino),
                int(observed.st_size),
                int(observed.st_mtime_ns),
            )
            if after_identity == before_identity:
                record["sha256"] = digest.hexdigest()
            else:
                record["content_changed"] = True
                record["after_identity"] = list(after_identity)
    return record


def _fingerprint_payload(root: Path) -> Tuple[str, Optional[str]]:
    root_record = _stat_record(root)
    records = [root_record]
    root_mode = root_record.get("mode")
    if (
        not isinstance(root_mode, int)
        or stat.S_ISLNK(root_mode)
        or not stat.S_ISDIR(root_mode)
    ):
        return _sha256_payload({"root": str(root), "records": records}), None
    try:
        children = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        payload = {
            "root": str(root),
            "records": records,
            "listing_error": type(exc).__name__,
        }
        return _sha256_payload(payload), type(exc).__name__
    for child in children:
        if not child.name.endswith("_atomicfiles"):
            continue
        child_record = _stat_record(child)
        records.append(child_record)
        child_mode = child_record.get("mode")
        if (
            not isinstance(child_mode, int)
            or stat.S_ISLNK(child_mode)
            or not stat.S_ISDIR(child_mode)
        ):
            continue
        try:
            atomic_children = sorted(child.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            records.append(
                {
                    "path": child.name + "/",
                    "listing_error": type(exc).__name__,
                }
            )
            continue
        for atomic_child in atomic_children:
            if atomic_child.suffix.lower() == ".int":
                record = _stat_record(atomic_child)
                record["path"] = child.name + "/" + atomic_child.name
                records.append(record)
    return _sha256_payload({"root": str(root), "records": records}), None


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_parsed_aimall_pointdir(pointdir: Any) -> tuple[bool, str]:
    """Validate the parser-level contract without applying quality gates."""
    ints = getattr(pointdir, "ints", None)
    if ints is None or not getattr(ints, "path", None):
        return False, "missing_atomicfiles_dir"
    int_path = Path(ints.path)
    if not int_path.is_dir():
        return False, "atomicfiles_not_a_dir"
    try:
        n_int = 0
        for int_file in ints.ints:
            _ = int_file.net_charge
            n_int += 1
    except Exception:
        return False, "int_parse_failure"
    if n_int == 0:
        return False, "no_int_files"
    try:
        n_atoms = len(pointdir.atoms)
    except Exception:
        return False, "aimall_geometry_unreadable"
    if n_int != n_atoms:
        return False, "aimall_partial_" + str(n_int) + "_of_" + str(n_atoms) + "_int"
    return True, ""


def assess_aimall_output(pointdir: Path) -> AIMAllOutputAssessment:
    """Assess path safety, terminal markers, atom identity and parser success."""
    from ichor.core.files.point_directory import PointDirectory

    root = Path(pointdir)
    fingerprint, listing_error = _fingerprint_payload(root)
    try:
        root_stat = root.lstat()
    except OSError:
        return AIMAllOutputAssessment(
            AIMALL_AUTHORITY_INVALID,
            "pointdir_missing_or_unreadable",
            fingerprint,
        )
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        return AIMAllOutputAssessment(
            AIMALL_AUTHORITY_INVALID,
            "pointdir_not_regular_directory",
            fingerprint,
        )
    if listing_error is not None:
        return AIMAllOutputAssessment(
            AIMALL_OUTPUT_INVALID,
            "atomicfiles_directory_unreadable",
            fingerprint,
        )
    try:
        parsed = PointDirectory(root)
        atoms = list(parsed.atoms)
        expected_atoms = [str(atom.name).capitalize() for atom in atoms]
    except Exception:
        return AIMAllOutputAssessment(
            AIMALL_AUTHORITY_INVALID,
            "geometry_unreadable",
            fingerprint,
        )
    if not expected_atoms or len(expected_atoms) != len(set(expected_atoms)):
        return AIMAllOutputAssessment(
            AIMALL_AUTHORITY_INVALID,
            "geometry_atom_identity_invalid",
            fingerprint,
        )
    try:
        atomic_directories = sorted(
            (
                child
                for child in root.iterdir()
                if child.name.endswith("_atomicfiles")
            ),
            key=lambda item: item.name,
        )
    except OSError:
        return AIMAllOutputAssessment(
            AIMALL_OUTPUT_INVALID,
            "atomicfiles_directory_unreadable",
            fingerprint,
        )
    if len(atomic_directories) != 1:
        return AIMAllOutputAssessment(
            AIMALL_OUTPUT_INVALID,
            "missing_or_ambiguous_atomicfiles_directory",
            fingerprint,
        )
    atomic_directory = atomic_directories[0]
    try:
        atomic_stat = atomic_directory.lstat()
    except OSError:
        return AIMAllOutputAssessment(
            AIMALL_OUTPUT_INVALID,
            "atomicfiles_directory_unreadable",
            fingerprint,
        )
    if stat.S_ISLNK(atomic_stat.st_mode):
        return AIMAllOutputAssessment(
            AIMALL_AUTHORITY_INVALID,
            "atomicfiles_directory_symlinked",
            fingerprint,
        )
    if not stat.S_ISDIR(atomic_stat.st_mode):
        return AIMAllOutputAssessment(
            AIMALL_OUTPUT_INVALID,
            "atomicfiles_directory_missing",
            fingerprint,
        )
    try:
        int_paths = sorted(
            (
                path
                for path in atomic_directory.iterdir()
                if path.suffix.lower() == ".int" and "_" not in path.name
            ),
            key=lambda item: item.name,
        )
    except OSError:
        return AIMAllOutputAssessment(
            AIMALL_OUTPUT_INVALID,
            "int_files_unreadable",
            fingerprint,
        )
    if len(int_paths) != len(expected_atoms):
        return AIMAllOutputAssessment(
            AIMALL_OUTPUT_INVALID,
            "missing_or_partial_int_set_"
            + str(len(int_paths))
            + "_of_"
            + str(len(expected_atoms)),
            fingerprint,
        )
    observed_atoms = []
    for int_path in int_paths:
        try:
            int_stat = int_path.lstat()
        except OSError:
            return AIMAllOutputAssessment(
                AIMALL_OUTPUT_INVALID,
                "int_file_unreadable:" + int_path.name,
                fingerprint,
            )
        if stat.S_ISLNK(int_stat.st_mode):
            return AIMAllOutputAssessment(
                AIMALL_AUTHORITY_INVALID,
                "int_file_symlinked:" + int_path.name,
                fingerprint,
            )
        if not stat.S_ISREG(int_stat.st_mode):
            return AIMAllOutputAssessment(
                AIMALL_OUTPUT_INVALID,
                "int_file_missing:" + int_path.name,
                fingerprint,
            )
        observed_atoms.append(int_path.stem.capitalize())
        try:
            text = int_path.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeError):
            return AIMAllOutputAssessment(
                AIMALL_OUTPUT_INVALID,
                "int_file_unreadable:" + int_path.name,
                fingerprint,
            )
        if "Total time" not in text:
            return AIMAllOutputAssessment(
                AIMALL_OUTPUT_INVALID,
                "int_file_incomplete:" + int_path.name,
                fingerprint,
            )
    if sorted(observed_atoms) != sorted(expected_atoms):
        return AIMAllOutputAssessment(
            AIMALL_OUTPUT_INVALID,
            "int_atom_identity_mismatch",
            fingerprint,
        )
    ok, reason = validate_parsed_aimall_pointdir(parsed)
    if not ok:
        return AIMAllOutputAssessment(
            AIMALL_OUTPUT_INVALID,
            str(reason or "int_parse_failure"),
            fingerprint,
        )
    try:
        parsed_atoms = sorted(
            str(int_file.atom_name).capitalize()
            for int_file in parsed.ints.ints
        )
    except Exception:
        return AIMAllOutputAssessment(
            AIMALL_OUTPUT_INVALID,
            "int_atom_identity_unreadable",
            fingerprint,
        )
    if parsed_atoms != sorted(expected_atoms):
        return AIMAllOutputAssessment(
            AIMALL_OUTPUT_INVALID,
            "int_atom_identity_mismatch",
            fingerprint,
        )
    return AIMAllOutputAssessment(AIMALL_OUTPUT_VALID, "", fingerprint)


def aimall_visibility_issue(pointdir: Path) -> Optional[str]:
    assessment = assess_aimall_output(pointdir)
    return None if assessment.valid else assessment.reason


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate one completed AIMAll task output",
    )
    parser.add_argument("--pointdir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        assessment = assess_aimall_output(Path(args.pointdir))
    except Exception as exc:
        print(
            "AIMAll output validator failed: "
            + type(exc).__name__
            + ": "
            + str(exc),
            file=sys.stderr,
        )
        return 2
    if assessment.valid:
        return 0
    print("AIMAll output invalid: " + assessment.reason, file=sys.stderr)
    if assessment.category == AIMALL_OUTPUT_INVALID:
        return AIMALL_STRUCTURAL_INVALID_EXIT_CODE
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AIMALL_AUTHORITY_INVALID",
    "AIMALL_OUTPUT_INVALID",
    "AIMALL_OUTPUT_VALID",
    "AIMALL_STRUCTURAL_INVALID_EXIT_CODE",
    "AIMAllOutputAssessment",
    "aimall_visibility_issue",
    "assess_aimall_output",
    "validate_parsed_aimall_pointdir",
]
