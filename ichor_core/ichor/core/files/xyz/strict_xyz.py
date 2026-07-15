"""Strict XYZ parsing shared by single-frame and trajectory readers."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import List, Union

from ichor.core.atoms import Atom, Atoms


_ATOM_COUNT_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)")


class XYZParseError(ValueError):
    """Raised when an XYZ file does not satisfy the complete frame grammar."""


def read_xyz_frames(path: Union[str, Path]) -> List[Atoms]:
    """Read every complete XYZ frame without skipping undeclared records."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8", newline=None) as handle:
            records = list(enumerate(handle, start=1))
    except (OSError, UnicodeError) as exc:
        raise XYZParseError("XYZ file is unreadable: " + str(source)) from exc

    frames: List[Atoms] = []
    cursor = 0
    n_records = len(records)
    while cursor < n_records:
        while cursor < n_records and not records[cursor][1].strip():
            cursor += 1
        if cursor >= n_records:
            break
        count_line_number, count_line = records[cursor]
        count_text = count_line.strip()
        if _ATOM_COUNT_PATTERN.fullmatch(count_text) is None:
            raise XYZParseError(
                "XYZ line "
                + str(count_line_number)
                + " must contain only a non-negative atom count"
            )
        atom_count = int(count_text)
        cursor += 1
        if cursor >= n_records:
            raise XYZParseError(
                "XYZ frame beginning on line "
                + str(count_line_number)
                + " has no comment line"
            )
        cursor += 1  # The comment line is mandatory and may be empty.
        atoms = Atoms()
        for atom_offset in range(atom_count):
            if cursor >= n_records:
                raise XYZParseError(
                    "XYZ frame beginning on line "
                    + str(count_line_number)
                    + " is truncated: expected "
                    + str(atom_count)
                    + " atom records"
                )
            line_number, atom_line = records[cursor]
            cursor += 1
            fields = atom_line.split()
            if len(fields) < 4:
                raise XYZParseError(
                    "XYZ atom record on line "
                    + str(line_number)
                    + " must contain element, x, y, and z"
                )
            try:
                coordinates = tuple(float(value) for value in fields[1:4])
            except ValueError as exc:
                raise XYZParseError(
                    "XYZ atom record on line "
                    + str(line_number)
                    + " contains a non-numeric coordinate"
                ) from exc
            if not all(math.isfinite(value) for value in coordinates):
                raise XYZParseError(
                    "XYZ atom record on line "
                    + str(line_number)
                    + " contains a non-finite coordinate"
                )
            atom = Atom(fields[0], *coordinates)
            try:
                _ = atom.mass
            except (KeyError, TypeError, ValueError) as exc:
                raise XYZParseError(
                    "XYZ atom record on line "
                    + str(line_number)
                    + " contains an unsupported element: "
                    + repr(fields[0])
                ) from exc
            atoms.add(atom)
        frames.append(atoms)
    return frames


__all__ = ["XYZParseError", "read_xyz_frames"]
