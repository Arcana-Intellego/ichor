"""Strict XYZ parsing shared by single-frame and trajectory readers."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterator, List, TextIO, Tuple, Union

from ichor.core.atoms import Atom, Atoms


_ATOM_COUNT_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)")


class XYZParseError(ValueError):
    """Raised when an XYZ file does not satisfy the complete frame grammar."""


def _numbered_lines(handle: TextIO) -> Iterator[Tuple[int, str]]:
    for line_number, line in enumerate(handle, start=1):
        yield line_number, line


def _iter_xyz_frames(handle: TextIO) -> Iterator[Atoms]:
    """Parse XYZ frames from an open text stream using the complete grammar."""
    records = _numbered_lines(handle)
    while True:
        try:
            count_line_number, count_line = next(records)
        except StopIteration:
            return
        while not count_line.strip():
            try:
                count_line_number, count_line = next(records)
            except StopIteration:
                return
        count_text = count_line.strip()
        if _ATOM_COUNT_PATTERN.fullmatch(count_text) is None:
            raise XYZParseError(
                "XYZ line "
                + str(count_line_number)
                + " must contain only a non-negative atom count"
            )
        atom_count = int(count_text)
        try:
            next(records)  # The comment line is mandatory and may be empty.
        except StopIteration:
            raise XYZParseError(
                "XYZ frame beginning on line "
                + str(count_line_number)
                + " has no comment line"
            )
        atoms = Atoms()
        for _atom_offset in range(atom_count):
            try:
                line_number, atom_line = next(records)
            except StopIteration:
                raise XYZParseError(
                    "XYZ frame beginning on line "
                    + str(count_line_number)
                    + " is truncated: expected "
                    + str(atom_count)
                    + " atom records"
                )
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
        yield atoms


def iter_xyz_frames(path: Union[str, Path]) -> Iterator[Atoms]:
    """Yield every complete XYZ frame without materialising the full file."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8", newline=None) as handle:
            yield from _iter_xyz_frames(handle)
    except XYZParseError:
        raise
    except (OSError, UnicodeError) as exc:
        raise XYZParseError("XYZ file is unreadable: " + str(source)) from exc


def read_xyz_frames(path: Union[str, Path]) -> List[Atoms]:
    """Read every complete XYZ frame without skipping undeclared records."""
    return list(iter_xyz_frames(path))


__all__ = ["XYZParseError", "iter_xyz_frames", "read_xyz_frames"]
