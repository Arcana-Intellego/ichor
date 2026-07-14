"""Check complete ownership of audited daemon findings."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

FINDING_RE = re.compile(r"\bW\d{2}-P[0-3]-\d{3}\b")
PLAN_ROW_RE = re.compile(r"^\| \`(W\d{2}-P[0-3]-\d{3})\` \|", re.MULTILINE)


def audit_ids(path: Path) -> list[str]:
    return sorted(set(FINDING_RE.findall(path.read_text(encoding="utf-8"))))


def plan_ids(path: Path) -> list[str]:
    return PLAN_ROW_RE.findall(path.read_text(encoding="utf-8"))


def check(audit_path: Path, plan_path: Path) -> None:
    audited = audit_ids(audit_path)
    planned = plan_ids(plan_path)
    if len(audited) != 246:
        raise AssertionError(f"expected 246 unique audit IDs, found {len(audited)}")
    if len(planned) != 246:
        raise AssertionError(f"expected 246 plan rows, found {len(planned)}")
    if len(set(planned)) != len(planned):
        raise AssertionError("plan contains duplicate finding ownership")
    if set(audited) != set(planned):
        missing = sorted(set(audited) - set(planned))
        unexpected = sorted(set(planned) - set(audited))
        raise AssertionError(
            f"traceability mismatch; missing={missing}, unexpected={unexpected}"
        )
    text = plan_path.read_text(encoding="utf-8")
    unresolved = [
        finding_id
        for finding_id in planned
        if re.search(
            rf"^\| \`{re.escape(finding_id)}\` \|.*\|\s*(?:TBD|unresolved)\s*\|",
            text,
            re.MULTILINE | re.IGNORECASE,
        )
    ]
    if unresolved:
        raise AssertionError(f"unresolved dispositions: {unresolved}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    check(args.audit, args.plan)
    print("daemon audit traceability: 246/246 findings owned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
