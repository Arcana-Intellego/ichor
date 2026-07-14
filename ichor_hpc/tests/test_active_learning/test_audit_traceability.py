from pathlib import Path

import pytest

from scripts.check_daemon_audit_traceability import audit_ids, check


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
SNAPSHOT = REPO_ROOT / "docs" / "audits" / "daemon-audit-finding-ids.txt"
AUDIT = WORKSPACE_ROOT / "daemon-pipeline-full-audit.md"
PLAN = WORKSPACE_ROOT / "daemon-pipeline-comprehensive-patch-plan.md"


def test_audit_finding_snapshot_is_complete_and_unique():
    identifiers = [
        line.strip()
        for line in SNAPSHOT.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(identifiers) == 246
    assert len(set(identifiers)) == 246
    assert identifiers == sorted(identifiers)


@pytest.mark.skipif(
    not AUDIT.is_file() or not PLAN.is_file(),
    reason="workspace audit planning documents are not distributed with the package",
)
def test_workspace_audit_and_plan_have_exact_traceability():
    check(AUDIT, PLAN)
    snapshot_ids = [
        line.strip()
        for line in SNAPSHOT.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert audit_ids(AUDIT) == snapshot_ids

