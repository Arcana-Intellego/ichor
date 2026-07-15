"""Documentation/source consistency checks for active-learning phase order."""
from pathlib import Path
import re

from ichor.hpc.active_learning.daemon.daemon import PHASE_ORDER


def test_active_learning_daemon_docs_match_executed_phase_order():
    wanted = ["PHASE_B_DIVERSITY", "SPLIT", "GAUSSIAN", "AIMALL", "APPEND"]
    source_slice = [phase.value for phase in PHASE_ORDER if phase.value in wanted]
    assert source_slice == wanted

    repo_root = Path(__file__).resolve().parents[3]
    docs = repo_root / "docs" / "source" / "active_learning_daemon.rst"
    text = docs.read_text(encoding="utf-8")
    start = text.index("Architecture sketch")
    end = text.index("State transitions", start)
    architecture = text[start:end]

    documented_phases = re.findall(
        r"^\s{3}([A-Z][A-Z0-9_]+)\b",
        architecture,
        flags=re.MULTILINE,
    )
    positions = [documented_phases.index(phase) for phase in wanted]
    assert positions == sorted(positions)
