"""Documentation/source consistency checks for active-learning phase order."""
from pathlib import Path
import re

from ichor.hpc.active_learning.daemon.daemon import next_phase
from ichor.hpc.active_learning.daemon.state import CampaignPhase


def test_active_learning_daemon_docs_match_executed_phase_order():
    wanted_phases = [
        CampaignPhase.PHASE_B_DIVERSITY,
        CampaignPhase.SPLIT,
        CampaignPhase.GAUSSIAN,
        CampaignPhase.AIMALL,
        CampaignPhase.ALLOCATION_CHECK,
        CampaignPhase.REFERENCE_COMMIT,
        CampaignPhase.FEREBUS,
    ]
    observed = [wanted_phases[0]]
    while observed[-1] is not CampaignPhase.FEREBUS:
        following, _iteration = next_phase(observed[-1], 1, 2)
        observed.append(following)
    assert observed == wanted_phases
    wanted = [phase.value for phase in wanted_phases]

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
    positions = []
    cursor = 0
    for phase in wanted:
        position = documented_phases.index(phase, cursor)
        positions.append(position)
        cursor = position + 1
    assert positions == sorted(positions)
