import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DAEMON_ROOT = ROOT / "ichor" / "hpc" / "active_learning" / "daemon"


def _calls(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _call_name(call: ast.Call) -> str:
    return ast.unparse(call.func)


def _star_expressions(call: ast.Call) -> list[str]:
    return [
        ast.unparse(keyword.value)
        for keyword in call.keywords
        if keyword.arg is None
    ]


def test_partial_array_recovery_journals_do_not_expand_raw_summary():
    live_executor = DAEMON_ROOT / "live_executor.py"
    offenders = []
    for call in _calls(live_executor):
        if not _call_name(call).endswith("_journal_event"):
            continue
        if not call.args:
            continue
        event = call.args[0]
        if not isinstance(event, ast.Constant):
            continue
        if str(event.value) not in {
            "partial_array_recovery_prepared",
            "partial_array_recovery_postprocess_only",
        }:
            continue
        if "recovery_summary" in _star_expressions(call):
            offenders.append(call.lineno)
    assert offenders == []


def test_sbatch_journal_does_not_expand_raw_submission_metadata():
    daemon = DAEMON_ROOT / "daemon.py"
    offenders = []
    for call in _calls(daemon):
        if not _call_name(call).endswith("_journal"):
            continue
        if not call.args:
            continue
        event = call.args[0]
        if not isinstance(event, ast.Constant) or event.value != "sbatch":
            continue
        for expression in _star_expressions(call):
            if "submission_metadata" in expression and not expression.startswith(
                "_journal_metadata_payload("
            ):
                offenders.append((call.lineno, expression))
    assert offenders == []
