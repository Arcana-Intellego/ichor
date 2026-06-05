from pathlib import Path


def test_ichor_hpc_declares_tqdm_dependency():
    setup_cfg = Path(__file__).resolve().parents[2] / "setup.cfg"
    text = setup_cfg.read_text(encoding="utf-8").lower()

    assert "tqdm" in text
