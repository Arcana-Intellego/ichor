from pathlib import Path


def test_ichor_hpc_declares_tqdm_dependency():
    setup_cfg = Path(__file__).resolve().parents[2] / "setup.cfg"
    text = setup_cfg.read_text(encoding="utf-8").lower()

    assert "tqdm" in text


def test_ichor_core_declares_metadynamics_runtime_dependencies():
    setup_cfg = Path(__file__).resolve().parents[3] / "ichor_core" / "setup.cfg"
    text = setup_cfg.read_text(encoding="utf-8").lower()

    assert "ase" in text
    assert "xtb" in text
    assert "plumed" in text
