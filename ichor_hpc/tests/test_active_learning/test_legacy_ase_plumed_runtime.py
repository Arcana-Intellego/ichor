from pathlib import Path

from ichor.core.files import XTB
from ichor.core.files.mtd import MtdTrajScript


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_submit_single_ase_xyz_default_method_is_gfn2_xtb():
    source = (REPO_ROOT / "ichor_hpc/ichor/hpc/main/ase.py").read_text(
        encoding="utf-8"
    )

    assert 'method="GFN2-xTB"' in source
    assert 'method="GFN2-xT",' not in source


def test_generated_xtb_script_imports_xtb_ase_calculator(tmp_path):
    script_path = tmp_path / "xtb_opt.py"
    input_xyz = tmp_path / "mol.xyz"
    xtb_script = XTB(
        script_path,
        input_xyz_path=input_xyz,
        input_xtb_path=input_xyz,
        output_xyz_path=tmp_path / "mol_optimised.xyz",
        traj_path=tmp_path / "mol.traj",
        log_path=tmp_path / "mol.log",
    )

    xtb_script.write()
    text = script_path.read_text(encoding="utf-8")

    assert "from xtb.ase.calculator import XTB" in text
    assert 'XTB(method="GFN2-xTB"' in text


def test_generated_metadynamics_script_has_os_import_and_plain_xtb_method(tmp_path):
    script_path = tmp_path / "mtd.py"
    xyz_path = tmp_path / "mol.xyz"
    mtd_script = MtdTrajScript(
        script_path,
        xyz_path,
        collective_variables=[[0, 1]],
    )

    mtd_script.write()
    text = script_path.read_text(encoding="utf-8")

    assert "import os" in text
    assert 'xtb_calc = XTB(method="GFN2-xTB"' in text
    assert 'method=""GFN2-xTB""' not in text
    assert 'os.environ.get("PLUMED_KERNEL")' in text


def test_generated_metadynamics_cv_indices_are_one_based_for_plumed(tmp_path):
    script_path = tmp_path / "mtd.py"
    xyz_path = tmp_path / "mol.xyz"
    mtd_script = MtdTrajScript(
        script_path,
        xyz_path,
        collective_variables=[[0, 1], [1, 2, 0]],
    )

    mtd_script.write()
    text = script_path.read_text(encoding="utf-8")

    assert '"GROUP ATOMS=1,2 LABEL=g1"' in text
    assert '"GROUP ATOMS=2,3,1 LABEL=g2"' in text
    assert '"m1: DISTANCE ATOMS=g1"' in text
    assert '"m2: ANGLE ATOMS=g2"' in text


def test_single_metadynamics_cv_indices_are_one_based_for_plumed(tmp_path):
    script_path = tmp_path / "mtd.py"
    xyz_path = tmp_path / "mol.xyz"
    mtd_script = MtdTrajScript(
        script_path,
        xyz_path,
        collective_variables=[[0, 1]],
    )

    mtd_script.write()
    text = script_path.read_text(encoding="utf-8")

    assert '"m1: DISTANCE ATOMS=1,2"' in text


def test_metadynamics_submission_uses_python_command_not_anaconda_command():
    source = (
        REPO_ROOT / "ichor_hpc/ichor/hpc/molecular_dynamics/metadynamics.py"
    ).read_text(encoding="utf-8")

    assert "PythonCommand" in source
    assert "AnacondaCommand" not in source
    assert "ensure_metadynamics_available" in source


def test_python_command_exports_configured_plumed_runtime():
    source = (
        REPO_ROOT / "ichor_hpc/ichor/hpc/submission_commands/python_command.py"
    ).read_text(encoding="utf-8")

    assert '"software", "plumed", "kernel_path"' in source
    assert '"software", "plumed", "library_path"' in source
    assert "export PLUMED_KERNEL" in source
    assert "export LD_LIBRARY_PATH" in source
    assert '_configured_modules("python") + _configured_modules("plumed")' in source


def test_python_command_exports_native_paths_before_plumed(monkeypatch):
    from ichor.hpc.submission_commands import python_command

    values = {
        ("software", "python", "library_path"): [
            "/home/user/opt/python-3.11.15/lib",
            "/home/modules/compilers/gcc/11.1.0/lib64",
        ],
        ("software", "plumed", "kernel_path"): (
            "/home/user/opt/plumed-2.10.0/lib/libplumedKernel.so"
        ),
        ("software", "plumed", "library_path"): (
            "/home/user/opt/plumed-2.10.0/lib"
        ),
    }

    monkeypatch.setattr(
        python_command,
        "_config_value",
        lambda *keys, default=None: values.get(tuple(keys), default),
    )
    monkeypatch.setattr(
        python_command,
        "_expand_path",
        lambda value: str(value),
    )

    exports = python_command._configured_plumed_exports()

    assert exports == [
        (
            "export PLUMED_KERNEL="
            "/home/user/opt/plumed-2.10.0/lib/libplumedKernel.so"
        ),
        (
            "export LD_LIBRARY_PATH="
            "/home/user/opt/python-3.11.15/lib:"
            "/home/modules/compilers/gcc/11.1.0/lib64:"
            "/home/user/opt/plumed-2.10.0/lib"
            "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        ),
    ]
