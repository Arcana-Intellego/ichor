"""Legacy direct FEREBUS TOML writer.

The live daemon path now delegates FEREBUS config/script generation to
pyferebus. This writer remains for old tests and ad-hoc diagnostics only, but
it is intentionally strict: direct TOML must include the atom and 1-indexed ALF
instead of emitting the unsafe historical blank ``atoms =`` line.
"""
from __future__ import annotations

from pathlib import Path


def write_ferebus_toml(
    path,
    *,
    system_name: str,
    natoms: int,
    atom: str,
    alf,
    properties=("iqa",),
    kernel: str = "rbfc_per",
    training: int = 1,
    validation: int = 1,
    target_prop=None,
):
    atom_name = str(atom).strip()
    if not atom_name:
        raise ValueError("atom is required for direct FEREBUS TOML")
    try:
        alf_values = [int(x) for x in alf]
    except TypeError as exc:
        raise ValueError("alf must be an iterable of three 1-indexed integers") from exc
    if len(alf_values) != 3 or any(v <= 0 for v in alf_values):
        raise ValueError("alf must contain exactly three positive 1-indexed integers")
    props = "[" + ", ".join('"' + str(p) + '"' for p in properties) + "]"
    atoms_line = (
        'atoms = [{name="' + atom_name + '", alf=['
        + ", ".join(str(v) for v in alf_values)
        + "]}]"
    )
    # mirrors pyferebus template_config.py key order + default values. the few
    # campaign-driven ones (name / natoms / properties / kernel / training /
    # validation) are filled from arguments; the rest are the template defaults.
    lines = [
        'name = "' + str(system_name) + '"',
        "natoms = " + str(int(natoms)),
        "properties = " + props,
        atoms_line,
        'optimiser = "gwo_ruhl"',
        "first_gen_pop_size = 100",
        'training_protocol = "ihocv"',
        "training = " + str(int(training)),
        "validation = " + str(int(validation)),
        "popProps = 0",
        "nil_test = 0",
        "min_theta_alf = 0.0",
        "max_theta_alf = 0.1",
        "min_theta = 0.0",
        "max_theta = 0.1",
        'min_WN = "1.0E-10"',
        'max_WN = "1.0E-4"',
        "min_KPF = 1.0",
        "max_KPF = 1.0",
        "iterations = 200",
        "reinitialize = 1",
        "number_of_agents = 20",
        'kernel = "' + str(kernel) + '"',
        "nLuckyAgents = 5",
        "UpdateFreq = 5",
        "iqaDeviationFactor = 1.0",
        "ndim_stoc_relaxation = -1",
        "ARD = 1",
        "full_ARD = 1",
        "scaling = 1",
        "scale_angles = 1",
        "scale_prop = 1",
        "scale_feats = 1",
        "feat_min = 0.0",
        "feat_max = 1.0",
        "unscale_noise = 0",
        "max_reg_weight = 40000.0",
        "mean_type = 15",
        "cmean_type_range_factor = 5.0",
        "prefactor = -1",
        "KPM = 2",
        'ihocv_loss = "mse"',
        "is_constant_noise = 1",
        "early_stopping_omega = 4000000",
        "earlyStop = 0",
        "stagnation_check_parameter = 1.2",
        "regularisation_noise = 1.0E-4",
        "print_opt = 1",
        "print_sol = 0",
        "max_dumping = 2.0",
        "min_dumping = 0.0",
        "a_decay_factor = 1.0",
        "min_A_vector = -2.0",
        "max_A_vector = 2.0",
        "min_C_vector = 0.0",
        "max_C_vector = 2.0",
        "batch_size = 100",
        "weights_lambda = 1.0e-10",
        "transfer_learning = 0",
        "final_check = 0",
        "split_ratio = 0.25",
        'split_method = "random"',
        "relax_weight = 0.25",
        "gwo_cycles = 1",
        "elitism = 1",
        "print_step = 1",
        "ruhl = 5",
        "lucky_move = 0.05",
        "multisource = 0",
        "full_seeding = 1",
        "nsources = 4",
        "seedRNG = 0",
        "NP = 0.01",
        'level_of_theory = "b3lyp/6-31+g(d,p)"',
    ]
    # pyferebus writes target_prop_{min,max,range,mean,median,std,cv} from the property data and,
    # with scaling on, FEREBUS leans on them. we now fill them (computed per atom from that atom's
    # iqa column, see ferebus_dataset.prop_stats) instead of leaving them out. when target_prop is
    # None we still omit them -- same as before -- so the off-cluster split tests with no real iqa
    # keep working.
    if target_prop:
        for _k in ("min", "max", "range", "mean", "median", "std", "cv"):
            if _k in target_prop:
                lines.append("target_prop_" + _k + " = " + repr(float(target_prop[_k])))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return p
