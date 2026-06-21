"""Edit AriadneConfigBlock fields submenu."""

from ichor.cli.main_menu_submenus.active_learning_campaign_menu.field_menu import (
    get_attr_path,
    make_field_menu,
    set_attr_path,
    spec,
)
from ichor.cli.menu_description import MenuDescription


EDIT_ARIADNE_BLOCK_MENU_DESCRIPTION = MenuDescription(
    "Edit ARIADNE Block",
    subtitle=(
        "Per-seed ARIADNE optimiser hyperparameters used during the "
        "adversarial attack phase of each iteration.\n"
    ),
)


def _get_block():
    """Return the live AriadneConfigBlock from the parent config editor."""
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_submenus.edit_campaign_config_menu import (
        get_campaign_config,
    )

    return get_campaign_config().ariadne


def _get_value(path: str):
    return get_attr_path(_get_block(), path)


def _set_value(path: str, value):
    set_attr_path(_get_block(), path, value)


def _sync_options_from_block(block):
    """Compatibility no-op; the field menu reads directly from the live block."""
    return None


ARIADNE_FIELD_SPECS = [
    spec(
        "optimiser",
        "choice",
        choices=["trust_region_qn", "dissipative_symplectic"],
        prompt="Optimiser: ",
    ),
    spec(
        "hessian_model",
        "choice",
        choices=["CONSTANT", "ALMLOF", "LINDH", "SCHLEGEL"],
        transform=str.upper,
        prompt="Hessian model: ",
    ),
    spec("max_iter", "int", prompt="Max iterations per seed: "),
    spec("gradf_tol", "float", prompt="Gradient norm tolerance (Hartree/Angstrom): "),
    spec("f_tol", "float", prompt="Value tolerance (Hartree): "),
    spec("delta0", "float", prompt="Initial trust radius (Bohr): "),
    spec("delta_max", "float", prompt="Maximum trust radius (Bohr): "),
    spec("gamma", "float", prompt="Dissipative-symplectic damping (gamma): "),
    spec(
        "fallback_to_ds",
        "bool",
        prompt="Fallback to dissipative_symplectic on BFGS curvature reject: ",
    ),
    spec(
        "trqn_scale_mode",
        "choice",
        choices=["off", "fixed", "adaptive_initial_gradient"],
        prompt="TRQN objective scale mode: ",
    ),
    spec(
        "trqn_target_initial_grad_norm",
        "float",
        prompt="TRQN target initial gradient norm: ",
    ),
    spec(
        "trqn_retry_target_initial_grad_norm",
        "float",
        prompt="TRQN retry target initial gradient norm: ",
    ),
    spec(
        "trqn_min_objective_scale",
        "float",
        prompt="TRQN minimum objective scale: ",
    ),
    spec(
        "trqn_max_objective_scale",
        "float",
        prompt="TRQN maximum objective scale: ",
    ),
    spec(
        "trqn_fixed_objective_scale",
        "float",
        prompt="TRQN fixed objective scale: ",
    ),
    spec(
        "trqn_retry_on_no_proposal",
        "bool",
        prompt="Retry TRQN once after no-proposal failure: ",
    ),
    spec(
        "trqn_backtransform_mode",
        "choice",
        choices=["geodesic", "newton"],
        prompt="TRQN backtransform mode: ",
    ),
    spec(
        "trqn_geodesic_bt_mode",
        "choice",
        choices=["dense", "matrix_free"],
        prompt="TRQN geodesic backtransform mode: ",
    ),
    spec(
        "trqn_geodesic_dt",
        "float",
        prompt="TRQN geodesic ODE step size: ",
    ),
    spec(
        "trqn_geodesic_tol",
        "float",
        prompt="TRQN geodesic ODE tolerance: ",
    ),
    spec(
        "trqn_bt_ic_tol",
        "float",
        prompt="TRQN backtransform IC tolerance: ",
    ),
    spec(
        "trqn_max_backtransform_iter",
        "int",
        prompt="TRQN max backtransform iterations: ",
    ),
    spec(
        "trqn_trust_min",
        "float",
        prompt="TRQN minimum trust radius: ",
    ),
]


edit_ariadne_block_menu = make_field_menu(
    EDIT_ARIADNE_BLOCK_MENU_DESCRIPTION.title,
    EDIT_ARIADNE_BLOCK_MENU_DESCRIPTION.subtitle,
    ARIADNE_FIELD_SPECS,
    _get_value,
    _set_value,
    prologue_text="Current values for this campaign.yaml block:\n",
)


edit_ariadne_block_menu_options = edit_ariadne_block_menu.this_menu_options
