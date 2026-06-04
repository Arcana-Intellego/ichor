"""Edit AriadneConfigBlock fields submenu.

Holds a reference to the parent campaign config's ariadne attribute.
Edits mutate the shared block in place, so when the parent's "Save"
action serialises the campaign config, the ARIADNE edits are included
automatically.
"""
from dataclasses import dataclass

from consolemenu.items import FunctionItem
from ichor.cli.console_menu import ConsoleMenu, add_items_to_menu
from ichor.cli.menu_description import MenuDescription
from ichor.cli.menu_options import MenuOptions
from ichor.cli.useful_functions import (
    user_input_bool,
    user_input_float,
    user_input_int,
    user_input_restricted,
)


EDIT_ARIADNE_BLOCK_MENU_DESCRIPTION = MenuDescription(
    "Edit ARIADNE Block",
    subtitle=(
        "Per-seed ARIADNE optimiser hyperparameters used during the "
        "adversarial attack phase of each iteration.\n"
    ),
)


@dataclass
class EditAriadneBlockMenuOptions(MenuOptions):
    optimiser: str = "trust_region_qn"
    hessian_model: str = "ALMLOF"
    max_iter: int = 200
    gradf_tol: float = 1.0e-4
    f_tol: float = 1.0e-6
    delta0: float = 0.10
    delta_max: float = 0.40
    gamma: float = 0.10
    fallback_to_ds: bool = True


edit_ariadne_block_menu_options = EditAriadneBlockMenuOptions()


def _sync_options_from_block(block):
    """Pull current values from the live AriadneConfigBlock into the
    MenuOptions so the prologue reflects them."""
    edit_ariadne_block_menu_options.optimiser = block.optimiser
    edit_ariadne_block_menu_options.hessian_model = block.hessian_model
    edit_ariadne_block_menu_options.max_iter = block.max_iter
    edit_ariadne_block_menu_options.gradf_tol = block.gradf_tol
    edit_ariadne_block_menu_options.f_tol = block.f_tol
    edit_ariadne_block_menu_options.delta0 = block.delta0
    edit_ariadne_block_menu_options.delta_max = block.delta_max
    edit_ariadne_block_menu_options.gamma = block.gamma
    edit_ariadne_block_menu_options.fallback_to_ds = block.fallback_to_ds


def _get_block():
    """Pull the live AriadneConfigBlock instance from the parent's module
    state via a deferred import to avoid an import cycle. We import the
    *module* (not via the package, which re-exports the ConsoleMenu object
    of the same name) so we can call its ``get_campaign_config``."""
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_submenus.edit_campaign_config_menu import (
        get_campaign_config,
    )

    return get_campaign_config().ariadne


class EditAriadneBlockFunctions:
    @staticmethod
    def select_optimiser():
        block = _get_block()
        chosen = user_input_restricted(
            ["trust_region_qn", "dissipative_symplectic"],
            "Optimiser: ",
            block.optimiser,
        )
        if chosen is not None:
            block.optimiser = chosen
        _sync_options_from_block(block)

    @staticmethod
    def select_hessian_model():
        block = _get_block()
        chosen = user_input_restricted(
            ["CONSTANT", "ALMLOF", "LINDH", "SCHLEGEL"],
            "Hessian model: ",
            block.hessian_model,
        )
        if chosen is not None:
            block.hessian_model = chosen.upper()
        _sync_options_from_block(block)

    @staticmethod
    def select_max_iter():
        block = _get_block()
        block.max_iter = user_input_int("Max iterations per seed: ", block.max_iter)
        _sync_options_from_block(block)

    @staticmethod
    def select_gradf_tol():
        block = _get_block()
        block.gradf_tol = user_input_float(
            "Gradient norm tolerance (Hartree/Angstrom): ", block.gradf_tol,
        )
        _sync_options_from_block(block)

    @staticmethod
    def select_f_tol():
        block = _get_block()
        block.f_tol = user_input_float("Value tolerance (Hartree): ", block.f_tol)
        _sync_options_from_block(block)

    @staticmethod
    def select_trust_radii():
        block = _get_block()
        block.delta0 = user_input_float("Initial trust radius (Bohr): ", block.delta0)
        block.delta_max = user_input_float("Maximum trust radius (Bohr): ", block.delta_max)
        _sync_options_from_block(block)

    @staticmethod
    def select_gamma():
        block = _get_block()
        block.gamma = user_input_float(
            "Dissipative-symplectic damping (gamma): ", block.gamma,
        )
        _sync_options_from_block(block)

    @staticmethod
    def select_fallback():
        block = _get_block()
        block.fallback_to_ds = user_input_bool(
            "Fallback to dissipative_symplectic on BFGS curvature reject: ",
            block.fallback_to_ds,
        )
        _sync_options_from_block(block)


edit_ariadne_block_menu_items = [
    FunctionItem("Edit optimiser", EditAriadneBlockFunctions.select_optimiser),
    FunctionItem("Edit hessian model", EditAriadneBlockFunctions.select_hessian_model),
    FunctionItem("Edit max iterations", EditAriadneBlockFunctions.select_max_iter),
    FunctionItem("Edit gradient tolerance", EditAriadneBlockFunctions.select_gradf_tol),
    FunctionItem("Edit value tolerance", EditAriadneBlockFunctions.select_f_tol),
    FunctionItem("Edit trust radii", EditAriadneBlockFunctions.select_trust_radii),
    FunctionItem("Edit gamma (DS damping)", EditAriadneBlockFunctions.select_gamma),
    FunctionItem("Edit fallback policy", EditAriadneBlockFunctions.select_fallback),
]


edit_ariadne_block_menu = ConsoleMenu(
    this_menu_options=edit_ariadne_block_menu_options,
    title=EDIT_ARIADNE_BLOCK_MENU_DESCRIPTION.title,
    subtitle=EDIT_ARIADNE_BLOCK_MENU_DESCRIPTION.subtitle,
    prologue_text=EDIT_ARIADNE_BLOCK_MENU_DESCRIPTION.prologue_description_text,
    epilogue_text=EDIT_ARIADNE_BLOCK_MENU_DESCRIPTION.epilogue_description_text,
    show_exit_option=EDIT_ARIADNE_BLOCK_MENU_DESCRIPTION.show_exit_option,
)


add_items_to_menu(edit_ariadne_block_menu, edit_ariadne_block_menu_items)
