"""Initialise a campaign and import its configured user inputs."""
import argparse
from dataclasses import dataclass

from consolemenu.items import FunctionItem
from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
    CampaignSelectionError,
    print_campaign_selection_error,
    selected_campaign_dir,
)
from ichor.cli.main_menu_submenus.active_learning_campaign_menu.field_menu import (
    get_attr_path,
    make_field_menu,
    set_attr_path,
    spec,
)
from ichor.cli.menu_description import MenuDescription
from ichor.cli.menu_options import MenuOptions
from ichor.cli.useful_functions import user_input_free_flow


IMPORT_TRAJECTORY_POOL_MENU_DESCRIPTION = MenuDescription(
    "Initialise Campaign / Import Inputs",
    subtitle=(
        "Populate campaign.yaml from the active-learning template and import "
        "pool.xyz and any configured bootstrap/ inputs into "
        "campaign-owned storage.\n"
    ),
)


@dataclass
class ImportTrajectoryPoolMenuOptions(MenuOptions):
    source: str = ""
    force_reimport: bool = False


import_trajectory_pool_menu_options = ImportTrajectoryPoolMenuOptions()


def _get_value(path: str):
    return get_attr_path(import_trajectory_pool_menu_options, path)


def _set_value(path: str, value):
    set_attr_path(import_trajectory_pool_menu_options, path, value)


def _pause():
    user_input_free_flow("Press enter to return to the menu: ", "")


class ImportTrajectoryPoolFunctions:
    @staticmethod
    def run_import():
        from ichor.hpc.active_learning.cli import cmd_init

        try:
            campaign_dir = selected_campaign_dir()
        except CampaignSelectionError as exc:
            print_campaign_selection_error(exc)
            _pause()
            return
        source = import_trajectory_pool_menu_options.source or None
        force = bool(import_trajectory_pool_menu_options.force_reimport)
        if force:
            answer = user_input_free_flow(
                "Force reimport invalidates committed provenance. Type YES: ",
                "",
            )
            if answer != "YES":
                print("Cancelled.")
                import_trajectory_pool_menu_options.force_reimport = False
                _pause()
                return
        ns = argparse.Namespace(
            campaign_dir=str(campaign_dir),
            source=source,
            force=force,
            yes=False,
        )
        rc = cmd_init(ns)
        if rc == 13 and not force:
            print("Pool already exists. Set force_reimport=true to replace it.")
        elif rc != 0:
            print("init returned exit code " + str(rc))
        import_trajectory_pool_menu_options.force_reimport = False
        _pause()


IMPORT_TRAJECTORY_POOL_FIELD_SPECS = [
    spec(
        "source",
        "clearable_str",
        prompt="Source trajectory path",
        item_text="Set source path (blank uses campaign pool.xyz)",
    ),
    spec(
        "force_reimport",
        "bool",
        prompt="Force reimport existing pool? ",
        item_text="Set force-reimport",
    ),
]


import_trajectory_pool_menu_items = [
    FunctionItem(
        "Initialise campaign / import configured inputs",
        ImportTrajectoryPoolFunctions.run_import,
    ),
]


import_trajectory_pool_menu = make_field_menu(
    IMPORT_TRAJECTORY_POOL_MENU_DESCRIPTION.title,
    IMPORT_TRAJECTORY_POOL_MENU_DESCRIPTION.subtitle,
    IMPORT_TRAJECTORY_POOL_FIELD_SPECS,
    _get_value,
    _set_value,
    prologue_text="Current init options:\n",
    extra_items=import_trajectory_pool_menu_items,
    status_for_path=lambda _path: "scope: menu-only; applies to next init/import command",
)
