from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Union

import ichor.hpc.global_variables

from consolemenu.items import SubmenuItem
from ichor.cli.console_menu import add_items_to_menu, ConsoleMenu
from ichor.cli.menu_description import MenuDescription
from ichor.cli.menu_options import MenuOptions


@dataclass
class MainMenuMenuOptions(MenuOptions):
    # defaults to the current working directory
    selected_ichor_config_file: Path

    def check_selected_points_directory_path(self) -> Union[str, None]:
        """Checks whether the ichor_config.yaml is present in the user home directory."""
        if not self.selected_ichor_config_file.exists():
            return "The ichor_config.yaml file is not in the home directory!\nIt is required to use the menu system."


# initialize dataclass for storing information for menu
main_menu_menu_options = MainMenuMenuOptions(
    ichor.hpc.global_variables.ICHOR_CONFIG_PATH
)

MAIN_MENU_DESCRIPTION = MenuDescription(
    "Main Menu", subtitle="Welcome to ichor's main menu!"
)

# no main menu options for now
# note: need to have typing on classes, otherwise they will not show up in the prologue
# dataclasses need to have typing


@dataclass
class MainMenuOptions(MenuOptions):
    pass


# make instance of options
main_menu_options = MainMenuOptions()

_SUBMENU_SPECS = [
    (
        "ichor.cli.main_menu_submenus.initial_structure_menu",
        "initial_structure_menu",
        "INITIAL_STRUCTURE_MENU_DESCRIPTION",
    ),
    (
        "ichor.cli.main_menu_submenus.trajectory_creation_menu.trajectory_creation_menu",
        "trajectory_creation_menu",
        "TRAJECTORY_CREATION_MENU_DESCRIPTION",
    ),
    (
        "ichor.cli.main_menu_submenus.sampling_menu",
        "sampling_menu",
        "SAMPLING_MENU_DESCRIPTION",
    ),
    (
        "ichor.cli.main_menu_submenus.points_directory_menu",
        "points_directory_menu",
        "POINTS_DIRECTORY_MENU_DESCRIPTION",
    ),
    (
        "ichor.cli.main_menu_submenus.training_menu.training_menu",
        "training_menu",
        "TRAINING_MENU_DESCRIPTION",
    ),
    (
        "ichor.cli.main_menu_submenus.analysis_menu",
        "analysis_menu",
        "ANALYSIS_MENU_DESCRIPTION",
    ),
    (
        "ichor.cli.main_menu_submenus.tools_menu.tools_menu",
        "tools_menu",
        "TOOLS_MENU_DESCRIPTION",
    ),
    (
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu",
        "active_learning_campaign_menu",
        "ACTIVE_LEARNING_CAMPAIGN_MENU_DESCRIPTION",
    ),
]

main_menu = None


def build_main_menu():
    menu = ConsoleMenu(
        this_menu_options=main_menu_options,
        title=MAIN_MENU_DESCRIPTION.title,
        subtitle=MAIN_MENU_DESCRIPTION.subtitle,
        prologue_text=MAIN_MENU_DESCRIPTION.prologue_description_text,
        epilogue_text=MAIN_MENU_DESCRIPTION.epilogue_description_text,
        show_exit_option=MAIN_MENU_DESCRIPTION.show_exit_option,
    )
    items = []
    for module_name, menu_attr, description_attr in _SUBMENU_SPECS:
        module = import_module(module_name)
        submenu = getattr(module, menu_attr)
        description = getattr(module, description_attr)
        items.append(SubmenuItem(description.title, submenu, menu))
    add_items_to_menu(menu, items)
    return menu


# this function will be used by setuptools entry points
def run_main_menu():
    """Runs main ichor menu."""
    global main_menu
    if main_menu is None:
        main_menu = build_main_menu()
    main_menu.show()
