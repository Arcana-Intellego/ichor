"""Lazy exports for top-level menu descriptions.

Importing one submenu package should not import every other menu and its
optional workflow dependencies. Active-learning menu tests, for example, do
not need RDKit just because the trajectory/metadynamics menu exists.
"""

_DESCRIPTION_IMPORTS = {
    "INITIAL_STRUCTURE_MENU_DESCRIPTION": (
        "ichor.cli.main_menu_submenus.initial_structure_menu.initial_structure_menu"
    ),
    "ANALYSIS_MENU_DESCRIPTION": (
        "ichor.cli.main_menu_submenus.analysis_menu.analysis_menu"
    ),
    "TRAJECTORY_CREATION_MENU_DESCRIPTION": (
        "ichor.cli.main_menu_submenus.trajectory_creation_menu.trajectory_creation_menu"
    ),
    "POINTS_DIRECTORY_MENU_DESCRIPTION": (
        "ichor.cli.main_menu_submenus.points_directory_menu.points_directory_menu"
    ),
    "TOOLS_MENU_DESCRIPTION": (
        "ichor.cli.main_menu_submenus.tools_menu.tools_menu"
    ),
}

__all__ = sorted(_DESCRIPTION_IMPORTS)


def __getattr__(name):
    if name not in _DESCRIPTION_IMPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module = import_module(_DESCRIPTION_IMPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
