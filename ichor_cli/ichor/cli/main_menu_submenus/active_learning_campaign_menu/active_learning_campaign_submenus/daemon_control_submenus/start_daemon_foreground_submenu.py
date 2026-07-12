"""Foreground-launch submenu for the active-learning daemon.

Selecting "Launch (blocks)" calls ``cmd_start`` or ``cmd_resume`` with a
synthesised ``argparse.Namespace``. The menu blocks until the daemon exits
(DONE / HALTED / shutdown / max_ticks reached). Good for short test runs with
``--max-ticks`` and for development.
"""
import argparse
from dataclasses import dataclass

import ichor.cli.global_menu_variables
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
from ichor.cli.useful_functions import (
    user_input_free_flow,
    user_input_int,
    user_input_restricted,
)


START_DAEMON_FOREGROUND_MENU_DESCRIPTION = MenuDescription(
    "Start/Resume Daemon (Foreground)",
    subtitle=(
        "Configure mode + optional limits and launch the daemon. The menu "
        "BLOCKS until the daemon exits. Best for short test runs.\n"
    ),
)


START_DAEMON_FOREGROUND_DEFAULTS = {
    "selected_command": "start",
    "selected_mode": "live",
    "selected_max_ticks": 0,            # 0 = unset
    "selected_poll_interval": 0,         # 0 = use config default
    "selected_config": "",
    "selected_preset": "",
    "reopen_converged": False,
}


@dataclass
class StartDaemonForegroundMenuOptions(MenuOptions):
    selected_command: str
    selected_mode: str
    selected_max_ticks: int
    selected_poll_interval: int
    selected_config: str
    selected_preset: str
    reopen_converged: bool


start_daemon_foreground_menu_options = StartDaemonForegroundMenuOptions(
    *START_DAEMON_FOREGROUND_DEFAULTS.values()
)


def _format_max_ticks(value):
    return "0 (unlimited)" if int(value) == 0 else str(value)


def _format_poll_interval(value):
    return "0 (campaign default)" if int(value) == 0 else str(value)


def _get_value(path: str):
    return get_attr_path(start_daemon_foreground_menu_options, path)


def _set_value(path: str, value):
    set_attr_path(start_daemon_foreground_menu_options, path, value)


class StartDaemonForegroundFunctions:
    @staticmethod
    def select_command():
        """Pick start or resume."""
        chosen = user_input_restricted(
            ["start", "resume"],
            "Select command: ",
            start_daemon_foreground_menu_options.selected_command,
        )
        if chosen is not None:
            start_daemon_foreground_menu_options.selected_command = chosen

    @staticmethod
    def select_mode():
        """Pick mock-ariadne / dry-run / live (restricted)."""
        chosen = user_input_restricted(
            ["mock-ariadne", "dry-run", "live"],
            "Select mode: ",
            start_daemon_foreground_menu_options.selected_mode,
        )
        if chosen is not None:
            start_daemon_foreground_menu_options.selected_mode = chosen

    @staticmethod
    def select_max_ticks():
        """Cap the number of state-machine ticks (0 = unlimited)."""
        start_daemon_foreground_menu_options.selected_max_ticks = user_input_int(
            "Max ticks (0 for unlimited): ",
            start_daemon_foreground_menu_options.selected_max_ticks,
        )

    @staticmethod
    def select_poll_interval():
        """Override config.poll_interval_seconds (0 = use config)."""
        start_daemon_foreground_menu_options.selected_poll_interval = user_input_int(
            "Poll interval seconds (0 to use config default): ",
            start_daemon_foreground_menu_options.selected_poll_interval,
        )

    @staticmethod
    def select_config():
        start_daemon_foreground_menu_options.selected_config = user_input_free_flow(
            "Config path override (blank to use campaign.yaml): ",
            start_daemon_foreground_menu_options.selected_config,
        ) or ""

    @staticmethod
    def select_preset():
        start_daemon_foreground_menu_options.selected_preset = user_input_free_flow(
            "Preset name (blank for none): ",
            start_daemon_foreground_menu_options.selected_preset,
        ) or ""

    @staticmethod
    def launch():
        """Launch the daemon in the foreground; the menu blocks until exit."""
        import importlib

        edit_menu = importlib.import_module(
            "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
            "active_learning_campaign_submenus.edit_campaign_config_menu"
        )

        try:
            campaign_dir = selected_campaign_dir()
        except CampaignSelectionError as exc:
            print_campaign_selection_error(exc)
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        if edit_menu.has_unsaved_config_changes():
            print("There are unsaved campaign.yaml edits:")
            for path in edit_menu.dirty_paths()[:20]:
                print("  " + path)
            print("Save or discard them before starting the daemon.")
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        config_override = (
            start_daemon_foreground_menu_options.selected_config
            if start_daemon_foreground_menu_options.selected_config
            else None
        )
        preset_name = (
            start_daemon_foreground_menu_options.selected_preset
            if start_daemon_foreground_menu_options.selected_preset
            else None
        )
        if edit_menu.saved_config_review_failed(config_override, preset_name):
            print("Selected config override could not be reviewed (or selected preset is invalid):")
            print("  " + edit_menu.saved_config_lock_review_error())
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        if edit_menu.saved_config_has_lock_changes(config_override, preset_name):
            if config_override:
                print("Selected config override differs from the config lock:")
            else:
                print("Saved campaign.yaml differs from the config lock:")
            print(edit_menu.format_saved_config_lock_review(config_override, preset_name))
            print("Run reconcile --apply for safe edits or revert blocked edits before starting.")
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        from ichor.hpc.active_learning.cli import cmd_resume, cmd_start

        if (
            start_daemon_foreground_menu_options.reopen_converged
            and start_daemon_foreground_menu_options.selected_command != "resume"
        ):
            print("Completed-campaign reopen is valid only with the resume command.")
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        mode = start_daemon_foreground_menu_options.selected_mode
        ns = argparse.Namespace(
            campaign_dir=str(campaign_dir),
            config=config_override,
            mock_ariadne=(mode == "mock-ariadne"),
            dry_run=(mode == "dry-run"),
            live=(mode == "live"),
            poll_interval=(
                start_daemon_foreground_menu_options.selected_poll_interval
                if start_daemon_foreground_menu_options.selected_poll_interval > 0
                else None
            ),
            max_ticks=(
                start_daemon_foreground_menu_options.selected_max_ticks
                if start_daemon_foreground_menu_options.selected_max_ticks > 0
                else None
            ),
            preset=preset_name,
            reopen_converged=bool(
                start_daemon_foreground_menu_options.reopen_converged
            ),
            foreground=True,
            background=False,
            background_child=False,
            background_log=None,
            background_pid=None,
        )
        if start_daemon_foreground_menu_options.selected_command == "resume":
            rc = cmd_resume(ns)
        else:
            rc = cmd_start(ns)
        print("daemon exited with code " + str(rc))
        user_input_free_flow("Press enter to return to the menu: ", "")


start_daemon_foreground_menu_items = [
    FunctionItem("Launch (blocks)", StartDaemonForegroundFunctions.launch),
]


START_DAEMON_FOREGROUND_FIELD_SPECS = [
    spec(
        "selected_command",
        "choice",
        choices=["start", "resume"],
        prompt="Select command: ",
        item_text="Set command",
        display_path="command",
    ),
    spec(
        "selected_mode",
        "choice",
        choices=["mock-ariadne", "dry-run", "live"],
        prompt="Select mode: ",
        item_text="Set mode",
        display_path="mode",
    ),
    spec(
        "selected_max_ticks",
        "int",
        prompt="Max ticks (0 for unlimited): ",
        item_text="Set max-ticks",
        display_path="max_ticks",
        formatter=_format_max_ticks,
    ),
    spec(
        "selected_poll_interval",
        "int",
        prompt="Poll interval seconds (0 to use config default): ",
        item_text="Set poll-interval",
        display_path="poll_interval",
        formatter=_format_poll_interval,
    ),
    spec(
        "selected_config",
        "clearable_str",
        prompt="Config path override",
        item_text="Set config override",
        display_path="config",
    ),
    spec(
        "selected_preset",
        "clearable_str",
        prompt="Preset name",
        item_text="Set preset",
        display_path="preset",
    ),
    spec(
        "reopen_converged",
        "bool",
        prompt="Explicitly reopen a completed campaign? ",
        item_text="Set explicit completed-campaign reopen",
        display_path="reopen_converged",
    ),
]


start_daemon_foreground_menu = make_field_menu(
    START_DAEMON_FOREGROUND_MENU_DESCRIPTION.title,
    START_DAEMON_FOREGROUND_MENU_DESCRIPTION.subtitle,
    START_DAEMON_FOREGROUND_FIELD_SPECS,
    _get_value,
    _set_value,
    prologue_text="Current launch options:\n",
    extra_items=start_daemon_foreground_menu_items,
    status_for_path=lambda _path: "scope: menu-only; applies to next daemon launch",
)
