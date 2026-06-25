"""Background-launch submenu for the active-learning daemon.

Selecting "Launch (detached)" spawns 'ichor-al-daemon start ...' or
'ichor-al-daemon resume ...' as a
detached child via :func:'launch_daemon_detached' and returns control to
the menu. The child's PID is written to
<campaign_dir>/.DATA/ACTIVE_LEARNING/menu_launched.pid for later
correlation, and its stdout / stderr are appended to
daemon.menu_launched.out next to it. The daemon's own daemon.lock
flock remains the source of truth for whether the daemon is alive.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import ichor.cli.global_menu_variables
import ichor.hpc.global_variables
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
from ichor.cli.useful_functions.launch_helpers import launch_daemon_detached_checked


START_DAEMON_BACKGROUND_MENU_DESCRIPTION = MenuDescription(
    "Start/Resume Daemon (Background)",
    subtitle=(
        "Spawn a detached daemon process and return to the menu. "
        "The PID is recorded in <campaign_dir>/.DATA/ACTIVE_LEARNING/menu_launched.pid.\n"
    ),
)


START_DAEMON_BACKGROUND_DEFAULTS = {
    "selected_command": "start",
    "selected_mode": "dry-run",
    "selected_poll_interval": 0,
    "selected_max_ticks": 0,
    "selected_config": "",
    "selected_preset": "",
}


@dataclass
class StartDaemonBackgroundMenuOptions(MenuOptions):
    selected_command: str
    selected_mode: str
    selected_poll_interval: int
    selected_max_ticks: int
    selected_config: str
    selected_preset: str


start_daemon_background_menu_options = StartDaemonBackgroundMenuOptions(
    *START_DAEMON_BACKGROUND_DEFAULTS.values()
)


def _format_max_ticks(value):
    return "0 (unlimited)" if int(value) == 0 else str(value)


def _format_poll_interval(value):
    return "0 (campaign default)" if int(value) == 0 else str(value)


def _get_value(path: str):
    return get_attr_path(start_daemon_background_menu_options, path)


def _set_value(path: str, value):
    set_attr_path(start_daemon_background_menu_options, path, value)


class StartDaemonBackgroundFunctions:
    @staticmethod
    def select_command():
        """Pick start or resume."""
        chosen = user_input_restricted(
            ["start", "resume"],
            "Select command: ",
            start_daemon_background_menu_options.selected_command,
        )
        if chosen is not None:
            start_daemon_background_menu_options.selected_command = chosen

    @staticmethod
    def select_mode():
        """Pick mock-ariadne / dry-run / live (restricted)."""
        chosen = user_input_restricted(
            ["mock-ariadne", "dry-run", "live"],
            "Select mode: ",
            start_daemon_background_menu_options.selected_mode,
        )
        if chosen is not None:
            start_daemon_background_menu_options.selected_mode = chosen

    @staticmethod
    def select_poll_interval():
        """Override config.poll_interval_seconds (0 = use config)."""
        start_daemon_background_menu_options.selected_poll_interval = user_input_int(
            "Poll interval seconds (0 to use config default): ",
            start_daemon_background_menu_options.selected_poll_interval,
        )

    @staticmethod
    def select_max_ticks():
        start_daemon_background_menu_options.selected_max_ticks = user_input_int(
            "Max ticks (0 for unlimited): ",
            start_daemon_background_menu_options.selected_max_ticks,
        )

    @staticmethod
    def select_config():
        start_daemon_background_menu_options.selected_config = user_input_free_flow(
            "Config path override (blank to use campaign.yaml): ",
            start_daemon_background_menu_options.selected_config,
        ) or ""

    @staticmethod
    def select_preset():
        start_daemon_background_menu_options.selected_preset = user_input_free_flow(
            "Preset name (blank for none): ",
            start_daemon_background_menu_options.selected_preset,
        ) or ""

    @staticmethod
    def launch():
        """Spawn the detached daemon and return immediately."""
        import importlib

        try:
            campaign_dir = selected_campaign_dir()
        except CampaignSelectionError as exc:
            print_campaign_selection_error(exc)
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        if not campaign_dir.is_dir():
            print("Campaign directory does not exist: " + str(campaign_dir))
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        edit_menu = importlib.import_module(
            "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
            "active_learning_campaign_submenus.edit_campaign_config_menu"
        )
        if edit_menu.has_unsaved_config_changes():
            print("There are unsaved campaign.yaml edits:")
            for path in edit_menu.dirty_paths()[:20]:
                print("  " + path)
            print("Save or discard them before starting the daemon.")
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        try:
            result = launch_daemon_detached_checked(
                campaign_dir,
                command=start_daemon_background_menu_options.selected_command,
                mode=start_daemon_background_menu_options.selected_mode,
                poll_interval=(
                    start_daemon_background_menu_options.selected_poll_interval
                    if start_daemon_background_menu_options.selected_poll_interval > 0
                    else None
                ),
                max_ticks=(
                    start_daemon_background_menu_options.selected_max_ticks
                    if start_daemon_background_menu_options.selected_max_ticks > 0
                    else None
                ),
                config=(
                    Path(start_daemon_background_menu_options.selected_config)
                    if start_daemon_background_menu_options.selected_config
                    else None
                ),
                preset=(
                    start_daemon_background_menu_options.selected_preset
                    if start_daemon_background_menu_options.selected_preset
                    else None
                ),
            )
        except Exception as exc:
            print("Failed to launch detached daemon: " + str(exc))
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        if result.exited_during_startup:
            print(
                "Detached daemon exited during startup with code "
                + str(result.returncode)
            )
        else:
            print("Detached daemon launched with PID " + str(result.pid))
        print("Log: " + str(result.log_path))
        ichor.hpc.global_variables.LOGGER.info(
            "Background daemon launched for campaign " + str(campaign_dir)
            + " with PID " + str(result.pid)
        )
        user_input_free_flow("Press enter to return to the menu: ", "")


start_daemon_background_menu_items = [
    FunctionItem("Launch (detached)", StartDaemonBackgroundFunctions.launch),
]


START_DAEMON_BACKGROUND_FIELD_SPECS = [
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
        "selected_poll_interval",
        "int",
        prompt="Poll interval seconds (0 to use config default): ",
        item_text="Set poll-interval",
        display_path="poll_interval",
        formatter=_format_poll_interval,
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
]


start_daemon_background_menu = make_field_menu(
    START_DAEMON_BACKGROUND_MENU_DESCRIPTION.title,
    START_DAEMON_BACKGROUND_MENU_DESCRIPTION.subtitle,
    START_DAEMON_BACKGROUND_FIELD_SPECS,
    _get_value,
    _set_value,
    prologue_text="Current launch options:\n",
    extra_items=start_daemon_background_menu_items,
)
