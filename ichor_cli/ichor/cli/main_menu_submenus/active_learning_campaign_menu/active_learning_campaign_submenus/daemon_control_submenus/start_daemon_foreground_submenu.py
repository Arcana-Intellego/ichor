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
from ichor.cli.console_menu import ConsoleMenu, add_items_to_menu
from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
    CampaignSelectionError,
    print_campaign_selection_error,
    selected_campaign_dir,
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
    "selected_mode": "dry-run",
    "selected_max_ticks": 0,            # 0 = unset
    "selected_poll_interval": 0,         # 0 = use config default
    "selected_config": "",
    "selected_preset": "",
}


@dataclass
class StartDaemonForegroundMenuOptions(MenuOptions):
    selected_command: str
    selected_mode: str
    selected_max_ticks: int
    selected_poll_interval: int
    selected_config: str
    selected_preset: str


start_daemon_foreground_menu_options = StartDaemonForegroundMenuOptions(
    *START_DAEMON_FOREGROUND_DEFAULTS.values()
)


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
        from ichor.hpc.active_learning.cli import cmd_resume, cmd_start

        try:
            campaign_dir = selected_campaign_dir()
        except CampaignSelectionError as exc:
            print_campaign_selection_error(exc)
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        mode = start_daemon_foreground_menu_options.selected_mode
        ns = argparse.Namespace(
            campaign_dir=str(campaign_dir),
            config=(
                start_daemon_foreground_menu_options.selected_config
                if start_daemon_foreground_menu_options.selected_config
                else None
            ),
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
            preset=(
                start_daemon_foreground_menu_options.selected_preset
                if start_daemon_foreground_menu_options.selected_preset
                else None
            ),
        )
        if start_daemon_foreground_menu_options.selected_command == "resume":
            rc = cmd_resume(ns)
        else:
            rc = cmd_start(ns)
        print("daemon exited with code " + str(rc))
        user_input_free_flow("Press enter to return to the menu: ", "")


start_daemon_foreground_menu_items = [
    FunctionItem("Select command", StartDaemonForegroundFunctions.select_command),
    FunctionItem("Select mode", StartDaemonForegroundFunctions.select_mode),
    FunctionItem("Set max-ticks", StartDaemonForegroundFunctions.select_max_ticks),
    FunctionItem("Set poll-interval", StartDaemonForegroundFunctions.select_poll_interval),
    FunctionItem("Set config override", StartDaemonForegroundFunctions.select_config),
    FunctionItem("Set preset", StartDaemonForegroundFunctions.select_preset),
    FunctionItem("Launch (blocks)", StartDaemonForegroundFunctions.launch),
]


start_daemon_foreground_menu = ConsoleMenu(
    this_menu_options=start_daemon_foreground_menu_options,
    title=START_DAEMON_FOREGROUND_MENU_DESCRIPTION.title,
    subtitle=START_DAEMON_FOREGROUND_MENU_DESCRIPTION.subtitle,
    prologue_text=START_DAEMON_FOREGROUND_MENU_DESCRIPTION.prologue_description_text,
    epilogue_text=START_DAEMON_FOREGROUND_MENU_DESCRIPTION.epilogue_description_text,
    show_exit_option=START_DAEMON_FOREGROUND_MENU_DESCRIPTION.show_exit_option,
)


add_items_to_menu(start_daemon_foreground_menu, start_daemon_foreground_menu_items)
