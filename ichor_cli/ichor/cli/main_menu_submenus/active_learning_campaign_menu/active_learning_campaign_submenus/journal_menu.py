"""Journal viewer submenu.

Each item synthesises an argparse.Namespace and dispatches to
ichor.hpc.active_learning.cli.cmd_journal which prints NDJSON lines to
stdout. Filters (since timestamp / event types / last N) are collected
via user_input prompts before the call.
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
)


JOURNAL_MENU_DESCRIPTION = MenuDescription(
    "Journal Menu",
    subtitle="View the campaign journal (events recorded by the daemon).\n",
)


@dataclass
class JournalMenuOptions(MenuOptions):
    pass


journal_menu_options = JournalMenuOptions()


def _campaign_dir_str():
    return str(selected_campaign_dir())


def _guarded_campaign_dir_str():
    try:
        return _campaign_dir_str()
    except CampaignSelectionError as exc:
        print_campaign_selection_error(exc)
        return None


def _pause():
    user_input_free_flow("Press enter to return to the menu: ", "")


class JournalFunctions:
    @staticmethod
    def view_all_events():
        """Print every event in the campaign journal."""
        from ichor.hpc.active_learning.cli import cmd_journal

        campaign_dir = _guarded_campaign_dir_str()
        if campaign_dir is None:
            _pause()
            return
        ns = argparse.Namespace(
            campaign_dir=campaign_dir,
            since=None,
            event_type=None,
        )
        rc = cmd_journal(ns)
        if rc != 0:
            print("journal returned exit code " + str(rc))
        _pause()

    @staticmethod
    def view_events_since():
        """Filter events by ISO timestamp lower bound."""
        from ichor.hpc.active_learning.cli import cmd_journal

        since = user_input_free_flow(
            "Enter ISO timestamp (e.g. 2026-05-23T00:00:00Z): ", "",
        )
        if not since:
            print("No timestamp given; cancelled.")
            _pause()
            return
        campaign_dir = _guarded_campaign_dir_str()
        if campaign_dir is None:
            _pause()
            return
        ns = argparse.Namespace(
            campaign_dir=campaign_dir,
            since=since,
            event_type=None,
        )
        rc = cmd_journal(ns)
        if rc != 0:
            print("journal returned exit code " + str(rc))
        _pause()

    @staticmethod
    def filter_by_event_type():
        """Filter events by one or more event types (comma-separated)."""
        from ichor.hpc.active_learning.cli import cmd_journal

        raw = user_input_free_flow(
            "Enter event types, comma-separated "
            "(e.g. sbatch,phase_transition,phase_succeeded): ",
            "",
        )
        if not raw:
            print("No event types given; cancelled.")
            _pause()
            return
        campaign_dir = _guarded_campaign_dir_str()
        if campaign_dir is None:
            _pause()
            return
        event_types = [t.strip() for t in raw.split(",") if t.strip()]
        ns = argparse.Namespace(
            campaign_dir=campaign_dir,
            since=None,
            event_type=event_types,
        )
        rc = cmd_journal(ns)
        if rc != 0:
            print("journal returned exit code " + str(rc))
        _pause()

    @staticmethod
    def view_last_n_events():
        """Read every event then print the last N. Implemented in-process
        (no CLI dispatch) so we can slice; cmd_journal does not support a
        tail flag yet."""
        from ichor.hpc.active_learning.daemon.journal import iter_events
        from pathlib import Path
        import json

        n = user_input_int("How many last events? ", 50)
        if n is None or n <= 0:
            print("Invalid N; cancelled.")
            _pause()
            return
        campaign_dir = _guarded_campaign_dir_str()
        if campaign_dir is None:
            _pause()
            return
        journal_path = (
            Path(campaign_dir)
            / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"
        )
        if not journal_path.exists():
            print("No journal at " + str(journal_path))
            _pause()
            return
        events = list(iter_events(journal_path))
        for event in events[-int(n):]:
            print(json.dumps(event, sort_keys=True))
        _pause()


journal_menu = ConsoleMenu(
    this_menu_options=journal_menu_options,
    title=JOURNAL_MENU_DESCRIPTION.title,
    subtitle=JOURNAL_MENU_DESCRIPTION.subtitle,
    prologue_text=JOURNAL_MENU_DESCRIPTION.prologue_description_text,
    epilogue_text=JOURNAL_MENU_DESCRIPTION.epilogue_description_text,
    show_exit_option=JOURNAL_MENU_DESCRIPTION.show_exit_option,
)


journal_menu_items = [
    FunctionItem("View all events", JournalFunctions.view_all_events),
    FunctionItem("View events since ISO timestamp", JournalFunctions.view_events_since),
    FunctionItem("Filter by event type", JournalFunctions.filter_by_event_type),
    FunctionItem("View last N events", JournalFunctions.view_last_n_events),
]


add_items_to_menu(journal_menu, journal_menu_items)
