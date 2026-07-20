"""Journal viewer submenu with persistent visible filters."""
import argparse
from dataclasses import dataclass, field

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
from ichor.cli.useful_functions import user_input_free_flow


JOURNAL_MENU_DESCRIPTION = MenuDescription(
    "Journal Menu",
    subtitle="View the campaign journal (events recorded by the daemon).\n",
)


@dataclass
class JournalMenuOptions(MenuOptions):
    since: str = ""
    event_types: list[str] = field(default_factory=list)
    last_n: int = 50
    output_mode: str = "text"
    verbose: bool = False


journal_menu_options = JournalMenuOptions()


def _get_value(path: str):
    return get_attr_path(journal_menu_options, path)


def _set_value(path: str, value):
    set_attr_path(journal_menu_options, path, value)


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


def _journal_output_kwargs() -> dict:
    mode = str(journal_menu_options.output_mode)
    return {
        "json": mode == "json",
        "raw": mode == "raw",
        "verbose": bool(journal_menu_options.verbose),
        "last_n": journal_menu_options.last_n,
        "list_event_types": False,
    }


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
            **_journal_output_kwargs(),
        )
        ns.last_n = None
        cmd_journal(ns)
        _pause()

    @staticmethod
    def view_matching_events():
        """Print events matching the currently visible filters."""
        from ichor.hpc.active_learning.cli import cmd_journal

        campaign_dir = _guarded_campaign_dir_str()
        if campaign_dir is None:
            _pause()
            return
        ns = argparse.Namespace(
            campaign_dir=campaign_dir,
            since=journal_menu_options.since or None,
            event_type=(
                list(journal_menu_options.event_types)
                if journal_menu_options.event_types
                else None
            ),
            **_journal_output_kwargs(),
        )
        cmd_journal(ns)
        _pause()

    @staticmethod
    def view_events_since():
        """Compatibility alias for the filtered journal view."""
        JournalFunctions.view_matching_events()

    @staticmethod
    def filter_by_event_type():
        """Compatibility alias for the filtered journal view."""
        JournalFunctions.view_matching_events()

    @staticmethod
    def view_last_n_events():
        """Print the last configured N events using current display options."""
        from ichor.hpc.active_learning.cli import cmd_journal

        n = journal_menu_options.last_n
        if n is None or int(n) <= 0:
            print("Invalid N; set last_n to a positive integer.")
            _pause()
            return
        campaign_dir = _guarded_campaign_dir_str()
        if campaign_dir is None:
            _pause()
            return
        ns = argparse.Namespace(
            campaign_dir=campaign_dir,
            since=None,
            event_type=None,
            **_journal_output_kwargs(),
        )
        cmd_journal(ns)
        _pause()

    @staticmethod
    def list_event_types():
        from ichor.hpc.active_learning.cli import cmd_journal

        campaign_dir = _guarded_campaign_dir_str()
        if campaign_dir is None:
            _pause()
            return
        ns = argparse.Namespace(
            campaign_dir=campaign_dir,
            since=None,
            event_type=None,
            last_n=None,
            json=False,
            raw=False,
            verbose=False,
            list_event_types=True,
        )
        cmd_journal(ns)
        _pause()

    @staticmethod
    def clear_filters():
        journal_menu_options.since = ""
        journal_menu_options.event_types = []
        journal_menu_options.last_n = 50
        journal_menu_options.output_mode = "text"
        journal_menu_options.verbose = False


JOURNAL_FIELD_SPECS = [
    spec(
        "since",
        "clearable_str",
        prompt="Since ISO timestamp",
        item_text="Set since timestamp",
    ),
    spec(
        "event_types",
        "csv_list",
        prompt="Event types, comma-separated",
        item_text="Set event types",
    ),
    spec("last_n", "int", prompt="How many last events? ", item_text="Set last_n"),
    spec(
        "output_mode",
        "choice",
        choices=["text", "json", "raw"],
        prompt="Journal output mode: ",
        item_text="Set output mode",
    ),
    spec("verbose", "bool", prompt="Verbose text output? ", item_text="Set verbose"),
]


journal_menu_items = [
    FunctionItem("View matching events", JournalFunctions.view_matching_events),
    FunctionItem("View all events", JournalFunctions.view_all_events),
    FunctionItem("View last N events", JournalFunctions.view_last_n_events),
    FunctionItem("List event types", JournalFunctions.list_event_types),
    FunctionItem("Clear journal filters", JournalFunctions.clear_filters),
]


journal_menu = make_field_menu(
    JOURNAL_MENU_DESCRIPTION.title,
    JOURNAL_MENU_DESCRIPTION.subtitle,
    JOURNAL_FIELD_SPECS,
    _get_value,
    _set_value,
    prologue_text="Current journal filters:\n",
    extra_items=journal_menu_items,
    status_for_path=lambda _path: "scope: menu-only; applies to next journal view",
)
