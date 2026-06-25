"""Top-level Active Learning Campaign menu.

Surfaces the `ichor-al-daemon` workflow inside `ichor-cli`. All daemon
operations are scoped to the currently selected campaign directory (held
in :mod:`ichor.cli.global_menu_variables`), so multiple concurrent
campaigns are supported by switching the selection.

Submenus:
    - Edit campaign config  -- field-by-field campaign.yaml editor
    - Daemon control        -- start / stop / status / reconcile
    - Journal               -- view + filter daemon journal events

The standalone ichor-al-daemon console script remains fully operational;
this menu just dispatches into the same ichor.hpc.active_learning.cli
functions and Daemon.run() entry point.
"""
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Union

import ichor.cli.global_menu_variables
from consolemenu.items import FunctionItem, SubmenuItem
from ichor.cli.console_menu import ConsoleMenu, add_items_to_menu
from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_submenus import (
    daemon_control_menu,
    DAEMON_CONTROL_MENU_DESCRIPTION,
    edit_campaign_config_menu,
    EDIT_CAMPAIGN_CONFIG_MENU_DESCRIPTION,
    journal_menu,
    JOURNAL_MENU_DESCRIPTION,
)
from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
    set_selected_campaign_dir,
)
from ichor.cli.menu_description import MenuDescription
from ichor.cli.menu_options import MenuOptions
from ichor.cli.useful_functions import user_input_path


ACTIVE_LEARNING_CAMPAIGN_MENU_DESCRIPTION = MenuDescription(
    "Active Learning Campaign Menu",
    subtitle=(
        "Drive an ichor-al-daemon active learning campaign without leaving the menu. "
        "All daemon operations target the currently selected campaign directory; switch "
        "the selection to manage concurrent campaigns.\n"
    ),
)


@dataclass
class ActiveLearningCampaignMenuOptions(MenuOptions):
    selected_active_learning_campaign_directory_path: Path = (
        ichor.cli.global_menu_variables.SELECTED_ACTIVE_LEARNING_CAMPAIGN_DIRECTORY_PATH
    )
    # Refreshed on every prologue render from <campaign_dir>/.DATA/ACTIVE_LEARNING/state.json.
    # Default placeholder is shown until the user selects a campaign directory.
    daemon_status_summary: str = "(no campaign selected)"

    def check_selected_active_learning_campaign_directory_path(self) -> Union[str, None]:
        """Validate the campaign directory before any daemon op is dispatched.

        Returns a warning string when the path is unset / does not exist /
        is not a directory; the prologue surfaces it in red so the operator
        cannot accidentally drive a daemon against a bad path.
        """
        path = Path(self.selected_active_learning_campaign_directory_path)
        if str(path) == "" or path == Path("").absolute():
            # The global is initialised to Path("").absolute() == cwd; warn only
            # if the cwd does not look like a campaign dir (no .DATA subtree yet).
            if not (path / ".DATA" / "ACTIVE_LEARNING").exists():
                return (
                    "No active learning campaign directory selected yet. "
                    "Use 'Select / switch campaign directory' to choose one."
                )
            return None
        if not path.exists():
            return f"Campaign directory {path} does not exist."
        if not path.is_dir():
            return f"Campaign path {path} is not a directory."

    def __call__(self):
        """Refresh daemon_status_summary from state.json on every prologue
        render, then delegate to the base class formatter.

        Every failure mode is swallowed and surfaced as a human-readable
        placeholder so the menu prologue can NEVER crash on a corrupt
        state.json -- the operator falls back to 'Reconcile state' from the
        daemon control submenu instead.
        """
        self._refresh_daemon_status_summary()
        return super().__call__()

    def _refresh_daemon_status_summary(self) -> None:
        import json

        path = Path(self.selected_active_learning_campaign_directory_path)
        # Treat the initial cwd default as "no campaign" unless it actually
        # has the expected .DATA/ACTIVE_LEARNING subtree.
        state_path = path / ".DATA" / "ACTIVE_LEARNING" / "state.json"
        if not state_path.is_file():
            self.daemon_status_summary = "(no state.json yet)"
            return
        try:
            from ichor.hpc.active_learning.daemon.state import (
                StateSchemaError,
                read_state,
            )
            state = read_state(state_path)
        except (StateSchemaError, json.JSONDecodeError) as exc:
            self.daemon_status_summary = (
                "(state.json unreadable; run Reconcile -- "
                + type(exc).__name__ + ")"
            )
            return
        except OSError as exc:
            self.daemon_status_summary = (
                "(state.json read error: " + type(exc).__name__ + ")"
            )
            return
        except Exception as exc:  # pragma: no cover -- last-line defence
            self.daemon_status_summary = (
                "(status read failed: " + type(exc).__name__ + ")"
            )
            return
        phase = state.phase.value if hasattr(state.phase, "value") else str(state.phase)
        pending = sum(1 for v in state.pending_jobs.values() if v)
        self.daemon_status_summary = (
            phase + " (iter " + str(state.iteration) + "/"
            + str(state.max_iterations) + ", pending jobs: " + str(pending) + ")"
        )


active_learning_campaign_menu_options = ActiveLearningCampaignMenuOptions()


class ActiveLearningCampaignFunctions:
    """Functions invoked by top-level menu items."""

    @staticmethod
    def select_campaign_directory():
        """Prompt for a campaign directory and update both the module-level
        global and the menu options instance so the new selection shows up
        in the prologue immediately."""
        new_path = user_input_path(
            "Enter campaign directory path: ",
            default_path=str(
                ichor.cli.global_menu_variables.SELECTED_ACTIVE_LEARNING_CAMPAIGN_DIRECTORY_PATH
            ),
        )
        cfg_menu = importlib.import_module(
            "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
            "active_learning_campaign_submenus.edit_campaign_config_menu"
        )
        candidate = Path(new_path).expanduser().absolute()
        if not cfg_menu.load_config_for_campaign_dir(
            candidate,
            quiet=False,
            prompt_if_dirty=True,
        ):
            return
        selected = set_selected_campaign_dir(candidate)
        active_learning_campaign_menu_options.selected_active_learning_campaign_directory_path = (
            selected
        )


active_learning_campaign_menu = ConsoleMenu(
    this_menu_options=active_learning_campaign_menu_options,
    title=ACTIVE_LEARNING_CAMPAIGN_MENU_DESCRIPTION.title,
    subtitle=ACTIVE_LEARNING_CAMPAIGN_MENU_DESCRIPTION.subtitle,
    prologue_text=ACTIVE_LEARNING_CAMPAIGN_MENU_DESCRIPTION.prologue_description_text,
    epilogue_text=ACTIVE_LEARNING_CAMPAIGN_MENU_DESCRIPTION.epilogue_description_text,
    show_exit_option=ACTIVE_LEARNING_CAMPAIGN_MENU_DESCRIPTION.show_exit_option,
)


active_learning_campaign_menu_items = [
    FunctionItem(
        "Select / switch campaign directory",
        ActiveLearningCampaignFunctions.select_campaign_directory,
    ),
    SubmenuItem(
        EDIT_CAMPAIGN_CONFIG_MENU_DESCRIPTION.title,
        edit_campaign_config_menu,
        active_learning_campaign_menu,
    ),
    SubmenuItem(
        DAEMON_CONTROL_MENU_DESCRIPTION.title,
        daemon_control_menu,
        active_learning_campaign_menu,
    ),
    SubmenuItem(
        JOURNAL_MENU_DESCRIPTION.title,
        journal_menu,
        active_learning_campaign_menu,
    ),
]


add_items_to_menu(active_learning_campaign_menu, active_learning_campaign_menu_items)
