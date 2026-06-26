"""Daemon control submenu.

Bridges into ichor.hpc.active_learning.cli.cmd_* for read / stop /
reconcile operations (no subprocess needed) and into the start/import submenus
(foreground / background) for the launch path.
"""
from dataclasses import dataclass

from consolemenu.items import FunctionItem, SubmenuItem
from ichor.cli.console_menu import ConsoleMenu, add_items_to_menu
from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
    CampaignSelectionError,
    campaign_dir_ns,
    print_campaign_selection_error,
    selected_campaign_dir,
)
from ichor.cli.main_menu_submenus.active_learning_campaign_menu.protocol_summary import (
    format_saved_sampling_protocol_summary,
)
from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_submenus.daemon_control_submenus import (
    import_trajectory_pool_menu,
    IMPORT_TRAJECTORY_POOL_MENU_DESCRIPTION,
    start_daemon_background_menu,
    START_DAEMON_BACKGROUND_MENU_DESCRIPTION,
    start_daemon_foreground_menu,
    START_DAEMON_FOREGROUND_MENU_DESCRIPTION,
)
from ichor.cli.menu_description import MenuDescription
from ichor.cli.menu_options import MenuOptions
from ichor.cli.useful_functions import user_input_free_flow


DAEMON_CONTROL_MENU_DESCRIPTION = MenuDescription(
    "Daemon Control Menu",
    subtitle="Start / stop / inspect the active-learning daemon for the selected campaign.\n",
)


@dataclass
class DaemonControlMenuOptions(MenuOptions):
    pass


daemon_control_menu_options = DaemonControlMenuOptions()


def _campaign_dir_ns():
    """Build the minimal Namespace shape every cmd_* helper expects."""
    return campaign_dir_ns()


def _guarded_campaign_dir_ns():
    try:
        return _campaign_dir_ns()
    except CampaignSelectionError as exc:
        print_campaign_selection_error(exc)
        return None


class DaemonControlFunctions:
    """Static dispatchers into ichor.hpc.active_learning.cli."""

    @staticmethod
    def show_status():
        """Print state.json as JSON (same output as ichor-al-daemon status)."""
        from ichor.hpc.active_learning.cli import cmd_status

        ns = _guarded_campaign_dir_ns()
        if ns is None:
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        rc = cmd_status(ns)
        if rc != 0:
            print("status returned exit code " + str(rc))
        user_input_free_flow("Press enter to return to the menu: ", "")

    @staticmethod
    def preflight_backends():
        """Probe sbatch / sacct / Gaussian / AIMAll / FEREBUS / ariadne and
        print a human-readable availability report."""
        from ichor.hpc.active_learning.daemon.preflight import (
            check_backends,
            missing_backend_message,
        )

        avail = check_backends()
        if avail.all_present:
            print("All live backends present.")
            print("  sbatch:   " + avail.sbatch_path)
            print("  sacct:    " + avail.sacct_path)
            print("  gaussian: " + avail.gaussian_binary)
            print("  aimall:   " + avail.aimall_path)
            print("  ferebus:  " + avail.ferebus_path)
            print("  ariadne:  importable")
            print("  polus_rs: importable")
            print("  pyferebus: importable")
            print("  bc:       " + avail.bc_path)
        else:
            print(missing_backend_message(avail))
        user_input_free_flow("Press enter to return to the menu: ", "")

    @staticmethod
    def show_sampling_protocol_summary():
        try:
            campaign_dir = selected_campaign_dir()
        except CampaignSelectionError as exc:
            print_campaign_selection_error(exc)
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        print(format_saved_sampling_protocol_summary(campaign_dir))
        user_input_free_flow("Press enter to return to the menu: ", "")

    @staticmethod
    def stop_daemon():
        """Set shutdown_requested=true in state.json so the running daemon
        finishes its current tick and exits cleanly on the next."""
        from ichor.hpc.active_learning.cli import cmd_stop

        ns = _guarded_campaign_dir_ns()
        if ns is None:
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        answer = user_input_free_flow(
            "Cancel active Slurm jobs too? Type YES to confirm: ",
            "",
        )
        ns.cancel_jobs = str(answer).strip() == "YES"
        rc = cmd_stop(ns)
        if rc != 0:
            print("stop returned exit code " + str(rc))
        user_input_free_flow("Press enter to return to the menu: ", "")

    @staticmethod
    def reconcile():
        """Write a proposed state.json to .proposed for user review."""
        from ichor.hpc.active_learning.cli import cmd_reconcile

        ns = _guarded_campaign_dir_ns()
        if ns is None:
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        ns.allow_fresh_init = False
        ns.apply = False
        rc = cmd_reconcile(ns)
        if rc != 0:
            print("reconcile returned exit code " + str(rc))
        user_input_free_flow("Press enter to return to the menu: ", "")

    @staticmethod
    def reconcile_apply():
        """Safely promote a reconcile proposal when config-lock checks pass."""
        from ichor.hpc.active_learning.cli import cmd_reconcile

        ns = _guarded_campaign_dir_ns()
        if ns is None:
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        answer = user_input_free_flow(
            "Apply safe reconcile proposal and clean stale uncommitted staging? Type YES: ",
            "",
        )
        if answer != "YES":
            print("Cancelled.")
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        ns.allow_fresh_init = False
        ns.apply = True
        rc = cmd_reconcile(ns)
        if rc != 0:
            print("reconcile --apply returned exit code " + str(rc))
        user_input_free_flow("Press enter to return to the menu: ", "")

    @staticmethod
    def reconcile_restore_config_from_lock():
        """Write campaign.yaml.proposed from config_lock.json."""
        from ichor.hpc.active_learning.cli import cmd_reconcile

        ns = _guarded_campaign_dir_ns()
        if ns is None:
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        answer = user_input_free_flow(
            "Write campaign.yaml.proposed from config_lock.json? Type YES: ",
            "",
        )
        if answer != "YES":
            print("Cancelled.")
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        ns.allow_fresh_init = False
        ns.apply = False
        ns.restore_config_from_lock = True
        rc = cmd_reconcile(ns)
        if rc != 0:
            print("reconcile --restore-config-from-lock returned exit code " + str(rc))
        user_input_free_flow("Press enter to return to the menu: ", "")

    @staticmethod
    def reconcile_allow_fresh_init():
        """Dangerous reconcile mode matching --allow-fresh-init."""
        from ichor.hpc.active_learning.cli import cmd_reconcile

        ns = _guarded_campaign_dir_ns()
        if ns is None:
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        answer = user_input_free_flow(
            "Allow fresh INIT over non-empty campaign data? This is dangerous. Type YES: ",
            "",
        )
        if answer != "YES":
            print("Cancelled.")
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        ns.allow_fresh_init = True
        ns.apply = False
        rc = cmd_reconcile(ns)
        if rc != 0:
            print("reconcile returned exit code " + str(rc))
        user_input_free_flow("Press enter to return to the menu: ", "")

    @staticmethod
    def import_trajectory_pool():
        """Compatibility dispatcher for the init/import option submenu."""
        from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_submenus.daemon_control_submenus.import_trajectory_pool_submenu import (
            ImportTrajectoryPoolFunctions,
        )

        ImportTrajectoryPoolFunctions.run_import()


daemon_control_menu = ConsoleMenu(
    this_menu_options=daemon_control_menu_options,
    title=DAEMON_CONTROL_MENU_DESCRIPTION.title,
    subtitle=DAEMON_CONTROL_MENU_DESCRIPTION.subtitle,
    prologue_text=DAEMON_CONTROL_MENU_DESCRIPTION.prologue_description_text,
    epilogue_text=DAEMON_CONTROL_MENU_DESCRIPTION.epilogue_description_text,
    show_exit_option=DAEMON_CONTROL_MENU_DESCRIPTION.show_exit_option,
)


daemon_control_menu_items = [
    FunctionItem("Show status", DaemonControlFunctions.show_status),
    FunctionItem(
        "Show sampling protocol summary",
        DaemonControlFunctions.show_sampling_protocol_summary,
    ),
    FunctionItem("Preflight backends", DaemonControlFunctions.preflight_backends),
    SubmenuItem(
        IMPORT_TRAJECTORY_POOL_MENU_DESCRIPTION.title,
        import_trajectory_pool_menu,
        daemon_control_menu,
    ),
    SubmenuItem(
        START_DAEMON_FOREGROUND_MENU_DESCRIPTION.title,
        start_daemon_foreground_menu,
        daemon_control_menu,
    ),
    SubmenuItem(
        START_DAEMON_BACKGROUND_MENU_DESCRIPTION.title,
        start_daemon_background_menu,
        daemon_control_menu,
    ),
    FunctionItem("Stop daemon", DaemonControlFunctions.stop_daemon),
    FunctionItem("Reconcile state", DaemonControlFunctions.reconcile),
    FunctionItem(
        "Reconcile and apply safe proposal",
        DaemonControlFunctions.reconcile_apply,
    ),
    FunctionItem(
        "Restore campaign.yaml proposal from config lock",
        DaemonControlFunctions.reconcile_restore_config_from_lock,
    ),
    FunctionItem(
        "Reconcile state with --allow-fresh-init",
        DaemonControlFunctions.reconcile_allow_fresh_init,
    ),
]


add_items_to_menu(daemon_control_menu, daemon_control_menu_items)
