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
    status_json: bool = False
    status_verbose: bool = False
    preflight_json: bool = False
    reconcile_json: bool = False
    reconcile_verbose: bool = False


daemon_control_menu_options = DaemonControlMenuOptions()


def _campaign_dir_ns():
    """Build the minimal Namespace shape every cmd_* helper expects."""
    return campaign_dir_ns()


def _guarded_campaign_dir_ns():
    try:
        ns = _campaign_dir_ns()
        campaign = selected_campaign_dir()
        if not (campaign / "campaign.yaml").is_file():
            raise CampaignSelectionError(
                "Selected campaign has no campaign.yaml. Save or initialise the campaign first: "
                + str(campaign)
            )
        return ns
    except CampaignSelectionError as exc:
        print_campaign_selection_error(exc)
        return None


class DaemonControlFunctions:
    """Static dispatchers into ichor.hpc.active_learning.cli."""

    @staticmethod
    def toggle_status_json():
        daemon_control_menu_options.status_json = (
            not bool(daemon_control_menu_options.status_json)
        )

    @staticmethod
    def toggle_status_verbose():
        daemon_control_menu_options.status_verbose = (
            not bool(daemon_control_menu_options.status_verbose)
        )

    @staticmethod
    def toggle_preflight_json():
        daemon_control_menu_options.preflight_json = not bool(
            daemon_control_menu_options.preflight_json
        )

    @staticmethod
    def toggle_reconcile_json():
        daemon_control_menu_options.reconcile_json = not bool(
            daemon_control_menu_options.reconcile_json
        )

    @staticmethod
    def toggle_reconcile_verbose():
        daemon_control_menu_options.reconcile_verbose = not bool(
            daemon_control_menu_options.reconcile_verbose
        )

    @staticmethod
    def show_status():
        """Print state.json as JSON (same output as ichor-al-daemon status)."""
        from ichor.hpc.active_learning.cli import cmd_status

        ns = _guarded_campaign_dir_ns()
        if ns is None:
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        ns.json = bool(daemon_control_menu_options.status_json)
        ns.verbose = bool(daemon_control_menu_options.status_verbose)
        rc = cmd_status(ns)
        if rc != 0:
            print("status returned exit code " + str(rc))
        user_input_free_flow("Press enter to return to the menu: ", "")

    @staticmethod
    def preflight_backends():
        """Run the same campaign-aware readiness gate as live start."""
        from ichor.hpc.active_learning.cli import cmd_preflight

        ns = _guarded_campaign_dir_ns()
        if ns is None:
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        ns.json = bool(daemon_control_menu_options.preflight_json)
        ns.verbose = True
        ns.submit_environment_smoke = False
        rc = cmd_preflight(ns)
        if rc != 0:
            print("campaign preflight returned exit code " + str(rc))
        user_input_free_flow("Press enter to return to the menu: ", "")

    @staticmethod
    def config_check():
        from ichor.hpc.active_learning.cli import cmd_config_check

        ns = _guarded_campaign_dir_ns()
        if ns is None:
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        try:
            rc = cmd_config_check(ns)
        except Exception as exc:
            print("config-check failed: " + type(exc).__name__ + ": " + str(exc))
            rc = 2
        if rc != 0:
            print("config-check returned exit code " + str(rc))
        user_input_free_flow("Press enter to return to the menu: ", "")

    @staticmethod
    def submitted_environment_smoke():
        """Run ordinary preflight, then submit the opt-in compute-node smoke."""
        from ichor.hpc.active_learning.cli import cmd_preflight

        ns = _guarded_campaign_dir_ns()
        if ns is None:
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        answer = user_input_free_flow(
            "Submit one five-minute, one-core environment smoke job? Type YES: ",
            "",
        )
        if str(answer).strip() != "YES":
            print("Cancelled.")
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        ns.json = False
        ns.verbose = True
        ns.submit_environment_smoke = True
        rc = cmd_preflight(ns)
        if rc != 0:
            print("submitted environment smoke returned exit code " + str(rc))
        user_input_free_flow("Press enter to return to the menu: ", "")

    @staticmethod
    def show_sampling_protocol_summary():
        try:
            campaign_dir = selected_campaign_dir()
        except CampaignSelectionError as exc:
            print_campaign_selection_error(exc)
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        print("Summary source: saved campaign.yaml.")
        print(format_saved_sampling_protocol_summary(campaign_dir))
        user_input_free_flow("Press enter to return to the menu: ", "")

    @staticmethod
    def show_saved_config_editability_windows():
        try:
            campaign_dir = selected_campaign_dir()
        except CampaignSelectionError as exc:
            print_campaign_selection_error(exc)
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        editor = __import__(
            "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
            "active_learning_campaign_submenus.edit_campaign_config_menu",
            fromlist=["_iter_campaign_field_specs"],
        )
        try:
            from ichor.cli.main_menu_submenus.active_learning_campaign_menu.field_menu import (
                format_field_value,
                get_attr_path,
            )
            from ichor.hpc.active_learning.config import CampaignConfig
            from ichor.hpc.active_learning.daemon import config_lock as lock_mod

            config = CampaignConfig.from_yaml(campaign_dir / "campaign.yaml")
        except Exception as exc:
            print(
                "Could not load saved campaign.yaml from "
                + str(campaign_dir)
                + ": "
                + str(exc)
            )
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        review = editor.saved_config_lock_review()
        print("Config editability windows")
        print("Source: saved campaign.yaml.")
        for path, _spec_obj in editor._iter_campaign_field_specs():
            status = None
            if review is not None:
                for change in review.blocked_changes:
                    if change.path == path:
                        status = "blocked: " + change.reason
                        break
                if status is None:
                    for change in review.allowed_changes:
                        if change.path == path:
                            status = "allowed: " + change.reason
                            break
            if status is None:
                window = lock_mod.describe_field_editability(path)
                status = (
                    "read-only: " + window
                    if path == "schema_version"
                    else "window: " + window
                )
            print(
                "  "
                + path
                + ": "
                + format_field_value(get_attr_path(config, path))
                + " ["
                + status
                + "]"
            )
        user_input_free_flow("Press enter to return to the menu: ", "")

    @staticmethod
    def show_recovery_dashboard():
        from ichor.hpc.active_learning.cli import format_recovery_dashboard

        try:
            campaign_dir = selected_campaign_dir()
        except CampaignSelectionError as exc:
            print_campaign_selection_error(exc)
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        print(format_recovery_dashboard(campaign_dir))
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
    def _set_reconcile_defaults(ns):
        ns.allow_fresh_init = bool(getattr(ns, "allow_fresh_init", False))
        ns.apply = bool(getattr(ns, "apply", False))
        ns.archive_staging = bool(getattr(ns, "archive_staging", False))
        ns.restore_config_from_lock = bool(
            getattr(ns, "restore_config_from_lock", False)
        )
        ns.restore_config_lock_history = bool(
            getattr(ns, "restore_config_lock_history", False)
        )
        ns.force_resubmit_array_tasks = bool(
            getattr(ns, "force_resubmit_array_tasks", False)
        )
        ns.archive_existing_array_task_outputs = bool(
            getattr(ns, "archive_existing_array_task_outputs", False)
        )
        ns.retrain_ferebus = bool(getattr(ns, "retrain_ferebus", False))
        ns.json = bool(getattr(ns, "json", False))
        ns.verbose = bool(getattr(ns, "verbose", False))
        return ns

    @staticmethod
    def reconcile():
        """Write a proposed state.json to .proposed for user review."""
        from ichor.hpc.active_learning.cli import cmd_reconcile

        ns = _guarded_campaign_dir_ns()
        if ns is None:
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        ns = DaemonControlFunctions._set_reconcile_defaults(ns)
        ns.allow_fresh_init = False
        ns.apply = False
        ns.json = bool(daemon_control_menu_options.reconcile_json)
        ns.verbose = bool(daemon_control_menu_options.reconcile_verbose)
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
        ns = DaemonControlFunctions._set_reconcile_defaults(ns)
        ns.allow_fresh_init = False
        ns.apply = True
        ns.archive_staging = False
        ns.restore_config_from_lock = False
        rc = cmd_reconcile(ns)
        if rc != 0:
            print("reconcile --apply returned exit code " + str(rc))
        user_input_free_flow("Press enter to return to the menu: ", "")

    @staticmethod
    def reconcile_archive_stale_staging():
        """Archive stale .DATA/STAGING through guarded reconcile --apply."""
        from ichor.hpc.active_learning.cli import cmd_reconcile

        ns = _guarded_campaign_dir_ns()
        if ns is None:
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        answer = user_input_free_flow(
            "Archive stale .DATA/STAGING and apply safe reconcile proposal? Type YES: ",
            "",
        )
        if answer != "YES":
            print("Cancelled.")
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        ns = DaemonControlFunctions._set_reconcile_defaults(ns)
        ns.allow_fresh_init = False
        ns.apply = True
        ns.archive_staging = True
        ns.restore_config_from_lock = False
        rc = cmd_reconcile(ns)
        if rc != 0:
            print("reconcile --archive-staging --apply returned exit code " + str(rc))
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
        ns = DaemonControlFunctions._set_reconcile_defaults(ns)
        ns.allow_fresh_init = False
        ns.apply = False
        ns.restore_config_from_lock = True
        rc = cmd_reconcile(ns)
        if rc != 0:
            print("reconcile --restore-config-from-lock returned exit code " + str(rc))
        user_input_free_flow("Press enter to return to the menu: ", "")

    @staticmethod
    def reconcile_restore_config_lock_history():
        """Restore a missing current lock from verified immutable history."""
        from ichor.hpc.active_learning.cli import cmd_reconcile

        ns = _guarded_campaign_dir_ns()
        if ns is None:
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        answer = user_input_free_flow(
            "Restore missing config_lock.json from verified history? Type YES: ",
            "",
        )
        if answer != "YES":
            print("Cancelled.")
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        ns = DaemonControlFunctions._set_reconcile_defaults(ns)
        ns.apply = False
        ns.restore_config_from_lock = False
        ns.restore_config_lock_history = True
        rc = cmd_reconcile(ns)
        if rc != 0:
            print("reconcile --restore-config-lock-history returned exit code " + str(rc))
        user_input_free_flow("Press enter to return to the menu: ", "")

    @staticmethod
    def reconcile_allow_fresh_init():
        """Reconcile mode matching identity-preserving --allow-fresh-init."""
        from ichor.hpc.active_learning.cli import cmd_reconcile

        ns = _guarded_campaign_dir_ns()
        if ns is None:
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        answer = user_input_free_flow(
            "Allow fresh INIT only if existing campaign identity remains trusted? "
            "This cannot mint a UID for a non-empty campaign. Type YES: ",
            "",
        )
        if answer != "YES":
            print("Cancelled.")
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        ns = DaemonControlFunctions._set_reconcile_defaults(ns)
        ns.allow_fresh_init = True
        ns.apply = False
        rc = cmd_reconcile(ns)
        if rc != 0:
            print("reconcile returned exit code " + str(rc))
        user_input_free_flow("Press enter to return to the menu: ", "")

    @staticmethod
    def reconcile_force_resubmit_array():
        """Force a full rerun of the current supported array phase."""
        from ichor.hpc.active_learning.cli import cmd_reconcile

        ns = _guarded_campaign_dir_ns()
        if ns is None:
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        answer = user_input_free_flow(
            "Archive existing task outputs and force-resubmit the current array? Type YES: ",
            "",
        )
        if answer != "YES":
            print("Cancelled.")
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        ns = DaemonControlFunctions._set_reconcile_defaults(ns)
        ns.allow_fresh_init = False
        ns.apply = True
        ns.force_resubmit_array_tasks = True
        ns.archive_existing_array_task_outputs = True
        rc = cmd_reconcile(ns)
        if rc != 0:
            print("reconcile --force-resubmit-array-tasks returned exit code " + str(rc))
        user_input_free_flow("Press enter to return to the menu: ", "")

    @staticmethod
    def reconcile_retrain_ferebus():
        """Archive complete rejected FEREBUS output and request a new fit."""
        from ichor.hpc.active_learning.cli import cmd_reconcile

        ns = _guarded_campaign_dir_ns()
        if ns is None:
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        answer = user_input_free_flow(
            "Archive complete FEREBUS output and retrain from scratch? Type YES: ",
            "",
        )
        if answer != "YES":
            print("Cancelled.")
            user_input_free_flow("Press enter to return to the menu: ", "")
            return
        ns = DaemonControlFunctions._set_reconcile_defaults(ns)
        ns.apply = True
        ns.retrain_ferebus = True
        rc = cmd_reconcile(ns)
        if rc != 0:
            print("reconcile --retrain-ferebus returned exit code " + str(rc))
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
    FunctionItem("Toggle status JSON output", DaemonControlFunctions.toggle_status_json),
    FunctionItem("Toggle status verbose output", DaemonControlFunctions.toggle_status_verbose),
    FunctionItem("Toggle preflight JSON output", DaemonControlFunctions.toggle_preflight_json),
    FunctionItem("Toggle reconcile JSON output", DaemonControlFunctions.toggle_reconcile_json),
    FunctionItem("Toggle reconcile verbose output", DaemonControlFunctions.toggle_reconcile_verbose),
    FunctionItem("Show status", DaemonControlFunctions.show_status),
    FunctionItem("Validate campaign config", DaemonControlFunctions.config_check),
    FunctionItem(
        "Show sampling protocol summary",
        DaemonControlFunctions.show_sampling_protocol_summary,
    ),
    FunctionItem(
        "Show config editability windows",
        DaemonControlFunctions.show_saved_config_editability_windows,
    ),
    FunctionItem(
        "Recovery dashboard",
        DaemonControlFunctions.show_recovery_dashboard,
    ),
    FunctionItem("Campaign live preflight", DaemonControlFunctions.preflight_backends),
    FunctionItem(
        "Submit compute-node environment smoke",
        DaemonControlFunctions.submitted_environment_smoke,
    ),
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
        "Reconcile --apply",
        DaemonControlFunctions.reconcile_apply,
    ),
    FunctionItem(
        "Reconcile --archive-staging --apply",
        DaemonControlFunctions.reconcile_archive_stale_staging,
    ),
    FunctionItem(
        "Reconcile force-resubmit current array",
        DaemonControlFunctions.reconcile_force_resubmit_array,
    ),
    FunctionItem(
        "Reconcile and retrain FEREBUS",
        DaemonControlFunctions.reconcile_retrain_ferebus,
    ),
    FunctionItem(
        "Restore campaign.yaml proposal from config lock",
        DaemonControlFunctions.reconcile_restore_config_from_lock,
    ),
    FunctionItem(
        "Restore missing config lock from verified history",
        DaemonControlFunctions.reconcile_restore_config_lock_history,
    ),
    FunctionItem(
        "Reconcile state with --allow-fresh-init",
        DaemonControlFunctions.reconcile_allow_fresh_init,
    ),
]


add_items_to_menu(daemon_control_menu, daemon_control_menu_items)
