"""Smoke tests for the Active Learning Campaign menu wiring.

These tests do not exercise the interactive prompts or invoke the daemon;
they verify that:

  * each new submenu module imports without raising,
  * each submenu's ConsoleMenu object has the expected items wired up,
  * the top-level menu is registered as the 8th item of the main menu,
  * the held CampaignConfig is shareable across the top-level config
    editor and the nested ARIADNE block submenu.

If any of these break, the menu wiring is the regression rather than the
underlying daemon / config code (which has its own M1-M8 test suite).
"""
import warnings


def test_top_level_menu_imports_and_has_expected_items():
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu import (
        active_learning_campaign_menu,
        ACTIVE_LEARNING_CAMPAIGN_MENU_DESCRIPTION,
    )
    assert ACTIVE_LEARNING_CAMPAIGN_MENU_DESCRIPTION.title == "Active Learning Campaign Menu"
    texts = [it.text for it in active_learning_campaign_menu.items]
    assert "Select / switch campaign directory" in texts
    assert "Edit Campaign Config Menu" in texts
    assert "Daemon Control Menu" in texts
    assert "Journal Menu" in texts


def test_daemon_control_menu_items():
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_submenus.daemon_control_menu import (
        daemon_control_menu,
    )
    texts = [it.text for it in daemon_control_menu.items]
    for expected in (
        "Show status",
        "Preflight backends",
        "Start/Resume Daemon (Foreground)",
        "Start/Resume Daemon (Background)",
        "Stop daemon",
        "Reconcile state",
        "Reconcile state with --allow-fresh-init",
    ):
        assert expected in texts, "missing item: " + expected


def test_edit_campaign_config_menu_items():
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_submenus.edit_campaign_config_menu import (
        edit_campaign_config_menu,
        get_campaign_config,
    )
    texts = [it.text for it in edit_campaign_config_menu.items]
    for expected in (
        "Show current config",
        "Load from disk",
        "Reset to defaults",
        "Validate current config",
        "Edit campaign identity",
        "Edit trajectory_pool",
        "Edit iteration control",
        "Edit resources",
        "Edit Gaussian block",
        "Edit FEREBUS block",
        "Edit acquisition core",
        "Edit ARIADNE Block",
        # M15 F15: blocks added in this milestone.
        "Edit acquisition.barrier",
        "Edit acquisition.stencils",
        "Edit acquisition.references",
        "Edit stop",
        "Edit outlier_filter",
        "Save to disk",
    ):
        assert expected in texts, "missing item: " + expected
    cfg = get_campaign_config()
    assert cfg.max_iterations >= 1
    assert hasattr(cfg, "ariadne")


def test_ariadne_submenu_shares_block_with_parent():
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_submenus.edit_campaign_config_menu import (
        get_campaign_config,
    )
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_submenus.edit_campaign_config_submenus.edit_ariadne_block_submenu import (
        _get_block,
        edit_ariadne_block_menu,
    )
    parent_block = get_campaign_config().ariadne
    submenu_block = _get_block()
    assert submenu_block is parent_block, (
        "ARIADNE submenu must mutate the same block instance as the parent config"
    )
    texts = [it.text for it in edit_ariadne_block_menu.items]
    assert "Edit optimiser" in texts
    assert "Edit max iterations" in texts


def test_journal_menu_items():
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_submenus.journal_menu import (
        journal_menu,
    )
    texts = [it.text for it in journal_menu.items]
    for expected in (
        "View all events",
        "View events since ISO timestamp",
        "Filter by event type",
        "View last N events",
    ):
        assert expected in texts, "missing item: " + expected


def test_top_level_menu_registered_in_main_menu():
    from pathlib import Path

    main_menu_source = (
        Path(__file__).parents[1] / "ichor" / "cli" / "main_menu.py"
    ).read_text(encoding="utf-8")
    assert "ACTIVE_LEARNING_CAMPAIGN_MENU_DESCRIPTION.title" in main_menu_source
    assert "active_learning_campaign_menu" in main_menu_source


def test_launch_helpers_builds_argv():
    from pathlib import Path

    from ichor.cli.useful_functions.launch_helpers import (
        build_daemon_argv,
        DetachedLaunchResult,
        MENU_LAUNCHED_PID_FILENAME,
        MENU_LAUNCHED_LOG_FILENAME,
    )
    argv = build_daemon_argv(
        Path("/some/campaign"),
        mode="dry-run",
        poll_interval=15,
        max_ticks=5,
    )
    assert argv[1] == "-m"
    assert argv[2] == "ichor.hpc.active_learning.cli"
    assert "start" in argv
    assert "--dry-run" in argv
    assert "--poll-interval" in argv
    assert "--max-ticks" in argv
    assert DetachedLaunchResult.__name__ == "DetachedLaunchResult"
    assert MENU_LAUNCHED_PID_FILENAME == "menu_launched.pid"
    assert MENU_LAUNCHED_LOG_FILENAME == "daemon.menu_launched.out"


def test_launch_helpers_builds_resume_argv_with_preset_and_config(tmp_path):
    from ichor.cli.useful_functions.launch_helpers import build_daemon_argv

    cfg = tmp_path / "override.yaml"
    argv = build_daemon_argv(
        tmp_path,
        command="resume",
        mode="live",
        config=cfg,
        preset="csf4",
    )
    assert argv[3] == "resume"
    assert "--live" in argv
    assert argv[argv.index("--config") + 1] == str(cfg)
    assert argv[argv.index("--preset") + 1] == "csf4"


def test_launch_helpers_rejects_unknown_mode():
    import pytest

    from ichor.cli.useful_functions.launch_helpers import build_daemon_argv
    from pathlib import Path

    with pytest.raises(ValueError):
        build_daemon_argv(Path("."), mode="bogus")

    with pytest.raises(ValueError):
        build_daemon_argv(Path("."), command="bogus", mode="dry-run")


def test_campaign_dir_global_added():
    from ichor.cli import global_menu_variables

    assert hasattr(
        global_menu_variables,
        "SELECTED_ACTIVE_LEARNING_CAMPAIGN_DIRECTORY_PATH",
    )


# --- Tests for the live phase display in the top-level prologue (N10) -----


def _make_state_json(target_path, *, phase="SEED_SELECT", iteration=3, max_iterations=50, pending_jobs=None):
    """Helper: write a valid state.json at target_path using the canonical writer
    so the file matches what the daemon would produce."""
    from ichor.hpc.active_learning.daemon.state import (
        CampaignState,
        CampaignPhase,
        write_state,
    )
    state = CampaignState(
        iteration=iteration,
        max_iterations=max_iterations,
        phase=CampaignPhase(phase),
        pending_jobs=pending_jobs or {},
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    write_state(target_path, state)


def test_prologue_summary_when_no_state_json(tmp_path):
    """Selecting a campaign dir with no state.json -> placeholder string."""
    from pathlib import Path

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_menu import (
        active_learning_campaign_menu_options,
    )

    campaign = tmp_path / "empty_campaign"
    campaign.mkdir()
    original = active_learning_campaign_menu_options.selected_active_learning_campaign_directory_path
    try:
        active_learning_campaign_menu_options.selected_active_learning_campaign_directory_path = campaign
        active_learning_campaign_menu_options._refresh_daemon_status_summary()
        assert active_learning_campaign_menu_options.daemon_status_summary == "(no state.json yet)"
    finally:
        active_learning_campaign_menu_options.selected_active_learning_campaign_directory_path = original


def test_prologue_summary_when_state_json_valid(tmp_path):
    """Valid state.json -> summary reflects phase + iteration + pending count."""
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_menu import (
        active_learning_campaign_menu_options,
    )

    campaign = tmp_path / "live_campaign"
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    _make_state_json(
        state_path,
        phase="ARIADNE_ARRAY",
        iteration=7,
        max_iterations=50,
        pending_jobs={"ARIADNE_ARRAY": "12345"},
    )

    original = active_learning_campaign_menu_options.selected_active_learning_campaign_directory_path
    try:
        active_learning_campaign_menu_options.selected_active_learning_campaign_directory_path = campaign
        active_learning_campaign_menu_options._refresh_daemon_status_summary()
        summary = active_learning_campaign_menu_options.daemon_status_summary
        assert "ARIADNE_ARRAY" in summary
        assert "iter 7/50" in summary
        assert "pending jobs: 1" in summary
    finally:
        active_learning_campaign_menu_options.selected_active_learning_campaign_directory_path = original


def test_prologue_summary_when_state_json_corrupt(tmp_path):
    """Corrupt state.json -> placeholder string; menu must NOT crash."""
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_menu import (
        active_learning_campaign_menu_options,
    )

    campaign = tmp_path / "corrupt_campaign"
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{not valid json at all", encoding="utf-8")

    original = active_learning_campaign_menu_options.selected_active_learning_campaign_directory_path
    try:
        active_learning_campaign_menu_options.selected_active_learning_campaign_directory_path = campaign
        # Must NOT raise even though state.json is unparseable.
        active_learning_campaign_menu_options._refresh_daemon_status_summary()
        summary = active_learning_campaign_menu_options.daemon_status_summary
        assert "unreadable" in summary or "read error" in summary or "read failed" in summary
        assert "Reconcile" in summary or "read error" in summary or "read failed" in summary
    finally:
        active_learning_campaign_menu_options.selected_active_learning_campaign_directory_path = original


def test_prologue_summary_when_state_json_schema_drift(tmp_path):
    """state.json that parses as JSON but fails schema check -> placeholder."""
    import json
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_menu import (
        active_learning_campaign_menu_options,
    )

    campaign = tmp_path / "drift_campaign"
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"schema_version": 9999, "phase": "INIT"}),
        encoding="utf-8",
    )

    original = active_learning_campaign_menu_options.selected_active_learning_campaign_directory_path
    try:
        active_learning_campaign_menu_options.selected_active_learning_campaign_directory_path = campaign
        active_learning_campaign_menu_options._refresh_daemon_status_summary()
        summary = active_learning_campaign_menu_options.daemon_status_summary
        assert "unreadable" in summary
        assert "Reconcile" in summary
    finally:
        active_learning_campaign_menu_options.selected_active_learning_campaign_directory_path = original


def test_prologue_call_refreshes_summary(tmp_path):
    """Calling MenuOptions() must refresh the summary before stringifying."""
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_menu import (
        active_learning_campaign_menu_options,
    )

    campaign = tmp_path / "call_campaign"
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    _make_state_json(state_path, phase="FEREBUS", iteration=11, max_iterations=20)

    original = active_learning_campaign_menu_options.selected_active_learning_campaign_directory_path
    try:
        active_learning_campaign_menu_options.selected_active_learning_campaign_directory_path = campaign
        active_learning_campaign_menu_options.daemon_status_summary = "(should be overwritten)"
        # Invoke the MenuOptions __call__ as the ConsoleMenu prologue would.
        rendered = active_learning_campaign_menu_options()
        # The rendered text contains the refreshed summary, NOT the old placeholder.
        assert "FEREBUS" in rendered
        assert "iter 11/20" in rendered
        assert "should be overwritten" not in rendered
    finally:
        active_learning_campaign_menu_options.selected_active_learning_campaign_directory_path = original


def test_campaign_context_refuses_implicit_cwd(monkeypatch):
    import pytest
    from pathlib import Path
    import ichor.cli.global_menu_variables as globals_
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu import campaign_context

    monkeypatch.setattr(campaign_context, "_explicit_selection", False)
    monkeypatch.setattr(
        globals_,
        "SELECTED_ACTIVE_LEARNING_CAMPAIGN_DIRECTORY_PATH",
        Path("").absolute(),
    )
    with pytest.raises(campaign_context.CampaignSelectionError):
        campaign_context.selected_campaign_dir()


def test_campaign_config_load_edit_save_preserves_hidden_fields(tmp_path, monkeypatch):
    import importlib
    import yaml
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    set_selected_campaign_dir(tmp_path)
    payload = {
        "schema_version": 2,
        "system_name": "WATER",
        "resources": {
            "partition": "multicore",
            "walltime_hours": 12,
            "mem_per_cpu": "5G",
            "cpus_per_task": 4,
            "ntasks": 1,
            "ariadne_cpus_per_task": 4,
            "gradient_parallel_backend": "serial",
        },
        "gaussian": {
            "method": "PBE0",
            "basis_set": "def2-SVP",
            "charge": 0,
            "spin_multiplicity": 1,
            "extra_keywords": "scf=tight",
            "nproc": 2,
            "mem": "6GB",
        },
        "ferebus": {
            "properties": ["iqa", "q00"],
            "scaling": False,
            "full_ARD": False,
        },
        "acquisition": {
            "property_name": "q00",
            "allow_uniform_posterior_fallback": True,
        },
    }
    (tmp_path / "campaign.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    monkeypatch.setattr(menu, "_pause", lambda: None)

    menu.EditCampaignConfigFunctions.load_from_disk()
    menu.get_campaign_config().system_name = "WATER_AL"
    menu.EditCampaignConfigFunctions.save_to_disk()

    reloaded = CampaignConfig.from_yaml(tmp_path / "campaign.yaml")
    assert reloaded.system_name == "WATER_AL"
    assert reloaded.resources.walltime_hours == 12
    assert reloaded.resources.gradient_parallel_backend == "serial"
    assert reloaded.gaussian.method == "PBE0"
    assert reloaded.ferebus.properties == ["iqa", "q00"]
    assert reloaded.ferebus.scaling is False
    assert reloaded.acquisition.allow_uniform_posterior_fallback is True


def test_config_save_rejects_invalid_without_writing(tmp_path, monkeypatch):
    import importlib
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    set_selected_campaign_dir(tmp_path)
    monkeypatch.setattr(menu, "_pause", lambda: None)
    menu._replace_campaign_config(CampaignConfig(system_name="bad name"), loaded_from=None)

    menu.EditCampaignConfigFunctions.save_to_disk()

    assert not (tmp_path / "campaign.yaml").exists()


def test_preflight_success_reports_all_current_dependencies(monkeypatch, capsys):
    import importlib
    from types import SimpleNamespace
    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_menu"
    )
    import ichor.hpc.active_learning.daemon.preflight as preflight

    fake = SimpleNamespace(
        all_present=True,
        sbatch_path="/bin/sbatch",
        sacct_path="/bin/sacct",
        gaussian_binary="/bin/g16",
        aimall_path="/bin/aimqb.ish",
        ferebus_path="/bin/ferebus",
        bc_path="/bin/bc",
    )
    monkeypatch.setattr(preflight, "check_backends", lambda: fake)
    monkeypatch.setattr(menu, "user_input_free_flow", lambda *args, **kwargs: "")

    menu.DaemonControlFunctions.preflight_backends()

    out = capsys.readouterr().out
    assert "polus_rs" in out
    assert "pyferebus" in out
    assert "bc:" in out


def test_reconcile_paths_pass_allow_fresh_init_flag(tmp_path, monkeypatch):
    import importlib
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_menu"
    )
    import ichor.hpc.active_learning.cli as daemon_cli

    set_selected_campaign_dir(tmp_path)
    calls = []
    monkeypatch.setattr(menu, "user_input_free_flow", lambda *args, **kwargs: "YES")
    monkeypatch.setattr(daemon_cli, "cmd_reconcile", lambda ns: calls.append(ns.allow_fresh_init) or 0)

    menu.DaemonControlFunctions.reconcile()
    menu.DaemonControlFunctions.reconcile_allow_fresh_init()

    assert calls == [False, True]


def test_import_pool_passes_no_outlier_filter_choice(tmp_path, monkeypatch):
    import importlib
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_menu"
    )
    import ichor.cli.useful_functions as useful_functions
    import ichor.hpc.active_learning.cli as daemon_cli

    set_selected_campaign_dir(tmp_path)
    calls = []
    monkeypatch.setattr(useful_functions, "user_input_path", lambda *args, **kwargs: "pool.xyz")
    monkeypatch.setattr(menu, "user_input_bool", lambda *args, **kwargs: True)
    monkeypatch.setattr(menu, "user_input_free_flow", lambda *args, **kwargs: "")
    monkeypatch.setattr(daemon_cli, "cmd_import_pool", lambda ns: calls.append(ns) or 0)

    menu.DaemonControlFunctions.import_trajectory_pool()

    assert calls
    assert calls[0].campaign_dir == str(tmp_path)
    assert calls[0].source == "pool.xyz"
    assert calls[0].no_outlier_filter is True


def test_foreground_launch_uses_resume_when_selected(tmp_path, monkeypatch):
    import importlib
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_submenus."
        "start_daemon_foreground_submenu"
    )
    import ichor.hpc.active_learning.cli as daemon_cli

    set_selected_campaign_dir(tmp_path)
    menu.start_daemon_foreground_menu_options.selected_command = "resume"
    menu.start_daemon_foreground_menu_options.selected_mode = "dry-run"
    menu.start_daemon_foreground_menu_options.selected_config = ""
    menu.start_daemon_foreground_menu_options.selected_preset = "csf4"
    menu.start_daemon_foreground_menu_options.selected_poll_interval = 3
    menu.start_daemon_foreground_menu_options.selected_max_ticks = 2
    calls = []
    monkeypatch.setattr(menu, "user_input_free_flow", lambda *args, **kwargs: "")
    monkeypatch.setattr(daemon_cli, "cmd_start", lambda ns: calls.append(("start", ns)) or 0)
    monkeypatch.setattr(daemon_cli, "cmd_resume", lambda ns: calls.append(("resume", ns)) or 0)

    menu.StartDaemonForegroundFunctions.launch()

    assert calls[0][0] == "resume"
    ns = calls[0][1]
    assert ns.campaign_dir == str(tmp_path)
    assert ns.dry_run is True
    assert ns.preset == "csf4"
    assert ns.poll_interval == 3
    assert ns.max_ticks == 2


def test_background_launch_reports_checked_result(tmp_path, monkeypatch, capsys):
    import importlib
    from types import SimpleNamespace
    from pathlib import Path
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_submenus."
        "start_daemon_background_submenu"
    )

    set_selected_campaign_dir(tmp_path)
    menu.start_daemon_background_menu_options.selected_command = "resume"
    menu.start_daemon_background_menu_options.selected_mode = "live"
    menu.start_daemon_background_menu_options.selected_config = ""
    menu.start_daemon_background_menu_options.selected_preset = ""
    menu.start_daemon_background_menu_options.selected_poll_interval = 0
    menu.start_daemon_background_menu_options.selected_max_ticks = 0
    seen = {}

    def fake_launch(campaign_dir, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            pid=123,
            log_path=Path(campaign_dir) / ".DATA" / "ACTIVE_LEARNING" / "daemon.menu_launched.out",
            returncode=12,
            exited_during_startup=True,
        )

    monkeypatch.setattr(menu, "launch_daemon_detached_checked", fake_launch)
    monkeypatch.setattr(menu, "user_input_free_flow", lambda *args, **kwargs: "")

    menu.StartDaemonBackgroundFunctions.launch()

    assert seen["command"] == "resume"
    assert seen["mode"] == "live"
    assert "exited during startup" in capsys.readouterr().out
