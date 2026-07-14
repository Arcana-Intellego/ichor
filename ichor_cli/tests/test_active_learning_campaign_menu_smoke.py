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
from types import SimpleNamespace


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
        "Toggle status JSON output",
        "Toggle status verbose output",
        "Show status",
        "Show sampling protocol summary",
        "Show config editability windows",
        "Recovery dashboard",
        "Campaign live preflight",
        "Submit compute-node environment smoke",
        "Initialise Campaign / Import Inputs",
        "Start/Resume Daemon (Foreground)",
        "Start/Resume Daemon (Background)",
        "Stop daemon",
        "Reconcile state",
        "Reconcile --apply",
        "Reconcile --archive-staging --apply",
        "Reconcile force-resubmit current array",
        "Reconcile and retrain FEREBUS",
        "Restore campaign.yaml proposal from config lock",
        "Reconcile state with --allow-fresh-init",
    ):
        assert expected in texts, "missing item: " + expected


def test_daemon_control_stop_can_request_job_cancellation(monkeypatch):
    import importlib

    import ichor.hpc.active_learning.cli as cli_mod
    menu_mod = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_menu"
    )

    seen = {}

    def fake_stop(ns):
        seen["cancel_jobs"] = ns.cancel_jobs
        seen["stop_mode"] = ns.stop_mode
        seen["after_iteration"] = ns.after_iteration
        return 0

    monkeypatch.setattr(
        menu_mod,
        "_guarded_campaign_dir_ns",
        lambda: SimpleNamespace(campaign_dir="/tmp/campaign"),
    )
    monkeypatch.setattr(menu_mod, "user_input_free_flow", lambda prompt, default: "YES")
    monkeypatch.setattr(cli_mod, "cmd_stop", fake_stop)

    menu_mod.DaemonControlFunctions.stop_daemon()

    assert seen["cancel_jobs"] is True
    assert seen["stop_mode"] == "immediate"
    assert seen["after_iteration"] is None


def test_daemon_control_stop_exposes_boundary_modes(monkeypatch):
    import importlib

    import ichor.hpc.active_learning.cli as cli_mod
    menu_mod = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_menu"
    )
    calls = []
    answers = iter(["2", "", "3", "4", ""])

    monkeypatch.setattr(
        menu_mod,
        "_guarded_campaign_dir_ns",
        lambda: SimpleNamespace(campaign_dir="/tmp/campaign"),
    )
    monkeypatch.setattr(
        menu_mod,
        "user_input_free_flow",
        lambda prompt, default: next(answers),
    )
    monkeypatch.setattr(cli_mod, "cmd_stop", lambda ns: calls.append(ns) or 0)

    menu_mod.DaemonControlFunctions.stop_daemon()
    menu_mod.DaemonControlFunctions.stop_daemon()

    assert calls[0].stop_mode == "after_phase"
    assert calls[0].after_iteration is None
    assert calls[0].cancel_jobs is False
    assert calls[1].stop_mode == "immediate"
    assert calls[1].after_iteration == 4
    assert calls[1].cancel_jobs is False


def test_daemon_control_archive_staging_dispatches_reconcile(monkeypatch):
    import importlib

    import ichor.hpc.active_learning.cli as cli_mod
    menu_mod = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_menu"
    )

    seen = {}

    def fake_reconcile(ns):
        seen["apply"] = ns.apply
        seen["archive_staging"] = ns.archive_staging
        seen["restore_config_from_lock"] = ns.restore_config_from_lock
        return 0

    monkeypatch.setattr(
        menu_mod,
        "_guarded_campaign_dir_ns",
        lambda: SimpleNamespace(campaign_dir="/tmp/campaign"),
    )
    monkeypatch.setattr(menu_mod, "user_input_free_flow", lambda prompt, default: "YES")
    monkeypatch.setattr(cli_mod, "cmd_reconcile", fake_reconcile)

    menu_mod.DaemonControlFunctions.reconcile_archive_stale_staging()

    assert seen == {
        "apply": True,
        "archive_staging": True,
        "restore_config_from_lock": False,
    }


def test_daemon_control_status_passes_visible_display_options(monkeypatch):
    import importlib

    import ichor.hpc.active_learning.cli as cli_mod
    menu_mod = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_menu"
    )

    seen = {}

    def fake_status(ns):
        seen["json"] = ns.json
        seen["verbose"] = ns.verbose
        return 0

    monkeypatch.setattr(
        menu_mod,
        "_guarded_campaign_dir_ns",
        lambda: SimpleNamespace(campaign_dir="/tmp/campaign"),
    )
    monkeypatch.setattr(menu_mod, "user_input_free_flow", lambda prompt, default: "")
    monkeypatch.setattr(cli_mod, "cmd_status", fake_status)
    menu_mod.daemon_control_menu_options.status_json = True
    menu_mod.daemon_control_menu_options.status_verbose = True

    menu_mod.DaemonControlFunctions.show_status()

    assert seen == {"json": True, "verbose": True}


def test_edit_campaign_config_menu_items():
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_submenus.edit_campaign_config_menu import (
        edit_acquisition_config_menu,
        edit_campaign_config_menu,
        get_campaign_config,
    )
    texts = [it.text for it in edit_campaign_config_menu.items]
    for expected in (
        "Show dense/internal diagnostic config",
        "Show sampling protocol summary",
        "Load from disk",
        "Reset to defaults",
        "Validate current config",
        "Edit campaign",
        "Edit point_allocation",
        "Edit resource defaults",
        "Edit POLUS resources",
        "Edit Gaussian runtime resources",
        "Edit AIMAll resources",
        "Edit ARIADNE resources",
        "Edit FEREBUS resources",
        "Edit Gaussian block",
        "Edit seed_selection",
        "Edit anti_overlap",
        "Edit phase_b",
        "Edit geometry_novelty",
        "Edit FEREBUS block",
        "Edit acquisition",
        "Edit ARIADNE Block",
        "Edit stop",
        "Edit adversarial_safety",
        "Edit error_calibration",
        "Edit quality_gates",
        "Edit runtime",
        "Show unsaved changes",
        "Show config lock review",
        "Show config editability windows",
        "Show pending config changes",
        "Discard unsaved changes / reload from disk",
        "Export dense/internal diagnostic snapshot",
        "Save to disk",
    ):
        assert expected in texts, "missing item: " + expected
    assert "Edit acquisition.subspace" not in texts
    assert "Edit acquisition.weights" not in texts
    assert "Edit acquisition.gradient" not in texts
    acquisition_texts = [it.text for it in edit_acquisition_config_menu.items]
    for expected in (
        "Edit acquisition core",
        "Edit acquisition.subspace",
        "Edit acquisition.weights",
        "Edit acquisition.spectral",
        "Edit acquisition.calibrated_energy",
        "Edit acquisition.fullspace_confinement",
        "Edit acquisition.size_normalisation",
        "Edit acquisition.driver",
        "Edit acquisition.gradient",
        "Edit acquisition.barrier",
        "Edit acquisition.stencils",
        "Edit acquisition.references",
    ):
        assert expected in acquisition_texts, "missing acquisition item: " + expected
    cfg = get_campaign_config()
    assert cfg.max_iterations >= 1
    assert hasattr(cfg, "ariadne")
    edit_acquisition_config_menu.parent = edit_campaign_config_menu
    acquisition_prologue = edit_acquisition_config_menu.get_prologue_text()
    assert "Acquisition config blocks:" in acquisition_prologue
    assert "Loaded from:" not in acquisition_prologue
    assert "Selected campaign:" not in acquisition_prologue


def test_campaign_config_block_submenus_show_values_and_edit_one_field(monkeypatch):
    import importlib

    import ichor.cli.main_menu_submenus.active_learning_campaign_menu.field_menu as field_menu
    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    menu._replace_campaign_config(CampaignConfig(), loaded_from=None)

    resources_menu = menu._BLOCK_MENUS_BY_LABEL["Edit resource defaults"]
    rendered = resources_menu.this_menu_options()
    assert "resources.defaults.partition" in rendered
    assert "resources.defaults.walltime_hours" in rendered
    assert "resources.defaults.cpus_per_task" in rendered
    assert "resources.defaults.mem_per_cpu" in rendered
    assert "resources.gradient_parallel_backend" in rendered

    allocation_menu = menu._BLOCK_MENUS_BY_LABEL["Edit point_allocation"]
    allocation_rendered = allocation_menu.this_menu_options()
    assert "point_allocation.bootstrap_training_size" in allocation_rendered
    assert "point_allocation.bootstrap_internal_validation_size" in allocation_rendered
    assert "point_allocation.bootstrap_external_validation_size" in allocation_rendered
    assert "point_allocation.batch_training_size" in allocation_rendered
    assert "point_allocation.batch_internal_validation_size" in allocation_rendered
    allocation_menu.parent = menu.edit_campaign_config_menu
    allocation_prologue = allocation_menu.get_prologue_text()
    assert "point_allocation.anchor" not in allocation_prologue

    texts = [it.text for it in resources_menu.items]
    assert "Set partition" in texts
    assert "Set walltime_hours" in texts
    assert "Set cpus_per_task" in texts
    resources_menu.parent = menu.edit_campaign_config_menu
    prologue = resources_menu.get_prologue_text()
    assert "Current values for this campaign.yaml block:" in prologue
    assert "resources.defaults.partition" in prologue
    assert "Loaded from:" not in prologue
    assert "Selected campaign:" not in prologue

    cfg = menu.get_campaign_config()
    old_partition = cfg.resources.defaults.partition
    spec = next(
        spec
        for spec in resources_menu.this_menu_options.fields
        if spec.path == "resources.defaults.walltime_hours"
    )
    monkeypatch.setattr(field_menu, "user_input_float", lambda prompt, default: 37)

    menu._edit_field(spec)

    assert cfg.resources.defaults.walltime_hours == 37
    assert cfg.resources.defaults.partition == old_partition
    assert "resources.defaults.walltime_hours: 37" in resources_menu.this_menu_options()


def test_campaign_config_fields_show_static_editability_windows_before_start():
    import importlib

    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    menu._replace_campaign_config(CampaignConfig(), loaded_from=None)

    gaussian_menu = menu._BLOCK_MENUS_BY_LABEL["Edit Gaussian block"]
    rendered = gaussian_menu.this_menu_options()

    assert "gaussian.basis_set" in rendered
    assert "window: editable until first Gaussian staging/submission" in rendered


def test_edit_gaussian_basis_set_saves_to_selected_campaign_yaml(
    tmp_path, monkeypatch, capsys,
):
    import importlib

    import ichor.cli.main_menu_submenus.active_learning_campaign_menu.field_menu as field_menu
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )

    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    cfg = CampaignConfig()
    old_basis = cfg.gaussian.basis_set
    cfg.to_yaml(campaign_dir / "campaign.yaml")
    set_selected_campaign_dir(campaign_dir)
    assert menu.load_config_for_campaign_dir(campaign_dir, quiet=True)

    gaussian_menu = menu._BLOCK_MENUS_BY_LABEL["Edit Gaussian block"]
    basis_spec = next(
        spec
        for spec in gaussian_menu.this_menu_options.fields
        if spec.path == "gaussian.basis_set"
    )

    monkeypatch.setattr(
        field_menu,
        "user_input_free_flow",
        lambda prompt, default: "6-31+G(d,p)",
    )
    monkeypatch.setattr(menu, "_pause", lambda: None)

    menu._edit_field(basis_spec)
    assert menu.get_campaign_config().gaussian.basis_set == "6-31+G(d,p)"
    assert menu.has_unsaved_config_changes()
    assert "gaussian.basis_set" in menu.dirty_paths()
    assert (
        "  gaussian.basis_set: " + old_basis + " -> 6-31+G(d,p)"
        in menu._pending_sparse_yaml()
    )

    menu.EditCampaignConfigFunctions.save_to_disk()
    output = capsys.readouterr().out
    assert "Changed fields saved:" in output
    assert (
        "  gaussian.basis_set: " + old_basis + " -> 6-31+G(d,p)"
        in output
    )

    loaded = CampaignConfig.from_yaml(campaign_dir / "campaign.yaml")
    assert loaded.gaussian.basis_set == "6-31+G(d,p)"
    raw = (campaign_dir / "campaign.yaml").read_text(encoding="utf-8")
    assert "basis_set" in raw
    assert "6-31+G(d,p)" in raw
    assert not menu.has_unsaved_config_changes()


def test_select_campaign_directory_auto_loads_campaign_yaml(tmp_path, monkeypatch):
    import importlib

    from ichor.hpc.active_learning.config import CampaignConfig

    top = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_menu"
    )
    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    cfg = CampaignConfig()
    cfg.gaussian.basis_set = "def2-TZVP"
    cfg.to_yaml(tmp_path / "campaign.yaml")

    monkeypatch.setattr(top, "user_input_path", lambda prompt, default_path: tmp_path)
    monkeypatch.setattr(menu, "user_input_free_flow", lambda *args, **kwargs: "YES")

    top.ActiveLearningCampaignFunctions.select_campaign_directory()

    assert menu.get_campaign_config().gaussian.basis_set == "def2-TZVP"
    assert str(tmp_path / "campaign.yaml") == menu.edit_campaign_config_menu_options.loaded_from
    assert not menu.has_unsaved_config_changes()


def test_active_learning_menu_auto_adopts_valid_campaign_cwd(tmp_path, monkeypatch):
    import importlib

    import ichor.cli.global_menu_variables as globals_
    import ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context as campaign_context
    from ichor.hpc.active_learning.config import CampaignConfig

    top = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_menu"
    )
    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    cfg = CampaignConfig()
    cfg.gaussian.basis_set = "def2-TZVP"
    cfg.to_yaml(tmp_path / "campaign.yaml")
    monkeypatch.setattr(campaign_context, "_explicit_selection", False)
    monkeypatch.setattr(
        globals_,
        "SELECTED_ACTIVE_LEARNING_CAMPAIGN_DIRECTORY_PATH",
        tmp_path,
    )
    top.active_learning_campaign_menu_options.selected_active_learning_campaign_directory_path = (
        tmp_path
    )

    rendered = top.active_learning_campaign_menu_options()

    assert str(tmp_path) in rendered
    assert "(none)" not in rendered
    assert campaign_context.selected_campaign_dir() == tmp_path
    assert menu.get_campaign_config().gaussian.basis_set == "def2-TZVP"
    assert str(tmp_path / "campaign.yaml") == menu.edit_campaign_config_menu_options.loaded_from


def test_save_to_disk_works_after_auto_adopting_valid_campaign_cwd(
    tmp_path, monkeypatch,
):
    import importlib

    import ichor.cli.global_menu_variables as globals_
    import ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context as campaign_context
    from ichor.hpc.active_learning.config import CampaignConfig

    top = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_menu"
    )
    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    CampaignConfig().to_yaml(tmp_path / "campaign.yaml")
    monkeypatch.setattr(campaign_context, "_explicit_selection", False)
    monkeypatch.setattr(
        globals_,
        "SELECTED_ACTIVE_LEARNING_CAMPAIGN_DIRECTORY_PATH",
        tmp_path,
    )
    monkeypatch.setattr(menu, "_pause", lambda: None)
    top.active_learning_campaign_menu_options.selected_active_learning_campaign_directory_path = (
        tmp_path
    )

    top.active_learning_campaign_menu_options()
    menu._set_config_value("gaussian.basis_set", "6-31+G(d,p)")
    menu.EditCampaignConfigFunctions.save_to_disk()

    loaded = CampaignConfig.from_yaml(tmp_path / "campaign.yaml")
    assert loaded.gaussian.basis_set == "6-31+G(d,p)"
    assert not menu.has_unsaved_config_changes()


def test_active_learning_menu_does_not_show_non_campaign_cwd_as_selected(
    tmp_path, monkeypatch,
):
    import importlib

    import ichor.cli.global_menu_variables as globals_
    import ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context as campaign_context

    top = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_menu"
    )
    monkeypatch.setattr(campaign_context, "_explicit_selection", False)
    monkeypatch.setattr(
        globals_,
        "SELECTED_ACTIVE_LEARNING_CAMPAIGN_DIRECTORY_PATH",
        tmp_path,
    )
    top.active_learning_campaign_menu_options.selected_active_learning_campaign_directory_path = (
        tmp_path
    )

    rendered = top.active_learning_campaign_menu_options()

    assert "Selected active learning campaign directory path: (none)" in rendered
    assert "No active learning campaign directory selected yet" in rendered


def test_switch_campaign_auto_loads_new_yaml_without_reusing_old_state(tmp_path):
    import importlib

    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    c1 = tmp_path / "c1"
    c2 = tmp_path / "c2"
    c1.mkdir()
    c2.mkdir()
    cfg1 = CampaignConfig()
    cfg1.gaussian.basis_set = "6-31+G(d,p)"
    cfg1.to_yaml(c1 / "campaign.yaml")
    cfg2 = CampaignConfig()
    cfg2.gaussian.basis_set = "def2-TZVP"
    cfg2.to_yaml(c2 / "campaign.yaml")

    assert menu.load_config_for_campaign_dir(c1, quiet=True)
    assert menu.get_campaign_config().gaussian.basis_set == "6-31+G(d,p)"

    assert menu.load_config_for_campaign_dir(c2, quiet=True)
    assert menu.get_campaign_config().gaussian.basis_set == "def2-TZVP"


def test_save_validation_failure_keeps_dirty_state(tmp_path, monkeypatch):
    import importlib

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    tmp_path.mkdir(exist_ok=True)
    set_selected_campaign_dir(tmp_path)
    cfg = CampaignConfig()
    cfg.to_yaml(tmp_path / "campaign.yaml")
    menu.load_config_for_campaign_dir(tmp_path, quiet=True)
    monkeypatch.setattr(menu, "_pause", lambda: None)

    menu._set_config_value("gaussian.basis_set", "6-31+G(d,p)")
    menu._set_config_value("point_allocation.batch_training_size", 0)

    menu.EditCampaignConfigFunctions.save_to_disk()

    assert menu.has_unsaved_config_changes()
    assert "gaussian.basis_set" in menu.dirty_paths()
    assert "Validation failed" in menu.edit_campaign_config_menu_options.last_error


def test_save_reload_failure_leaves_existing_campaign_yaml_untouched(tmp_path, monkeypatch):
    import importlib

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    original = CampaignConfig()
    original.gaussian.basis_set = "def2-TZVP"
    original.to_yaml(tmp_path / "campaign.yaml")
    original_text = (tmp_path / "campaign.yaml").read_text(encoding="utf-8")
    set_selected_campaign_dir(tmp_path)
    assert menu.load_config_for_campaign_dir(tmp_path, quiet=True)
    monkeypatch.setattr(menu, "_pause", lambda: None)
    import ichor.hpc.active_learning.campaign_yaml as campaign_yaml

    real_from_yaml = campaign_yaml.CampaignConfig.from_yaml

    def fail_reload(path):
        if ".verify." in str(path):
            raise RuntimeError("synthetic reload failure")
        return real_from_yaml(path)

    menu._set_config_value("gaussian.basis_set", "6-31+G(d,p)")
    monkeypatch.setattr(campaign_yaml.CampaignConfig, "from_yaml", fail_reload)

    menu.EditCampaignConfigFunctions.save_to_disk()

    assert (tmp_path / "campaign.yaml").read_text(encoding="utf-8") == original_text
    loaded = real_from_yaml(tmp_path / "campaign.yaml")
    assert loaded.gaussian.basis_set == "def2-TZVP"
    assert menu.has_unsaved_config_changes()
    assert "reload failed before write" in menu.edit_campaign_config_menu_options.last_error
    assert not list(tmp_path.glob("*.verify.*.tmp"))


def test_dirty_paths_clear_when_field_is_reverted(tmp_path):
    import importlib

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    cfg = CampaignConfig()
    cfg.gaussian.basis_set = "def2-TZVP"
    cfg.to_yaml(tmp_path / "campaign.yaml")
    set_selected_campaign_dir(tmp_path)
    assert menu.load_config_for_campaign_dir(tmp_path, quiet=True)

    menu._set_config_value("gaussian.basis_set", "6-31+G(d,p)")
    assert "gaussian.basis_set" in menu.dirty_paths()

    menu._set_config_value("gaussian.basis_set", "def2-TZVP")

    assert not menu.has_unsaved_config_changes()
    assert menu.dirty_paths() == []


def test_load_config_rejects_nonexistent_campaign_directory(tmp_path):
    import importlib

    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    existing = tmp_path / "existing"
    existing.mkdir()
    CampaignConfig().to_yaml(existing / "campaign.yaml")
    assert menu.load_config_for_campaign_dir(existing, quiet=True)
    previous_campaign = menu.edit_campaign_config_menu_options.selected_campaign

    missing = tmp_path / "missing"
    assert not menu.load_config_for_campaign_dir(missing, quiet=True)

    assert menu.edit_campaign_config_menu_options.selected_campaign == previous_campaign
    assert "does not exist" in menu.edit_campaign_config_menu_options.last_error


def test_field_transform_error_is_nonfatal(monkeypatch):
    import ichor.cli.main_menu_submenus.active_learning_campaign_menu.field_menu as field_menu

    calls = []
    spec = field_menu.spec("resources.aimall.cpus_per_task", "str", transform=int)
    monkeypatch.setattr(field_menu, "user_input_free_flow", lambda *args, **kwargs: "bad")

    field_menu.edit_field(spec, lambda path: "auto", lambda path, value: calls.append(value))

    assert calls == []


def test_daemon_launch_refuses_when_config_editor_dirty(tmp_path, monkeypatch, capsys):
    import importlib

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    edit_menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    start_menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_submenus.start_daemon_foreground_submenu"
    )
    cfg = CampaignConfig()
    cfg.to_yaml(tmp_path / "campaign.yaml")
    set_selected_campaign_dir(tmp_path)
    edit_menu.load_config_for_campaign_dir(tmp_path, quiet=True)
    edit_menu._set_config_value("gaussian.basis_set", "6-31+G(d,p)")
    monkeypatch.setattr(start_menu, "user_input_free_flow", lambda *args, **kwargs: "")

    start_menu.StartDaemonForegroundFunctions.launch()

    out = capsys.readouterr().out
    assert "unsaved campaign.yaml edits" in out
    assert "gaussian.basis_set" in out


def _write_started_campaign_with_lock(campaign, config):
    from ichor.hpc.active_learning.daemon.config_lock import write_config_lock
    from ichor.hpc.active_learning.daemon.state import fresh_campaign_state, write_state

    campaign.mkdir(exist_ok=True)
    config.to_yaml(campaign / "campaign.yaml")
    data = campaign / ".DATA" / "ACTIVE_LEARNING"
    data.mkdir(parents=True, exist_ok=True)
    write_state(data / "state.json", fresh_campaign_state())
    write_config_lock(campaign, config)
    pointdir = campaign / ".DATA" / "STAGING" / "initial" / "point-0000.pointdir"
    pointdir.mkdir(parents=True, exist_ok=True)
    (pointdir / "input.gjf").write_text("# synthetic staged Gaussian input\n", encoding="utf-8")


def test_config_lock_status_renders_in_field_menu(tmp_path):
    import importlib

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    original = CampaignConfig()
    _write_started_campaign_with_lock(tmp_path, original)
    set_selected_campaign_dir(tmp_path)
    menu.load_config_for_campaign_dir(tmp_path, quiet=True)
    menu._set_config_value("gaussian.basis_set", "6-31+G(d,p)")

    rendered = menu._BLOCK_MENUS_BY_LABEL["Edit Gaussian block"].this_menu_options()

    assert "gaussian.basis_set: 6-31+G(d,p) [blocked:" in rendered


def test_save_refuses_blocked_config_lock_change(tmp_path, monkeypatch):
    import importlib

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    original = CampaignConfig()
    _write_started_campaign_with_lock(tmp_path, original)
    set_selected_campaign_dir(tmp_path)
    menu.load_config_for_campaign_dir(tmp_path, quiet=True)
    menu._set_config_value("gaussian.basis_set", "6-31+G(d,p)")
    monkeypatch.setattr(menu, "_pause", lambda: None)

    menu.EditCampaignConfigFunctions.save_to_disk()

    loaded = CampaignConfig.from_yaml(tmp_path / "campaign.yaml")
    assert loaded.gaussian.basis_set == original.gaussian.basis_set
    assert menu.has_unsaved_config_changes()
    assert "Config lock blocked save" in menu.edit_campaign_config_menu_options.last_error


def test_save_refuses_when_started_config_lock_review_fails(tmp_path, monkeypatch, capsys):
    import importlib

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    original = CampaignConfig()
    _write_started_campaign_with_lock(tmp_path, original)
    set_selected_campaign_dir(tmp_path)
    menu.load_config_for_campaign_dir(tmp_path, quiet=True)
    menu._set_config_value("runtime.poll_interval_seconds", 7)
    monkeypatch.setattr(menu, "_pause", lambda: None)
    monkeypatch.setattr(
        menu,
        "_read_editor_state_for_lock",
        lambda campaign_dir: (_ for _ in ()).throw(RuntimeError("broken state")),
    )

    menu.EditCampaignConfigFunctions.save_to_disk()

    out = capsys.readouterr().out
    assert "config-lock review is unavailable" in out
    assert "broken state" in out


def test_save_allows_safe_runtime_change_under_config_lock(tmp_path, monkeypatch):
    import importlib

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    original = CampaignConfig()
    _write_started_campaign_with_lock(tmp_path, original)
    set_selected_campaign_dir(tmp_path)
    menu.load_config_for_campaign_dir(tmp_path, quiet=True)
    menu._set_config_value("resources.gaussian.walltime_hours", 3)
    monkeypatch.setattr(menu, "_pause", lambda: None)

    menu.EditCampaignConfigFunctions.save_to_disk()

    loaded = CampaignConfig.from_yaml(tmp_path / "campaign.yaml")
    assert loaded.resources.gaussian.walltime_hours == 3
    assert not menu.has_unsaved_config_changes()


def test_show_config_lock_review_lists_blocked_changes(tmp_path, monkeypatch, capsys):
    import importlib

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    original = CampaignConfig()
    _write_started_campaign_with_lock(tmp_path, original)
    set_selected_campaign_dir(tmp_path)
    menu.load_config_for_campaign_dir(tmp_path, quiet=True)
    menu._set_config_value("gaussian.basis_set", "6-31+G(d,p)")
    monkeypatch.setattr(menu, "_pause", lambda: None)

    menu.EditCampaignConfigFunctions.show_config_lock_review()

    out = capsys.readouterr().out
    assert "Blocked config changes" in out
    assert "gaussian.basis_set" in out


def test_daemon_launch_refuses_saved_blocked_config_change(
    tmp_path, monkeypatch, capsys,
):
    import importlib

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    edit_menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    start_menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_submenus.start_daemon_foreground_submenu"
    )
    original = CampaignConfig()
    _write_started_campaign_with_lock(tmp_path, original)
    changed = CampaignConfig()
    changed.gaussian.method = "b3lyp"
    changed.to_yaml(tmp_path / "campaign.yaml")
    set_selected_campaign_dir(tmp_path)
    edit_menu.load_config_for_campaign_dir(tmp_path, quiet=True)
    monkeypatch.setattr(start_menu, "user_input_free_flow", lambda *args, **kwargs: "")

    start_menu.StartDaemonForegroundFunctions.launch()

    out = capsys.readouterr().out
    assert "Saved campaign.yaml differs from the config lock" in out
    assert "gaussian.method" in out


def test_foreground_launch_reviews_selected_config_override(
    tmp_path, monkeypatch, capsys,
):
    import importlib

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    edit_menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    start_menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_submenus.start_daemon_foreground_submenu"
    )
    original = CampaignConfig()
    _write_started_campaign_with_lock(tmp_path, original)
    override = CampaignConfig()
    override.gaussian.method = "b3lyp"
    override_path = tmp_path / "override.yaml"
    override.to_yaml(override_path)
    set_selected_campaign_dir(tmp_path)
    edit_menu.load_config_for_campaign_dir(tmp_path, quiet=True)
    start_menu.start_daemon_foreground_menu_options.selected_command = "resume"
    start_menu.start_daemon_foreground_menu_options.selected_mode = "dry_run"
    start_menu.start_daemon_foreground_menu_options.selected_config = str(override_path)
    calls = []
    import ichor.hpc.active_learning.cli as daemon_cli

    monkeypatch.setattr(start_menu, "user_input_free_flow", lambda *args, **kwargs: "")
    monkeypatch.setattr(daemon_cli, "cmd_resume", lambda ns: calls.append(ns) or 0)

    start_menu.StartDaemonForegroundFunctions.launch()

    out = capsys.readouterr().out
    assert "Selected config override differs from the config lock" in out
    assert "gaussian.method" in out
    assert calls == []


def test_foreground_launch_refuses_missing_config_override(
    tmp_path, monkeypatch, capsys,
):
    import importlib

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    edit_menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    start_menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_submenus.start_daemon_foreground_submenu"
    )
    CampaignConfig().to_yaml(tmp_path / "campaign.yaml")
    set_selected_campaign_dir(tmp_path)
    edit_menu.load_config_for_campaign_dir(tmp_path, quiet=True)
    start_menu.start_daemon_foreground_menu_options.selected_command = "resume"
    start_menu.start_daemon_foreground_menu_options.selected_mode = "dry_run"
    start_menu.start_daemon_foreground_menu_options.selected_config = str(
        tmp_path / "missing.yaml"
    )
    calls = []
    import ichor.hpc.active_learning.cli as daemon_cli

    monkeypatch.setattr(start_menu, "user_input_free_flow", lambda *args, **kwargs: "")
    monkeypatch.setattr(daemon_cli, "cmd_resume", lambda ns: calls.append(ns) or 0)

    start_menu.StartDaemonForegroundFunctions.launch()

    out = capsys.readouterr().out
    assert "Selected config override could not be reviewed" in out
    assert "not a readable file" in out
    assert calls == []


def test_foreground_launch_refuses_invalid_config_override(
    tmp_path, monkeypatch, capsys,
):
    import importlib

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    edit_menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    start_menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_submenus.start_daemon_foreground_submenu"
    )
    CampaignConfig().to_yaml(tmp_path / "campaign.yaml")
    override_path = tmp_path / "invalid.yaml"
    override_path.write_text(
        "schema_version: 3\nsplit:\n  train_fraction: 0.9\n  val_mid_fraction: 0.9\n",
        encoding="utf-8",
    )
    set_selected_campaign_dir(tmp_path)
    edit_menu.load_config_for_campaign_dir(tmp_path, quiet=True)
    start_menu.start_daemon_foreground_menu_options.selected_command = "resume"
    start_menu.start_daemon_foreground_menu_options.selected_mode = "dry_run"
    start_menu.start_daemon_foreground_menu_options.selected_config = str(override_path)
    calls = []
    import ichor.hpc.active_learning.cli as daemon_cli

    monkeypatch.setattr(start_menu, "user_input_free_flow", lambda *args, **kwargs: "")
    monkeypatch.setattr(daemon_cli, "cmd_resume", lambda ns: calls.append(ns) or 0)

    start_menu.StartDaemonForegroundFunctions.launch()

    out = capsys.readouterr().out
    assert "Selected config override could not be reviewed" in out
    assert "could not be loaded" in out
    assert calls == []


def test_foreground_launch_allows_clean_override_when_campaign_yaml_is_dirty_on_disk(
    tmp_path, monkeypatch,
):
    import importlib

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    edit_menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    start_menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_submenus.start_daemon_foreground_submenu"
    )
    original = CampaignConfig()
    _write_started_campaign_with_lock(tmp_path, original)
    clean_override_path = tmp_path / "clean_override.yaml"
    original.to_yaml(clean_override_path)
    changed_campaign = CampaignConfig()
    changed_campaign.gaussian.method = "b3lyp"
    changed_campaign.to_yaml(tmp_path / "campaign.yaml")
    set_selected_campaign_dir(tmp_path)
    edit_menu.load_config_for_campaign_dir(tmp_path, quiet=True)
    start_menu.start_daemon_foreground_menu_options.selected_command = "resume"
    start_menu.start_daemon_foreground_menu_options.selected_mode = "dry_run"
    start_menu.start_daemon_foreground_menu_options.selected_config = str(clean_override_path)
    calls = []
    import ichor.hpc.active_learning.cli as daemon_cli

    monkeypatch.setattr(start_menu, "user_input_free_flow", lambda *args, **kwargs: "")
    monkeypatch.setattr(daemon_cli, "cmd_resume", lambda ns: calls.append(ns) or 0)

    start_menu.StartDaemonForegroundFunctions.launch()

    assert len(calls) == 1
    assert calls[0].config == str(clean_override_path)


def test_background_launch_reviews_selected_config_override(
    tmp_path, monkeypatch, capsys,
):
    import importlib

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    edit_menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    start_menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_submenus.start_daemon_background_submenu"
    )
    original = CampaignConfig()
    _write_started_campaign_with_lock(tmp_path, original)
    override = CampaignConfig()
    override.gaussian.method = "b3lyp"
    override_path = tmp_path / "override.yaml"
    override.to_yaml(override_path)
    set_selected_campaign_dir(tmp_path)
    edit_menu.load_config_for_campaign_dir(tmp_path, quiet=True)
    start_menu.start_daemon_background_menu_options.selected_command = "resume"
    start_menu.start_daemon_background_menu_options.selected_mode = "dry_run"
    start_menu.start_daemon_background_menu_options.selected_config = str(override_path)
    calls = []

    monkeypatch.setattr(start_menu, "user_input_free_flow", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        start_menu,
        "launch_daemon_detached_checked",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    start_menu.StartDaemonBackgroundFunctions.launch()

    out = capsys.readouterr().out
    assert "Selected config override differs from the config lock" in out
    assert "gaussian.method" in out
    assert calls == []


def test_background_launch_refuses_invalid_config_override(
    tmp_path, monkeypatch, capsys,
):
    import importlib

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    edit_menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    start_menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_submenus.start_daemon_background_submenu"
    )
    CampaignConfig().to_yaml(tmp_path / "campaign.yaml")
    override_path = tmp_path / "invalid.yaml"
    override_path.write_text(
        "schema_version: 3\nsplit:\n  train_fraction: 0.9\n  val_mid_fraction: 0.9\n",
        encoding="utf-8",
    )
    set_selected_campaign_dir(tmp_path)
    edit_menu.load_config_for_campaign_dir(tmp_path, quiet=True)
    start_menu.start_daemon_background_menu_options.selected_command = "resume"
    start_menu.start_daemon_background_menu_options.selected_mode = "dry_run"
    start_menu.start_daemon_background_menu_options.selected_config = str(override_path)
    calls = []

    monkeypatch.setattr(start_menu, "user_input_free_flow", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        start_menu,
        "launch_daemon_detached_checked",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    start_menu.StartDaemonBackgroundFunctions.launch()

    out = capsys.readouterr().out
    assert "Selected config override could not be reviewed" in out
    assert "could not be loaded" in out
    assert calls == []


def test_acquisition_gradient_menu_exposes_active_fd_controls():
    import importlib

    from ichor.hpc.active_learning.config import CampaignConfig, VALID_GRADIENT_MODES

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    cfg = CampaignConfig()
    cfg.acquisition.gradient.mode = "active_fd"
    cfg.acquisition.gradient.active_step = 3.0e-3
    menu._replace_campaign_config(cfg, loaded_from=None)

    gradient_menu = menu._BLOCK_MENUS_BY_LABEL["Edit acquisition.gradient"]
    rendered = gradient_menu.this_menu_options()
    field_specs = {spec.path: spec for spec in gradient_menu.this_menu_options.fields}

    assert set(field_specs) == {
        "acquisition.gradient.mode",
        "acquisition.gradient.cartesian_step",
        "acquisition.gradient.active_step",
        "acquisition.gradient.regularization",
        "acquisition.gradient.cartesian_step_floor",
        "acquisition.gradient.ghost_mass_threshold",
        "acquisition.gradient.max_acquisition_grad_per_ang",
    }
    assert set(field_specs["acquisition.gradient.mode"].choices) == set(VALID_GRADIENT_MODES)
    assert "active_fd" in rendered
    assert "acquisition.gradient.active_step: 0.003" in rendered


def test_acquisition_driver_menu_exposes_driver_controls():
    import importlib

    from ichor.hpc.active_learning.config import (
        CampaignConfig,
        VALID_ACQUISITION_DRIVER_GRADIENT_BACKENDS,
        VALID_ACQUISITION_DRIVER_OBJECTIVES,
    )

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    cfg = CampaignConfig()
    cfg.acquisition.driver.enabled = True
    cfg.acquisition.driver.lambda_energy = 0.5
    menu._replace_campaign_config(cfg, loaded_from=None)

    driver_menu = menu._BLOCK_MENUS_BY_LABEL["Edit acquisition.driver"]
    rendered = driver_menu.this_menu_options()
    field_specs = {spec.path: spec for spec in driver_menu.this_menu_options.fields}

    assert set(field_specs) == {
        "acquisition.driver.enabled",
        "acquisition.driver.objective",
        "acquisition.driver.gradient_backend",
        "acquisition.driver.include_stencils",
        "acquisition.driver.analytic_movement",
        "acquisition.driver.analytic_whitened_distance",
        "acquisition.driver.analytic_pair_barriers",
        "acquisition.driver.analytic_fullspace_rmsd",
        "acquisition.driver.finite_difference_energy",
        "acquisition.driver.analytic_validation",
        "acquisition.driver.analytic_validation_tol_cosine",
        "acquisition.driver.lambda_energy",
        "acquisition.driver.lambda_movement",
        "acquisition.driver.lambda_distance",
        "acquisition.driver.lambda_fullspace",
        "acquisition.driver.lambda_chemistry",
    }
    assert set(field_specs["acquisition.driver.objective"].choices) == set(
        VALID_ACQUISITION_DRIVER_OBJECTIVES
    )
    assert set(field_specs["acquisition.driver.gradient_backend"].choices) == set(
        VALID_ACQUISITION_DRIVER_GRADIENT_BACKENDS
    )
    assert "acquisition.driver.enabled: True" in rendered
    assert "acquisition.driver.gradient_backend: fd" in rendered
    assert "acquisition.driver.lambda_energy: 0.5" in rendered


def test_campaign_config_csv_and_optional_float_field_editors(monkeypatch):
    import importlib

    import ichor.cli.main_menu_submenus.active_learning_campaign_menu.field_menu as field_menu
    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    cfg = CampaignConfig()
    cfg.quality_gates.max_abs_integration_error = 0.2
    menu._replace_campaign_config(cfg, loaded_from=None)

    ferebus_menu = menu._BLOCK_MENUS_BY_LABEL["Edit FEREBUS block"]
    properties_spec = next(
        spec
        for spec in ferebus_menu.this_menu_options.fields
        if spec.path == "ferebus.properties"
    )
    monkeypatch.setattr(field_menu, "user_input_free_flow", lambda prompt, default: "iqa, q00")
    menu._edit_field(properties_spec)
    assert menu.get_campaign_config().ferebus.properties == ["iqa", "q00"]

    quality_menu = menu._BLOCK_MENUS_BY_LABEL["Edit quality_gates"]
    optional_spec = next(
        spec
        for spec in quality_menu.this_menu_options.fields
        if spec.path == "quality_gates.max_abs_integration_error"
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "null")
    menu._edit_field(optional_spec)
    assert menu.get_campaign_config().quality_gates.max_abs_integration_error is None


def test_campaign_config_optional_int_field_editor(monkeypatch):
    import importlib

    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    cfg = CampaignConfig()
    cfg.acquisition.spectral.max_modes = 12
    menu._replace_campaign_config(cfg, loaded_from=None)

    spectral_menu = menu._BLOCK_MENUS_BY_LABEL["Edit acquisition.spectral"]
    optional_int_spec = next(
        spec
        for spec in spectral_menu.this_menu_options.fields
        if spec.path == "acquisition.spectral.max_modes"
    )

    monkeypatch.setattr("builtins.input", lambda prompt: "")
    menu._edit_field(optional_int_spec)
    assert menu.get_campaign_config().acquisition.spectral.max_modes == 12

    monkeypatch.setattr("builtins.input", lambda prompt: "8")
    menu._edit_field(optional_int_spec)
    assert menu.get_campaign_config().acquisition.spectral.max_modes == 8

    monkeypatch.setattr("builtins.input", lambda prompt: "null")
    menu._edit_field(optional_int_spec)
    assert menu.get_campaign_config().acquisition.spectral.max_modes is None


def test_backend_cpu_override_can_be_cleared_to_inherit(monkeypatch):
    import importlib

    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    cfg = CampaignConfig()
    cfg.resources.polus.cpus_per_task = 4
    menu._replace_campaign_config(cfg, loaded_from=None)

    polus_menu = menu._BLOCK_MENUS_BY_LABEL["Edit POLUS resources"]
    cpu_spec = next(
        spec
        for spec in polus_menu.this_menu_options.fields
        if spec.path == "resources.polus.cpus_per_task"
    )

    monkeypatch.setattr("builtins.input", lambda prompt: "null")
    menu._edit_field(cpu_spec)

    updated = menu.get_campaign_config()
    assert updated.resources.polus.cpus_per_task is None
    assert updated.resources.cpus_for("PHASE_A_POLUS") == "auto"


def test_top_three_roi_config_blocks_render_current_values():
    import importlib

    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    cfg = CampaignConfig()
    cfg.seed_selection.strategy = "d_optimal"
    cfg.campaign.sampling_aggressiveness = 7
    cfg.error_calibration.mode = "apply_to_acquisition"
    menu._replace_campaign_config(cfg, loaded_from=None)

    seed_rendered = menu._BLOCK_MENUS_BY_LABEL["Edit seed_selection"].this_menu_options()
    assert "seed_selection.strategy: d_optimal" in seed_rendered
    assert "seed_selection.d_optimal_pool_multiplier" in seed_rendered
    assert "seed_selection.d_optimal_score_power" in seed_rendered

    campaign_rendered = menu._BLOCK_MENUS_BY_LABEL["Edit campaign"].this_menu_options()
    assert "campaign.sampling_aggressiveness: 7" in campaign_rendered
    assert "campaign.custom_bootstrap: False" in campaign_rendered

    calibration_rendered = menu._BLOCK_MENUS_BY_LABEL[
        "Edit error_calibration"
    ].this_menu_options()
    assert "error_calibration.mode: apply_to_acquisition" in calibration_rendered
    assert "error_calibration.apply_strength" in calibration_rendered
    assert "error_calibration.model_version_policy: rolling_normalised" in calibration_rendered
    assert "error_calibration.aggressiveness_match_required: True" in calibration_rendered
    assert "seed_selection.d_optimal_degenerate_policy" in seed_rendered


def test_legacy_sequential_campaign_editors_are_neutralized():
    import importlib

    import pytest

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )

    with pytest.raises(RuntimeError, match="Sequential campaign config editors"):
        menu.EditCampaignConfigFunctions.edit_seed_selection()


def test_in_memory_sampling_protocol_summary_contains_top_three_roi_knobs(capsys, monkeypatch):
    import importlib

    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    cfg = CampaignConfig()
    cfg.seed_selection.strategy = "d_optimal"
    cfg.error_calibration.mode = "apply_to_acquisition"
    cfg.error_calibration.apply_strength = 0.5
    cfg.campaign.sampling_aggressiveness = 6
    menu._replace_campaign_config(cfg, loaded_from=None)
    monkeypatch.setattr(menu, "_pause", lambda: None)

    menu.EditCampaignConfigFunctions.show_sampling_protocol_summary()

    out = capsys.readouterr().out
    assert "Summary source: current in-memory editor config." in out
    assert "Sampling protocol summary" in out
    assert "campaign.sampling_aggressiveness: 6" in out
    assert "campaign.pool_path: pool.xyz (fixed campaign input)" in out
    assert "campaign.custom_bootstrap: False" in out
    assert "campaign.bootstrap_path: bootstrap/ (fixed campaign input)" in out
    assert "degenerate_policy=score_backfill" in out
    assert "error_calibration.model_version_policy: rolling_normalised" in out
    assert "sampling_protocol.size_normalised_trust_radius" in out
    protocol_lines = [
        line for line in out.splitlines()
        if "sampling_protocol." in line
    ]
    assert not any(
        "sampling_protocol.resolution: unavailable" in line
        for line in protocol_lines
    ), protocol_lines
    assert "sampling_protocol.geometry_scale_source: profile fallback preview" in out
    assert "sampling_protocol.scale_model" in out
    assert "sampling_protocol.scale_model.geometry_motion_scale" in out
    assert "sampling_protocol.scale_model.aligned_rmsd_scale" in out
    assert "sampling_protocol.scale_model.per_atom_mobility" in out
    assert "sampling_protocol.scale_model.pair_reference" in out
    assert "sampling_protocol.dimensionless_landing_gates" in out
    assert "sampling_protocol.resolved_movement_band" in out
    assert "sampling_protocol.resolved_phase_b" in out
    assert "sampling_protocol.resolved_safety" in out
    assert "sampling_protocol.resolved_quality_gates" in out
    assert "sampling_protocol.resolved_acquisition_risk" in out
    assert "sampling_protocol.resolved_ariadne" in out
    assert "seed_selection.strategy: d_optimal" in out
    assert "error_calibration.mode: apply_to_acquisition" in out
    assert "error_calibration.apply_strength: 0.5" in out
    assert "resources.aimall.effective_cpus_per_task: auto" in out
    assert "resources.effective_phase_walltimes" in out
    assert "FEREBUS=24h" in out
    assert "aimall.naat: auto" in out
    assert "aimall.boaq: auto" in out
    assert "aimall.iasmesh: fine" in out


def test_protocol_summary_reads_latest_structured_geometry_novelty_sidecar(
    tmp_path,
):
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.protocol_summary import (
        _latest_geometry_novelty_payload,
    )
    from ichor.hpc.active_learning.geometry_novelty import (
        GEOMETRY_NOVELTY_SCALE_SCHEMA_VERSION,
        write_geometry_novelty_scale,
    )
    from ichor.hpc.active_learning.layout import active_iteration_dir

    payload = {
        "schema_version": GEOMETRY_NOVELTY_SCALE_SCHEMA_VERSION,
        "iteration": 1,
        "scale_angstrom": 0.125,
    }
    write_geometry_novelty_scale(active_iteration_dir(tmp_path, 1), payload)

    observed, status = _latest_geometry_novelty_payload(tmp_path)

    assert status == "latest_sidecar"
    assert observed["iteration"] == 1
    assert observed["scale_angstrom"] == 0.125


def test_daemon_control_sampling_protocol_summary_uses_saved_campaign(tmp_path, monkeypatch, capsys):
    import importlib

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_menu"
    )
    cfg = CampaignConfig()
    cfg.seed_selection.strategy = "d_optimal"
    cfg.error_calibration.mode = "record_only"
    cfg.campaign.sampling_aggressiveness = 8
    cfg.to_yaml(tmp_path / "campaign.yaml")
    set_selected_campaign_dir(tmp_path)
    monkeypatch.setattr(menu, "user_input_free_flow", lambda *args, **kwargs: "")

    menu.DaemonControlFunctions.show_sampling_protocol_summary()

    out = capsys.readouterr().out
    assert "Summary source: saved campaign.yaml." in out
    assert "campaign.sampling_aggressiveness: 8" in out
    assert "sampling_protocol.scale_model" in out
    assert "sampling_protocol.scale_model.pair_reference" in out
    assert "sampling_protocol.resolved_phase_b" in out
    assert "sampling_protocol.resolved_safety" in out
    assert "sampling_protocol.resolved_ariadne" in out
    assert "sampling_protocol.resolved_manifest_example" in out
    assert "sampling_protocol.audit_manifest_example" in out
    assert "seed_selection.strategy: d_optimal" in out
    assert "campaign.custom_bootstrap: False" in out
    assert "error_calibration.mode: record_only" in out


def test_campaign_config_menu_covers_every_config_leaf():
    from dataclasses import fields, is_dataclass

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_submenus.edit_campaign_config_menu import (
        _BLOCK_MENUS_BY_LABEL,
    )
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_submenus.edit_campaign_config_submenus.edit_ariadne_block_submenu import (
        ARIADNE_FIELD_SPECS,
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    def leaf_paths(obj, prefix=""):
        if is_dataclass(obj):
            out = []
            for field in fields(obj):
                child = getattr(obj, field.name)
                path = field.name if not prefix else prefix + "." + field.name
                out.extend(leaf_paths(child, path))
            return out
        return [prefix]

    specs = []
    for block_menu in _BLOCK_MENUS_BY_LABEL.values():
        specs.extend(block_menu.this_menu_options.fields)
    spec_paths = [spec.path for spec in specs]
    spec_paths.extend("ariadne." + spec.path for spec in ARIADNE_FIELD_SPECS)

    assert len(spec_paths) == len(set(spec_paths))
    expected_editable = (
        set(leaf_paths(CampaignConfig()))
        - {
            "schema_version",
            "ferebus.prior_mean_type",
            "ferebus.property_scaling",
        }
    )
    actual_editable = {
        spec.path
        for spec in specs
        if not spec.read_only
    } | {"ariadne." + spec.path for spec in ARIADNE_FIELD_SPECS}
    read_only = {spec.path for spec in specs if spec.read_only}

    assert read_only == {
        "schema_version",
        "ferebus.prior_mean_type",
        "ferebus.property_scaling",
    }
    assert actual_editable == expected_editable
    quality_rendered = _BLOCK_MENUS_BY_LABEL["Edit quality_gates"].this_menu_options()
    assert "quality_gates.ariadne_max_displacement_ang" in quality_rendered
    assert "quality_gates.ariadne_min_pair_distance_ang" in quality_rendered


def test_campaign_config_block_menus_are_reachable():
    from consolemenu.items import SubmenuItem

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_submenus.edit_campaign_config_menu import (
        _ACQUISITION_BLOCK_LABELS,
        _BLOCK_MENUS_BY_LABEL,
        edit_acquisition_config_menu,
        edit_campaign_config_menu,
    )
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_submenus.edit_campaign_config_submenus.edit_ariadne_block_submenu import (
        EDIT_ARIADNE_BLOCK_MENU_DESCRIPTION,
    )

    top_level_submenus = {
        item.text
        for item in edit_campaign_config_menu.items
        if isinstance(item, SubmenuItem)
    }
    acquisition_submenus = {
        item.text
        for item in edit_acquisition_config_menu.items
        if isinstance(item, SubmenuItem)
    }
    reachable = set(top_level_submenus) | set(acquisition_submenus)

    assert "Edit acquisition" in top_level_submenus
    assert EDIT_ARIADNE_BLOCK_MENU_DESCRIPTION.title in top_level_submenus
    assert set(_ACQUISITION_BLOCK_LABELS).issubset(acquisition_submenus)
    assert set(_BLOCK_MENUS_BY_LABEL).issubset(reachable)


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
    assert "Set optimiser" in texts
    assert "Set max_iter" in texts
    assert "Set delta0" in texts
    assert "Set delta_max" in texts
    assert "Set trqn_backtransform_mode" in texts
    assert "Set trqn_geodesic_bt_mode" in texts
    assert "Set trqn_geodesic_dt" in texts
    assert "Set trqn_geodesic_tol" in texts
    assert "Set trqn_bt_ic_tol" in texts
    assert "Set trqn_max_backtransform_iter" in texts
    assert "Set trqn_trust_min" in texts
    assert "Edit trust radii" not in texts
    rendered = edit_ariadne_block_menu.this_menu_options()
    assert "delta0" in rendered
    assert "window: editable until ARIADNE consumes this iteration" in rendered


def test_journal_menu_items():
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_submenus.journal_menu import (
        journal_menu,
        journal_menu_options,
    )
    texts = [it.text for it in journal_menu.items]
    for expected in (
        "Set since timestamp",
        "Set event types",
        "Set last_n",
        "Set output mode",
        "Set verbose",
        "View matching events",
        "View all events",
        "View last N events",
        "List event types",
        "Clear journal filters",
    ):
        assert expected in texts, "missing item: " + expected
    journal_menu_options.since = "2026-05-23T00:00:00Z"
    journal_menu_options.event_types = ["sbatch", "phase_succeeded"]
    journal_menu_options.last_n = 17
    journal_menu_options.output_mode = "json"
    journal_menu_options.verbose = True
    rendered = journal_menu.this_menu_options()
    assert "since: 2026-05-23T00:00:00Z" in rendered
    assert "event_types: sbatch,phase_succeeded" in rendered
    assert "last_n: 17" in rendered
    assert "output_mode: json" in rendered
    assert "verbose: True" in rendered
    assert "scope: menu-only; applies to next journal view" in rendered


def test_journal_matching_events_uses_visible_filters(tmp_path, monkeypatch):
    import importlib
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.journal_menu"
    )
    import ichor.hpc.active_learning.cli as daemon_cli

    (tmp_path / "campaign.yaml").write_text("{}\n", encoding="utf-8")
    set_selected_campaign_dir(tmp_path)
    menu.journal_menu_options.since = "2026-05-23T00:00:00Z"
    menu.journal_menu_options.event_types = ["sbatch", "phase_succeeded"]
    menu.journal_menu_options.last_n = 12
    menu.journal_menu_options.output_mode = "raw"
    menu.journal_menu_options.verbose = True
    calls = []
    monkeypatch.setattr(menu, "user_input_free_flow", lambda *args, **kwargs: "")
    monkeypatch.setattr(daemon_cli, "cmd_journal", lambda ns: calls.append(ns) or 0)

    menu.JournalFunctions.view_matching_events()

    assert calls
    assert calls[0].campaign_dir == str(tmp_path)
    assert calls[0].since == "2026-05-23T00:00:00Z"
    assert calls[0].event_type == ["sbatch", "phase_succeeded"]
    assert calls[0].last_n == 12
    assert calls[0].raw is True
    assert calls[0].json is False
    assert calls[0].verbose is True


def test_journal_list_event_types_dispatches_cli(tmp_path, monkeypatch):
    import importlib
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.journal_menu"
    )
    import ichor.hpc.active_learning.cli as daemon_cli

    (tmp_path / "campaign.yaml").write_text("{}\n", encoding="utf-8")
    set_selected_campaign_dir(tmp_path)
    calls = []
    monkeypatch.setattr(menu, "user_input_free_flow", lambda *args, **kwargs: "")
    monkeypatch.setattr(daemon_cli, "cmd_journal", lambda ns: calls.append(ns) or 0)

    menu.JournalFunctions.list_event_types()

    assert calls
    assert calls[0].list_event_types is True


def test_journal_clear_filters_resets_visible_state():
    import importlib

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.journal_menu"
    )

    menu.journal_menu_options.since = "2026-05-23T00:00:00Z"
    menu.journal_menu_options.event_types = ["sbatch"]
    menu.journal_menu_options.last_n = 3
    menu.journal_menu_options.output_mode = "json"
    menu.journal_menu_options.verbose = True

    menu.JournalFunctions.clear_filters()

    assert menu.journal_menu_options.since == ""
    assert menu.journal_menu_options.event_types == []
    assert menu.journal_menu_options.last_n == 50
    assert menu.journal_menu_options.output_mode == "text"
    assert menu.journal_menu_options.verbose is False


def test_top_level_menu_registered_in_main_menu():
    from pathlib import Path

    main_menu_source = (
        Path(__file__).parents[1] / "ichor" / "cli" / "main_menu.py"
    ).read_text(encoding="utf-8")
    assert "ACTIVE_LEARNING_CAMPAIGN_MENU_DESCRIPTION" in main_menu_source
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
        mode="dry_run",
        poll_interval=15,
        max_ticks=5,
    )
    assert argv[1] == "-m"
    assert argv[2] == "ichor.hpc.active_learning.cli"
    assert "start" in argv
    assert argv[argv.index("--mode") + 1] == "dry_run"
    assert "--poll-interval" in argv
    assert "--max-ticks" in argv
    assert DetachedLaunchResult.__name__ == "DetachedLaunchResult"
    assert MENU_LAUNCHED_PID_FILENAME == "menu_launched.pid"
    assert MENU_LAUNCHED_LOG_FILENAME == "daemon.menu_launched.out"


def test_detached_launch_waits_for_readiness_before_publishing_pid(
    tmp_path,
    monkeypatch,
):
    import json
    from pathlib import Path

    import ichor.cli.useful_functions.launch_helpers as helpers

    class Child:
        pid = 2718

        @staticmethod
        def poll():
            return None

    def fake_popen(argv, log_path, *, env):
        Path(env["ICHOR_DAEMON_READINESS_PATH"]).write_text(
            json.dumps({"schema_version": 1, "ready": True, "pid": Child.pid}),
            encoding="utf-8",
        )
        return Child()

    monkeypatch.setattr(helpers, "_popen_detached", fake_popen)
    result = helpers.launch_daemon_detached_checked(
        tmp_path,
        mode="dry_run",
        startup_timeout_seconds=1.0,
    )

    assert result.ready is True
    assert result.exited_during_startup is False
    assert result.pid_path.read_text(encoding="utf-8") == "2718\n"


def test_detached_launch_timeout_does_not_publish_pid(tmp_path, monkeypatch):
    import ichor.cli.useful_functions.launch_helpers as helpers

    class Child:
        pid = 3141
        terminated = False

        @staticmethod
        def poll():
            return None

        def terminate(self):
            self.terminated = True

    child = Child()
    monkeypatch.setattr(
        helpers,
        "_popen_detached",
        lambda argv, log_path, *, env: child,
    )
    result = helpers.launch_daemon_detached_checked(
        tmp_path,
        mode="dry_run",
        startup_timeout_seconds=0.0,
    )

    assert result.ready is False
    assert result.exited_during_startup is True
    assert child.terminated is True
    assert not result.pid_path.exists()


def test_launch_helpers_builds_resume_argv_with_config(tmp_path):
    from ichor.cli.useful_functions.launch_helpers import build_daemon_argv

    cfg = tmp_path / "override.yaml"
    argv = build_daemon_argv(
        tmp_path,
        command="resume",
        mode="live",
        config=cfg,
    )
    assert argv[3] == "resume"
    assert argv[argv.index("--mode") + 1] == "live"
    assert argv[argv.index("--config") + 1] == str(cfg)


def test_launch_helpers_builds_explicit_completed_campaign_reopen_argv(tmp_path):
    from ichor.cli.useful_functions.launch_helpers import build_daemon_argv

    argv = build_daemon_argv(
        tmp_path,
        command="resume",
        mode="live",
        reopen_converged=True,
    )

    assert argv[3] == "resume"
    assert "--reopen-converged" in argv


def test_launch_helpers_builds_stop_request_cancellation_argv(tmp_path):
    from ichor.cli.useful_functions.launch_helpers import build_daemon_argv

    argv = build_daemon_argv(
        tmp_path,
        command="resume",
        mode="live",
        cancel_stop_request=True,
    )

    assert "--cancel-stop-request" in argv

    import pytest

    with pytest.raises(ValueError, match="resume"):
        build_daemon_argv(
            tmp_path,
            command="start",
            mode="live",
            cancel_stop_request=True,
        )


def test_launch_helpers_rejects_completed_campaign_reopen_with_start(tmp_path):
    import pytest

    from ichor.cli.useful_functions.launch_helpers import build_daemon_argv

    with pytest.raises(ValueError, match="resume"):
        build_daemon_argv(
            tmp_path,
            command="start",
            mode="live",
            reopen_converged=True,
        )


def test_launch_helpers_rejects_unknown_mode():
    import pytest

    from ichor.cli.useful_functions.launch_helpers import build_daemon_argv
    from pathlib import Path

    with pytest.raises(ValueError):
        build_daemon_argv(Path("."), mode="bogus")

    with pytest.raises(ValueError):
        build_daemon_argv(Path("."), command="bogus", mode="dry_run")


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


def test_prologue_call_refreshes_summary(tmp_path, monkeypatch):
    """Calling MenuOptions() must refresh the summary before stringifying."""
    import ichor.cli.global_menu_variables as globals_
    import ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context as campaign_context
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_menu import (
        active_learning_campaign_menu_options,
    )

    campaign = tmp_path / "call_campaign"
    state_path = campaign / ".DATA" / "ACTIVE_LEARNING" / "state.json"
    _make_state_json(state_path, phase="FEREBUS", iteration=11, max_iterations=20)

    original = active_learning_campaign_menu_options.selected_active_learning_campaign_directory_path
    try:
        monkeypatch.setattr(campaign_context, "_explicit_selection", True)
        monkeypatch.setattr(
            globals_,
            "SELECTED_ACTIVE_LEARNING_CAMPAIGN_DIRECTORY_PATH",
            campaign,
        )
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
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    from ichor.hpc.active_learning.config import CampaignConfig

    (tmp_path / "campaign.yaml").write_text("{}\n", encoding="utf-8")
    set_selected_campaign_dir(tmp_path)
    original = CampaignConfig()
    original.system_name = "WATER"
    original.resources.defaults.walltime_hours = 12
    original.resources.defaults.mem_per_cpu = "5G"
    original.resources.polus.cpus_per_task = 4
    original.resources.ferebus.cpus_per_task = 4
    original.resources.gaussian.cpus_per_task = 2
    original.resources.aimall.cpus_per_task = 6
    original.resources.ariadne.cpus_per_task = 4
    original.resources.gradient_parallel_backend = "serial"
    original.gaussian.method = "B3LYP"
    original.gaussian.basis_set = "def2-TZVP"
    original.resources.gaussian.link0_mem = "6GB"
    original.ferebus.properties = ["iqa", "q00"]
    original.ferebus.feature_scaling = False
    original.acquisition.allow_uniform_posterior_fallback = True
    original.to_yaml(tmp_path / "campaign.yaml")
    monkeypatch.setattr(menu, "_pause", lambda: None)

    assert menu.load_config_for_campaign_dir(tmp_path, quiet=True, prompt_if_dirty=False)
    menu.get_campaign_config().system_name = "WATER_AL"
    menu.EditCampaignConfigFunctions.save_to_disk()

    reloaded = CampaignConfig.from_yaml(tmp_path / "campaign.yaml")
    assert reloaded.system_name == "WATER_AL"
    assert reloaded.resources.defaults.walltime_hours == 12
    assert reloaded.resources.polus_cpus_per_task == 4
    assert reloaded.resources.ferebus_cpus_per_task == 4
    assert reloaded.resources.gaussian_cpus_per_task == 2
    assert reloaded.resources.aimall_cpus_per_task == 6
    assert reloaded.resources.ariadne_cpus_per_task == 4
    assert reloaded.resources.gaussian_link0_mem == "6GB"
    assert reloaded.resources.gradient_parallel_backend == "serial"
    assert reloaded.gaussian.method == "B3LYP"
    assert reloaded.ferebus.properties == ["iqa", "q00"]
    assert reloaded.ferebus.feature_scaling is False
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


def test_preflight_menu_dispatches_campaign_aware_command(
    tmp_path,
    monkeypatch,
):
    import importlib
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_menu"
    )
    import ichor.hpc.active_learning.cli as daemon_cli

    (tmp_path / "campaign.yaml").write_text("{}\n", encoding="utf-8")
    set_selected_campaign_dir(tmp_path)
    seen = {}

    def fake_preflight(ns):
        seen["campaign_dir"] = ns.campaign_dir
        seen["json"] = ns.json
        seen["verbose"] = ns.verbose
        return 0

    monkeypatch.setattr(daemon_cli, "cmd_preflight", fake_preflight)
    monkeypatch.setattr(menu, "user_input_free_flow", lambda *args, **kwargs: "")

    menu.DaemonControlFunctions.preflight_backends()

    assert seen == {
        "campaign_dir": str(tmp_path.absolute()),
        "json": False,
        "verbose": True,
    }


def test_submitted_environment_smoke_menu_requires_confirmation(
    tmp_path, monkeypatch,
):
    import importlib

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_menu"
    )
    import ichor.hpc.active_learning.cli as daemon_cli

    (tmp_path / "campaign.yaml").write_text("{}\n", encoding="utf-8")
    set_selected_campaign_dir(tmp_path)
    calls = []
    responses = iter(["YES", ""])
    monkeypatch.setattr(
        menu,
        "user_input_free_flow",
        lambda *args, **kwargs: next(responses),
    )
    monkeypatch.setattr(
        daemon_cli,
        "cmd_preflight",
        lambda ns: calls.append(ns) or 0,
    )

    menu.DaemonControlFunctions.submitted_environment_smoke()

    assert len(calls) == 1
    assert calls[0].campaign_dir == str(tmp_path.absolute())
    assert calls[0].submit_environment_smoke is True
    assert calls[0].verbose is True


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

    (tmp_path / "campaign.yaml").write_text("{}\n", encoding="utf-8")
    set_selected_campaign_dir(tmp_path)
    calls = []
    monkeypatch.setattr(menu, "user_input_free_flow", lambda *args, **kwargs: "YES")
    monkeypatch.setattr(daemon_cli, "cmd_reconcile", lambda ns: calls.append(ns.allow_fresh_init) or 0)

    menu.DaemonControlFunctions.reconcile()
    menu.DaemonControlFunctions.reconcile_allow_fresh_init()

    assert calls == [False, True]


def test_init_pool_passes_verbatim_import_options(tmp_path, monkeypatch):
    import importlib
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_submenus."
        "import_trajectory_pool_submenu"
    )
    import ichor.hpc.active_learning.cli as daemon_cli

    set_selected_campaign_dir(tmp_path)
    menu.import_trajectory_pool_menu_options.source = "pool.xyz"
    menu.import_trajectory_pool_menu_options.force_reimport = False
    rendered = menu.import_trajectory_pool_menu.this_menu_options()
    assert "scope: menu-only; applies to next init/import command" in rendered
    assert "source: pool.xyz" in rendered
    calls = []
    monkeypatch.setattr(menu, "user_input_free_flow", lambda *args, **kwargs: "")
    monkeypatch.setattr(daemon_cli, "cmd_init", lambda ns: calls.append(ns) or 0)

    menu.ImportTrajectoryPoolFunctions.run_import()

    assert calls
    assert calls[0].campaign_dir == str(tmp_path)
    assert calls[0].source == "pool.xyz"
    assert calls[0].force is False
    assert calls[0].yes is False


def test_init_pool_force_requires_confirmation_and_resets(tmp_path, monkeypatch):
    import importlib
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )
    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_submenus."
        "import_trajectory_pool_submenu"
    )
    import ichor.hpc.active_learning.cli as daemon_cli

    set_selected_campaign_dir(tmp_path)
    menu.import_trajectory_pool_menu_options.source = "pool.xyz"
    menu.import_trajectory_pool_menu_options.force_reimport = True
    calls = []
    responses = iter(["YES", ""])
    monkeypatch.setattr(menu, "user_input_free_flow", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(daemon_cli, "cmd_init", lambda ns: calls.append(ns) or 0)

    menu.ImportTrajectoryPoolFunctions.run_import()

    assert calls
    assert calls[0].force is True
    assert menu.import_trajectory_pool_menu_options.force_reimport is False


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
    menu.start_daemon_foreground_menu_options.selected_mode = "dry_run"
    menu.start_daemon_foreground_menu_options.selected_config = ""
    menu.start_daemon_foreground_menu_options.selected_poll_interval = 3
    menu.start_daemon_foreground_menu_options.selected_max_ticks = 2
    menu.start_daemon_foreground_menu_options.reopen_converged = True
    menu.start_daemon_foreground_menu_options.cancel_stop_request = True
    rendered = menu.start_daemon_foreground_menu.this_menu_options()
    assert "command: resume" in rendered
    assert "poll_interval: 3" in rendered
    assert "max_ticks: 2" in rendered
    assert "reopen_converged: True" in rendered
    assert "cancel_stop_request: True" in rendered
    assert "scope: menu-only; applies to next daemon launch" in rendered
    calls = []
    monkeypatch.setattr(menu, "user_input_free_flow", lambda *args, **kwargs: "")
    monkeypatch.setattr(daemon_cli, "cmd_start", lambda ns: calls.append(("start", ns)) or 0)
    monkeypatch.setattr(daemon_cli, "cmd_resume", lambda ns: calls.append(("resume", ns)) or 0)

    menu.StartDaemonForegroundFunctions.launch()

    assert calls[0][0] == "resume"
    ns = calls[0][1]
    assert ns.campaign_dir == str(tmp_path)
    assert ns.mode == "dry_run"
    assert ns.poll_interval == 3
    assert ns.max_ticks == 2
    assert ns.reopen_converged is True
    assert ns.cancel_stop_request is True


def test_foreground_launch_rejects_completed_campaign_reopen_with_start(
    tmp_path, monkeypatch, capsys,
):
    import importlib

    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
        set_selected_campaign_dir,
    )

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.daemon_control_submenus."
        "start_daemon_foreground_submenu"
    )
    set_selected_campaign_dir(tmp_path)
    menu.start_daemon_foreground_menu_options.selected_command = "start"
    menu.start_daemon_foreground_menu_options.selected_mode = "dry_run"
    menu.start_daemon_foreground_menu_options.selected_config = ""
    menu.start_daemon_foreground_menu_options.reopen_converged = True
    menu.start_daemon_foreground_menu_options.cancel_stop_request = False
    monkeypatch.setattr(menu, "user_input_free_flow", lambda *args, **kwargs: "")

    menu.StartDaemonForegroundFunctions.launch()

    assert "valid only with the resume command" in capsys.readouterr().out


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
    menu.start_daemon_background_menu_options.selected_poll_interval = 0
    menu.start_daemon_background_menu_options.selected_max_ticks = 0
    menu.start_daemon_background_menu_options.reopen_converged = True
    menu.start_daemon_background_menu_options.cancel_stop_request = True
    menu.start_daemon_background_menu_options.selected_log_path = str(
        tmp_path / "daemon.custom.out"
    )
    menu.start_daemon_background_menu_options.selected_pid_path = str(
        tmp_path / "daemon.custom.pid"
    )
    rendered = menu.start_daemon_background_menu.this_menu_options()
    assert "poll_interval: 0 (campaign default)" in rendered
    assert "max_ticks: 0 (unlimited)" in rendered
    assert "config: <blank>" in rendered
    assert "background_log: " in rendered
    assert "background_pid: " in rendered
    assert "reopen_converged: True" in rendered
    assert "cancel_stop_request: True" in rendered
    assert "scope: menu-only; applies to next daemon launch" in rendered
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
    assert seen["log_path"] == Path(tmp_path / "daemon.custom.out")
    assert seen["pid_path"] == Path(tmp_path / "daemon.custom.pid")
    assert seen["reopen_converged"] is True
    assert seen["cancel_stop_request"] is True
    assert "exited during startup" in capsys.readouterr().out
