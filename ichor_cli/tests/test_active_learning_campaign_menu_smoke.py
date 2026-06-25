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
        "Show sampling protocol summary",
        "Preflight backends",
        "Import Trajectory Pool",
        "Start/Resume Daemon (Foreground)",
        "Start/Resume Daemon (Background)",
        "Stop daemon",
        "Reconcile state",
        "Reconcile state with --allow-fresh-init",
    ):
        assert expected in texts, "missing item: " + expected


def test_edit_campaign_config_menu_items():
    from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_submenus.edit_campaign_config_menu import (
        edit_acquisition_config_menu,
        edit_campaign_config_menu,
        get_campaign_config,
    )
    texts = [it.text for it in edit_campaign_config_menu.items]
    for expected in (
        "Show current in-memory config",
        "Show sampling protocol summary",
        "Load from disk",
        "Reset to defaults",
        "Validate current config",
        "Edit campaign identity",
        "Edit trajectory_pool",
        "Edit iteration control",
        "Edit initial sub-sample sizes",
        "Edit resources",
        "Edit Gaussian block",
        "Edit batch_sizing",
        "Edit seed_selection",
        "Edit anti_overlap",
        "Edit phase_b",
        "Edit split",
        "Edit FEREBUS block",
        "Edit robustness",
        "Edit acquisition",
        "Edit ARIADNE Block",
        "Edit stop",
        "Edit outlier_filter",
        "Edit adversarial_safety",
        "Edit error_calibration",
        "Edit quality_gates",
        "Edit runtime",
        "Show unsaved changes",
        "Show config lock review",
        "Show pending YAML diff",
        "Discard unsaved changes / reload from disk",
        "Export dense config snapshot",
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
        "Edit acquisition.movement_band",
        "Edit acquisition.movement_utility",
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


def test_campaign_config_block_submenus_show_values_and_edit_one_field(monkeypatch):
    import importlib

    import ichor.cli.main_menu_submenus.active_learning_campaign_menu.field_menu as field_menu
    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    menu._replace_campaign_config(CampaignConfig(), loaded_from=None)

    resources_menu = menu._BLOCK_MENUS_BY_LABEL["Edit resources"]
    rendered = resources_menu.this_menu_options()
    assert "resources.partition" in rendered
    assert "resources.default_walltime_hours" in rendered
    assert "resources.ferebus_walltime_hours" in rendered
    assert "resources.polus_cpus_per_task" in rendered
    assert "resources.aimall_cpus_per_task" in rendered
    assert "resources.gaussian_mem_per_cpu" in rendered
    assert "resources.gradient_parallel_backend" in rendered

    texts = [it.text for it in resources_menu.items]
    assert "Set partition" in texts
    assert "Set default_walltime_hours" in texts
    assert "Set ferebus_walltime_hours" in texts
    assert "Set aimall_cpus_per_task" in texts

    cfg = menu.get_campaign_config()
    old_partition = cfg.resources.partition
    spec = next(
        spec
        for spec in resources_menu.this_menu_options.fields
        if spec.path == "resources.default_walltime_hours"
    )
    monkeypatch.setattr(field_menu, "user_input_int", lambda prompt, default: 37)

    menu._edit_field(spec)

    assert cfg.resources.default_walltime_hours == 37
    assert cfg.resources.partition == old_partition
    assert "resources.default_walltime_hours: 37" in resources_menu.this_menu_options()


def test_edit_gaussian_basis_set_saves_to_selected_campaign_yaml(tmp_path, monkeypatch):
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

    menu.EditCampaignConfigFunctions.save_to_disk()

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
    cfg.gaussian.basis_set = "def2-SVP"
    cfg.to_yaml(tmp_path / "campaign.yaml")

    monkeypatch.setattr(top, "user_input_path", lambda prompt, default_path: tmp_path)
    monkeypatch.setattr(menu, "user_input_free_flow", lambda *args, **kwargs: "YES")

    top.ActiveLearningCampaignFunctions.select_campaign_directory()

    assert menu.get_campaign_config().gaussian.basis_set == "def2-SVP"
    assert str(tmp_path / "campaign.yaml") == menu.edit_campaign_config_menu_options.loaded_from
    assert not menu.has_unsaved_config_changes()


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
    cfg2.gaussian.basis_set = "def2-SVP"
    cfg2.to_yaml(c2 / "campaign.yaml")

    assert menu.load_config_for_campaign_dir(c1, quiet=True)
    assert menu.get_campaign_config().gaussian.basis_set == "6-31+G(d,p)"

    assert menu.load_config_for_campaign_dir(c2, quiet=True)
    assert menu.get_campaign_config().gaussian.basis_set == "def2-SVP"


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
    menu._set_config_value("split.train_fraction", 0.9)
    menu._set_config_value("split.val_mid_fraction", 0.9)

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
    original.gaussian.basis_set = "def2-SVP"
    original.to_yaml(tmp_path / "campaign.yaml")
    original_text = (tmp_path / "campaign.yaml").read_text(encoding="utf-8")
    set_selected_campaign_dir(tmp_path)
    assert menu.load_config_for_campaign_dir(tmp_path, quiet=True)
    monkeypatch.setattr(menu, "_pause", lambda: None)
    real_from_yaml = menu.CampaignConfig.from_yaml

    def fail_reload(path):
        if ".menu-save." in str(path):
            raise RuntimeError("synthetic reload failure")
        return real_from_yaml(path)

    menu._set_config_value("gaussian.basis_set", "6-31+G(d,p)")
    monkeypatch.setattr(menu.CampaignConfig, "from_yaml", fail_reload)

    menu.EditCampaignConfigFunctions.save_to_disk()

    assert (tmp_path / "campaign.yaml").read_text(encoding="utf-8") == original_text
    loaded = real_from_yaml(tmp_path / "campaign.yaml")
    assert loaded.gaussian.basis_set == "def2-SVP"
    assert menu.has_unsaved_config_changes()
    assert "Reload before save failed" in menu.edit_campaign_config_menu_options.last_error
    assert not list(tmp_path.glob("*.menu-save.*.tmp"))


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
    cfg.gaussian.basis_set = "def2-SVP"
    cfg.to_yaml(tmp_path / "campaign.yaml")
    set_selected_campaign_dir(tmp_path)
    assert menu.load_config_for_campaign_dir(tmp_path, quiet=True)

    menu._set_config_value("gaussian.basis_set", "6-31+G(d,p)")
    assert "gaussian.basis_set" in menu.dirty_paths()

    menu._set_config_value("gaussian.basis_set", "def2-SVP")

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
    spec = field_menu.spec("resources.aimall_cpus_per_task", "str", transform=int)
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
    menu._set_config_value("resources.gaussian_walltime_hours", 3)
    monkeypatch.setattr(menu, "_pause", lambda: None)

    menu.EditCampaignConfigFunctions.save_to_disk()

    loaded = CampaignConfig.from_yaml(tmp_path / "campaign.yaml")
    assert loaded.resources.gaussian_walltime_hours == 3
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
    changed.gaussian.method = "PBE0"
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
    override.gaussian.method = "PBE0"
    override_path = tmp_path / "override.yaml"
    override.to_yaml(override_path)
    set_selected_campaign_dir(tmp_path)
    edit_menu.load_config_for_campaign_dir(tmp_path, quiet=True)
    start_menu.start_daemon_foreground_menu_options.selected_command = "resume"
    start_menu.start_daemon_foreground_menu_options.selected_mode = "dry-run"
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
    changed_campaign.gaussian.method = "PBE0"
    changed_campaign.to_yaml(tmp_path / "campaign.yaml")
    set_selected_campaign_dir(tmp_path)
    edit_menu.load_config_for_campaign_dir(tmp_path, quiet=True)
    start_menu.start_daemon_foreground_menu_options.selected_command = "resume"
    start_menu.start_daemon_foreground_menu_options.selected_mode = "dry-run"
    start_menu.start_daemon_foreground_menu_options.selected_config = str(clean_override_path)
    start_menu.start_daemon_foreground_menu_options.selected_preset = ""
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
    override.gaussian.method = "PBE0"
    override_path = tmp_path / "override.yaml"
    override.to_yaml(override_path)
    set_selected_campaign_dir(tmp_path)
    edit_menu.load_config_for_campaign_dir(tmp_path, quiet=True)
    start_menu.start_daemon_background_menu_options.selected_command = "resume"
    start_menu.start_daemon_background_menu_options.selected_mode = "dry-run"
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
    field_specs = {spec.path: spec for spec in rendered.fields}

    assert set(field_specs) == {
        "acquisition.gradient.mode",
        "acquisition.gradient.cartesian_step",
        "acquisition.gradient.active_step",
        "acquisition.gradient.regularization",
        "acquisition.gradient.cartesian_step_floor",
        "acquisition.gradient.ghost_mass_threshold",
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
    field_specs = {spec.path: spec for spec in rendered.fields}

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


def test_top_three_roi_config_blocks_render_current_values():
    import importlib

    from ichor.hpc.active_learning.config import CampaignConfig

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.edit_campaign_config_menu"
    )
    cfg = CampaignConfig()
    cfg.seed_selection.strategy = "d_optimal"
    cfg.error_calibration.mode = "apply_to_acquisition"
    cfg.acquisition.spectral.max_modes = 9
    cfg.acquisition.calibrated_energy.band_low_ha = 0.001
    cfg.acquisition.fullspace_confinement.lambda_residual = 0.75
    menu._replace_campaign_config(cfg, loaded_from=None)

    seed_rendered = menu._BLOCK_MENUS_BY_LABEL["Edit seed_selection"].this_menu_options()
    assert "seed_selection.strategy: d_optimal" in seed_rendered
    assert "seed_selection.d_optimal_pool_multiplier" in seed_rendered
    assert "seed_selection.d_optimal_score_power" in seed_rendered

    calibration_rendered = menu._BLOCK_MENUS_BY_LABEL[
        "Edit error_calibration"
    ].this_menu_options()
    assert "error_calibration.mode: apply_to_acquisition" in calibration_rendered
    assert "error_calibration.apply_strength" in calibration_rendered

    spectral_rendered = menu._BLOCK_MENUS_BY_LABEL[
        "Edit acquisition.spectral"
    ].this_menu_options()
    assert "acquisition.spectral.max_modes: 9" in spectral_rendered

    energy_rendered = menu._BLOCK_MENUS_BY_LABEL[
        "Edit acquisition.calibrated_energy"
    ].this_menu_options()
    assert "acquisition.calibrated_energy.band_low_ha: 0.001" in energy_rendered

    fullspace_rendered = menu._BLOCK_MENUS_BY_LABEL[
        "Edit acquisition.fullspace_confinement"
    ].this_menu_options()
    assert "acquisition.fullspace_confinement.lambda_residual: 0.75" in fullspace_rendered


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
    cfg.acquisition.spectral.mode = "blend"
    cfg.acquisition.calibrated_energy.utility = "banded"
    cfg.acquisition.fullspace_confinement.enabled = True
    cfg.acquisition.driver.enabled = True
    menu._replace_campaign_config(cfg, loaded_from=None)
    monkeypatch.setattr(menu, "_pause", lambda: None)

    menu.EditCampaignConfigFunctions.show_sampling_protocol_summary()

    out = capsys.readouterr().out
    assert "Sampling protocol summary" in out
    assert "seed_selection.strategy: d_optimal" in out
    assert "error_calibration.mode: apply_to_acquisition" in out
    assert "error_calibration.apply_strength: 0.5" in out
    assert "acquisition.spectral.mode: blend" in out
    assert "acquisition.calibrated_energy.utility: banded" in out
    assert "acquisition.fullspace_confinement.enabled: True" in out
    assert "acquisition.driver.enabled: True" in out
    assert "acquisition.driver.objective: cheap_driver" in out
    assert "acquisition.stencils.weak_mode_gating_enabled: True" in out
    assert "acquisition.stencils.weak_mode_omega_band" in out
    assert "acquisition.stencils.anharmonic_caps" in out
    assert "resources.aimall_cpus_per_task: 8" in out
    assert "resources.effective_phase_walltimes" in out
    assert "FEREBUS=24h" in out
    assert "aimall.nproc: 8" in out
    assert "aimall.naat: auto" in out
    assert "aimall.boaq: auto" in out
    assert "aimall.iasmesh: fine" in out
    assert "ariadne.trqn_backtransform_mode: geodesic" in out
    assert "ariadne.trqn_geodesic_bt_mode: dense" in out
    assert "ariadne.trqn_backtransform_numerics" in out
    assert "max_iter=50" in out


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
    cfg.acquisition.spectral.lambda_spectral = 2.5
    cfg.acquisition.driver.enabled = True
    cfg.to_yaml(tmp_path / "campaign.yaml")
    set_selected_campaign_dir(tmp_path)
    monkeypatch.setattr(menu, "user_input_free_flow", lambda *args, **kwargs: "")

    menu.DaemonControlFunctions.show_sampling_protocol_summary()

    out = capsys.readouterr().out
    assert "seed_selection.strategy: d_optimal" in out
    assert "error_calibration.mode: record_only" in out
    assert "acquisition.spectral.lambda_spectral: 2.5" in out
    assert "acquisition.driver.enabled: True" in out
    assert "acquisition.stencils.weak_mode_gating_enabled: True" in out
    assert "ariadne.trqn_backtransform_mode: geodesic" in out
    assert "ariadne.trqn_geodesic_bt_mode: dense" in out
    assert "ariadne.trqn_backtransform_numerics" in out


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
    expected_editable = set(leaf_paths(CampaignConfig())) - {"schema_version"}
    actual_editable = {
        spec.path
        for spec in specs
        if not spec.read_only
    } | {"ariadne." + spec.path for spec in ARIADNE_FIELD_SPECS}
    read_only = {spec.path for spec in specs if spec.read_only}

    assert read_only == {"schema_version"}
    assert actual_editable == expected_editable


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
        "View matching events",
        "View all events",
        "View last N events",
        "Clear journal filters",
    ):
        assert expected in texts, "missing item: " + expected
    journal_menu_options.since = "2026-05-23T00:00:00Z"
    journal_menu_options.event_types = ["sbatch", "phase_succeeded"]
    journal_menu_options.last_n = 17
    rendered = journal_menu.this_menu_options()
    assert "since: 2026-05-23T00:00:00Z" in rendered
    assert "event_types: sbatch,phase_succeeded" in rendered
    assert "last_n: 17" in rendered


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

    set_selected_campaign_dir(tmp_path)
    menu.journal_menu_options.since = "2026-05-23T00:00:00Z"
    menu.journal_menu_options.event_types = ["sbatch", "phase_succeeded"]
    calls = []
    monkeypatch.setattr(menu, "user_input_free_flow", lambda *args, **kwargs: "")
    monkeypatch.setattr(daemon_cli, "cmd_journal", lambda ns: calls.append(ns) or 0)

    menu.JournalFunctions.view_matching_events()

    assert calls
    assert calls[0].campaign_dir == str(tmp_path)
    assert calls[0].since == "2026-05-23T00:00:00Z"
    assert calls[0].event_type == ["sbatch", "phase_succeeded"]


def test_journal_clear_filters_resets_visible_state():
    import importlib

    menu = importlib.import_module(
        "ichor.cli.main_menu_submenus.active_learning_campaign_menu."
        "active_learning_campaign_submenus.journal_menu"
    )

    menu.journal_menu_options.since = "2026-05-23T00:00:00Z"
    menu.journal_menu_options.event_types = ["sbatch"]
    menu.journal_menu_options.last_n = 3

    menu.JournalFunctions.clear_filters()

    assert menu.journal_menu_options.since == ""
    assert menu.journal_menu_options.event_types == []
    assert menu.journal_menu_options.last_n == 50


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
            "aimall_cpus_per_task": 6,
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
            "property_name": "iqa",
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
    assert reloaded.resources.default_walltime_hours == 12
    assert reloaded.resources.polus_cpus_per_task == 4
    assert reloaded.resources.ferebus_cpus_per_task == 4
    assert reloaded.resources.gaussian_cpus_per_task == 2
    assert reloaded.resources.aimall_cpus_per_task == 6
    assert reloaded.resources.ariadne_cpus_per_task == 4
    assert reloaded.resources.gaussian_link0_mem == "6GB"
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
        "active_learning_campaign_submenus.daemon_control_submenus."
        "import_trajectory_pool_submenu"
    )
    import ichor.hpc.active_learning.cli as daemon_cli

    set_selected_campaign_dir(tmp_path)
    menu.import_trajectory_pool_menu_options.source_path = "pool.xyz"
    menu.import_trajectory_pool_menu_options.no_outlier_filter = True
    menu.import_trajectory_pool_menu_options.force_reimport = False
    calls = []
    monkeypatch.setattr(menu, "user_input_free_flow", lambda *args, **kwargs: "")
    monkeypatch.setattr(daemon_cli, "cmd_import_pool", lambda ns: calls.append(ns) or 0)

    menu.ImportTrajectoryPoolFunctions.run_import()

    assert calls
    assert calls[0].campaign_dir == str(tmp_path)
    assert calls[0].source == "pool.xyz"
    assert calls[0].no_outlier_filter is True
    assert calls[0].force is False


def test_import_pool_force_requires_confirmation_and_resets(tmp_path, monkeypatch):
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
    menu.import_trajectory_pool_menu_options.source_path = "pool.xyz"
    menu.import_trajectory_pool_menu_options.no_outlier_filter = False
    menu.import_trajectory_pool_menu_options.force_reimport = True
    calls = []
    responses = iter(["YES", ""])
    monkeypatch.setattr(menu, "user_input_free_flow", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(daemon_cli, "cmd_import_pool", lambda ns: calls.append(ns) or 0)

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
    menu.start_daemon_foreground_menu_options.selected_mode = "dry-run"
    menu.start_daemon_foreground_menu_options.selected_config = ""
    menu.start_daemon_foreground_menu_options.selected_preset = "csf4"
    menu.start_daemon_foreground_menu_options.selected_poll_interval = 3
    menu.start_daemon_foreground_menu_options.selected_max_ticks = 2
    rendered = menu.start_daemon_foreground_menu.this_menu_options()
    assert "command: resume" in rendered
    assert "poll_interval: 3" in rendered
    assert "max_ticks: 2" in rendered
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
    rendered = menu.start_daemon_background_menu.this_menu_options()
    assert "poll_interval: 0 (campaign default)" in rendered
    assert "max_ticks: 0 (unlimited)" in rendered
    assert "config: <blank>" in rendered
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
