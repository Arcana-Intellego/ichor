"""Edit campaign config menu -- current active-learning schema.

Field-grouped editor over a single CampaignConfig instance held in module
state. The grouping follows the current CampaignConfig block structure: one
menu item per top-level block plus a separate submenu tree for the deeper
acquisition surface.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ichor.cli.global_menu_variables
from consolemenu.items import FunctionItem, SubmenuItem
from ichor.cli.console_menu import ConsoleMenu, add_items_to_menu
from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
    CampaignSelectionError,
    print_campaign_selection_error,
    selected_campaign_dir,
    selected_campaign_dir_or_none,
)
from ichor.cli.main_menu_submenus.active_learning_campaign_menu.field_menu import (
    FieldSpec as _FieldSpec,
    edit_field as _shared_edit_field,
    format_field_value,
    get_attr_path,
    make_field_menu,
    set_attr_path,
    spec as _shared_spec,
)
from ichor.cli.main_menu_submenus.active_learning_campaign_menu.protocol_summary import (
    format_sampling_protocol_summary,
)
from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_submenus.edit_campaign_config_submenus import (
    edit_ariadne_block_menu,
    EDIT_ARIADNE_BLOCK_MENU_DESCRIPTION,
)
from ichor.cli.menu_description import MenuDescription
from ichor.cli.menu_options import MenuOptions
from ichor.cli.useful_functions import user_input_free_flow
from ichor.hpc.active_learning.config import (
    CampaignConfig,
    ConfigValidationError,
    VALID_AIMALL_BOAQ_VALUES,
    VALID_AIMALL_IASMESH_VALUES,
    VALID_ACQUISITION_DRIVER_GRADIENT_BACKENDS,
    VALID_ACQUISITION_DRIVER_OBJECTIVES,
    VALID_CALIBRATED_ENERGY_UTILITIES,
    VALID_DESCRIPTORS,
    VALID_ERROR_CALIBRATION_MODES,
    VALID_ERROR_CALIBRATION_MODEL_VERSION_POLICIES,
    VALID_GAUSSIAN_MEMORY_MODES,
    VALID_GEOMETRY_NOVELTY_SCALE_SOURCES,
    VALID_GEOMETRY_NOVELTY_STATISTICS,
    VALID_GRADIENT_MODES,
    VALID_GRADIENT_PARALLEL_BACKENDS,
    VALID_MODE_WEIGHTING_POLICIES,
    VALID_NEGATIVE_CURVATURE_POLICIES,
    VALID_D_OPTIMAL_DEGENERATE_POLICIES,
    VALID_SEED_SELECTION_STRATEGIES,
    VALID_SIZE_NORMALISATION_BARRIER_MODES,
    VALID_SIZE_NORMALISATION_DISTANCE_MODES,
    VALID_SIZE_NORMALISATION_ENERGY_MODES,
    VALID_SPECTRAL_MODES,
)
from ichor.hpc.active_learning.ferebus_prior import SUPPORTED_LEVELS


EDIT_CAMPAIGN_CONFIG_MENU_DESCRIPTION = MenuDescription(
    "Edit Campaign Config Menu",
    subtitle=(
        "Edit campaign.yaml block-by-block. Changes accumulate in memory; "
        "use Save to disk to validate and write the yaml. Load from disk "
        "discards pending edits and re-reads the file.\n"
    ),
)


_campaign_config: CampaignConfig = CampaignConfig()
_loaded_from_path: Optional[Path] = None
_editor_selected_campaign_dir: Optional[Path] = None
_dirty_paths: set[str] = set()
_loaded_fingerprint: Optional[str] = None
_loaded_snapshot: Optional[dict] = None
_last_save_path: Optional[Path] = None
_last_error: str = ""
_last_config_lock_error: str = ""
_last_saved_config_lock_error: str = ""


def _log_info(message: str) -> None:
    try:
        import ichor.hpc.global_variables as global_variables

        global_variables.LOGGER.info(message)
    except Exception:
        pass


def _config_fingerprint(cfg: CampaignConfig) -> str:
    payload = cfg.to_dict()
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_campaign_config() -> CampaignConfig:
    return _campaign_config


def _replace_campaign_config(
    cfg,
    loaded_from,
    *,
    selected_dir: Optional[Path] = None,
    clear_dirty: bool = True,
):
    global _campaign_config, _loaded_from_path, _editor_selected_campaign_dir
    _campaign_config = cfg
    _loaded_from_path = Path(loaded_from) if loaded_from is not None else None
    if selected_dir is not None:
        _editor_selected_campaign_dir = Path(selected_dir)
    if clear_dirty:
        _clear_dirty_state()
    _sync_options_from_config()


def _clear_dirty_state():
    global _dirty_paths, _loaded_fingerprint, _loaded_snapshot, _last_error
    _dirty_paths = set()
    _loaded_fingerprint = _config_fingerprint(_campaign_config)
    _loaded_snapshot = _campaign_config.to_dict()
    _last_error = ""


def _mark_new_default_campaign(selected_dir: Path):
    global _campaign_config, _loaded_from_path, _editor_selected_campaign_dir
    global _dirty_paths, _loaded_fingerprint, _loaded_snapshot
    _campaign_config = CampaignConfig()
    _loaded_from_path = None
    _editor_selected_campaign_dir = Path(selected_dir)
    _dirty_paths = {"campaign.yaml"}
    _loaded_fingerprint = None
    _loaded_snapshot = None
    _sync_options_from_config()


def has_unsaved_config_changes() -> bool:
    if _loaded_fingerprint is None:
        return bool(_dirty_paths)
    return _config_fingerprint(_campaign_config) != _loaded_fingerprint


def dirty_paths() -> list[str]:
    if not has_unsaved_config_changes():
        return []
    if _loaded_snapshot is not None:
        paths = _diff_config_paths(_loaded_snapshot, _campaign_config.to_dict())
        return sorted(paths)
    return sorted(_dirty_paths)


def _diff_config_paths(before, after, prefix: str = "") -> set[str]:
    if before == after:
        return set()
    if isinstance(before, dict) and isinstance(after, dict):
        paths: set[str] = set()
        for key in sorted(set(before) | set(after)):
            child_prefix = key if not prefix else prefix + "." + key
            paths.update(
                _diff_config_paths(
                    before.get(key),
                    after.get(key),
                    child_prefix,
                )
            )
        return paths
    return {prefix or "campaign.yaml"}


def _state_path_for_campaign(campaign_dir: Path) -> Path:
    from ichor.hpc.active_learning.daemon.state import DEFAULT_STATE_FILENAME

    return campaign_dir / ".DATA" / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME


def _read_editor_state_for_lock(campaign_dir: Path):
    from ichor.hpc.active_learning.daemon.state import read_state

    return read_state(_state_path_for_campaign(campaign_dir))


def current_config_lock_review():
    global _last_config_lock_error
    campaign_dir = _editor_selected_campaign_dir or selected_campaign_dir_or_none()
    if campaign_dir is None:
        return None
    state_path = _state_path_for_campaign(campaign_dir)
    if not state_path.is_file():
        _last_config_lock_error = ""
        return None
    try:
        state = _read_editor_state_for_lock(campaign_dir)
        from ichor.hpc.active_learning.daemon.config_lock import (
            review_config_changes,
        )

        review = review_config_changes(campaign_dir, _campaign_config, state)
    except Exception as exc:
        _last_config_lock_error = (
            "config lock review failed: "
            + type(exc).__name__
            + ": "
            + str(exc)[:160]
        )
        return None
    _last_config_lock_error = ""
    return review


def _saved_config_path_for_review(config_override=None) -> Optional[Path]:
    campaign_dir = selected_campaign_dir_or_none()
    if campaign_dir is None:
        return None
    if config_override:
        return Path(config_override).expanduser().resolve()
    return campaign_dir / "campaign.yaml"


def _load_saved_config_for_review(config_path: Path) -> CampaignConfig:
    if not config_path.is_file():
        raise FileNotFoundError("campaign config is not a readable file: " + str(config_path))
    return CampaignConfig.from_yaml(config_path)


def saved_config_lock_review(config_override=None):
    global _last_saved_config_lock_error
    _last_saved_config_lock_error = ""
    override_requested = bool(config_override)
    campaign_dir = selected_campaign_dir_or_none()
    if campaign_dir is None:
        if override_requested:
            _last_saved_config_lock_error = "No campaign directory is selected."
        return None
    config_path = _saved_config_path_for_review(config_override)
    if config_path is None:
        return None
    state_path = _state_path_for_campaign(campaign_dir)
    if override_requested and not config_path.is_file():
        _last_saved_config_lock_error = (
            "Selected config override is not a readable file: "
            + str(config_path)
        )
        return None
    if not config_path.is_file():
        return None
    try:
        cfg = _load_saved_config_for_review(config_path)
    except Exception as exc:
        if override_requested:
            _last_saved_config_lock_error = (
                "Saved config could not be loaded for lock review: "
                + type(exc).__name__
                + ": "
                + str(exc)[:200]
            )
        return None
    if not state_path.is_file():
        return None
    try:
        state = _read_editor_state_for_lock(campaign_dir)
        from ichor.hpc.active_learning.daemon.config_lock import (
            review_config_changes,
        )

        return review_config_changes(campaign_dir, cfg, state)
    except Exception as exc:
        if override_requested:
            _last_saved_config_lock_error = (
                "Selected config override could not be reviewed: "
                + type(exc).__name__
                + ": "
                + str(exc)[:200]
            )
        return None


def blocked_config_lock_paths() -> set[str]:
    review = current_config_lock_review()
    if review is None:
        return set()
    return {change.path for change in review.blocked_changes}


def allowed_config_lock_paths() -> set[str]:
    review = current_config_lock_review()
    if review is None:
        return set()
    return {change.path for change in review.allowed_changes}


def _summarise_paths(paths) -> str:
    values = sorted(paths)
    if not values:
        return ""
    return ", ".join(values[:8]) + (" ..." if len(values) > 8 else "")


def _config_lock_change_status(path: str) -> str:
    review = current_config_lock_review()
    if review is not None:
        for change in review.blocked_changes:
            if change.path == path:
                return "blocked: " + change.reason
        for change in review.allowed_changes:
            if change.path == path:
                return "allowed: " + change.reason
    from ichor.hpc.active_learning.daemon import config_lock as lock_mod

    status = lock_mod.describe_field_editability(path)
    if status == "unclassified lock policy":
        return status
    if path == "schema_version":
        return "read-only: " + status
    return "window: " + status


def _refresh_config_lock_options() -> None:
    campaign_dir = _editor_selected_campaign_dir or selected_campaign_dir_or_none()
    started = bool(
        campaign_dir is not None
        and _state_path_for_campaign(campaign_dir).is_file()
    )
    edit_campaign_config_menu_options.campaign_started = (
        "yes" if started else "no"
    )
    review = current_config_lock_review()
    if not started:
        edit_campaign_config_menu_options.config_lock = "not started"
        edit_campaign_config_menu_options.blocked_config_fields = ""
        edit_campaign_config_menu_options.allowed_config_fields = ""
        return
    if _last_config_lock_error:
        edit_campaign_config_menu_options.config_lock = _last_config_lock_error
        edit_campaign_config_menu_options.blocked_config_fields = ""
        edit_campaign_config_menu_options.allowed_config_fields = ""
        return
    if review is None:
        edit_campaign_config_menu_options.config_lock = "unavailable"
        edit_campaign_config_menu_options.blocked_config_fields = ""
        edit_campaign_config_menu_options.allowed_config_fields = ""
        return
    if review.blocked_changes:
        edit_campaign_config_menu_options.config_lock = "blocked changes"
    elif review.allowed_changes:
        edit_campaign_config_menu_options.config_lock = "allowed changes"
    elif not review.lock_existed:
        edit_campaign_config_menu_options.config_lock = "missing"
    else:
        edit_campaign_config_menu_options.config_lock = "clean"
    edit_campaign_config_menu_options.blocked_config_fields = _summarise_paths(
        change.path for change in review.blocked_changes
    )
    edit_campaign_config_menu_options.allowed_config_fields = _summarise_paths(
        change.path for change in review.allowed_changes
    )


def format_current_config_lock_review() -> str:
    review = current_config_lock_review()
    if review is None:
        return _last_config_lock_error or "No config lock review is available."
    from ichor.hpc.active_learning.daemon.config_lock import format_config_review

    formatted = format_config_review(review)
    return formatted or "No campaign.yaml changes against the config lock."


def _iter_campaign_field_specs():
    seen = set()
    for label in sorted(_BLOCK_MENUS_BY_LABEL):
        block_menu = _BLOCK_MENUS_BY_LABEL[label]
        for spec in block_menu.this_menu_options.fields:
            if spec.path in seen:
                continue
            seen.add(spec.path)
            yield spec.path, spec
    try:
        from ichor.cli.main_menu_submenus.active_learning_campaign_menu.active_learning_campaign_submenus.edit_campaign_config_submenus.edit_ariadne_block_submenu import (
            ARIADNE_FIELD_SPECS,
        )
    except Exception:
        return
    for spec in ARIADNE_FIELD_SPECS:
        path = "ariadne." + spec.path
        if path in seen:
            continue
        seen.add(path)
        yield path, spec


def _format_editability_line(path: str) -> str:
    value = _get_config_value(path)
    return (
        path
        + ": "
        + format_field_value(value)
        + " ["
        + _config_lock_change_status(path)
        + "]"
    )


def saved_config_has_blocked_changes(config_override=None) -> bool:
    review = saved_config_lock_review(config_override)
    return bool(review is not None and review.blocked_changes)


def saved_config_has_lock_changes(config_override=None) -> bool:
    review = saved_config_lock_review(config_override)
    return bool(review is not None and review.changed)


def saved_config_review_failed(config_override=None) -> bool:
    if not config_override:
        return False
    saved_config_lock_review(config_override)
    return bool(_last_saved_config_lock_error)


def saved_config_lock_review_error() -> str:
    return _last_saved_config_lock_error


def format_saved_config_lock_review(config_override=None) -> str:
    review = saved_config_lock_review(config_override)
    if review is None:
        return _last_saved_config_lock_error or "No saved config lock review is available."
    from ichor.hpc.active_learning.daemon.config_lock import format_config_review

    formatted = format_config_review(review)
    return formatted or "No campaign.yaml changes against the config lock."


@dataclass
class EditCampaignConfigMenuOptions(MenuOptions):
    loaded_from: str = "defaults"
    selected_campaign: str = "(none)"
    unsaved_changes: str = "no"
    dirty_fields: str = ""
    last_save_path: str = ""
    last_error: str = ""
    campaign_started: str = "no"
    config_lock: str = "absent"
    blocked_config_fields: str = ""
    allowed_config_fields: str = ""
    system_name: str = "SYSTEM"
    max_iterations: int = 50
    n_seeds_per_iteration: int = 50
    phase_b_descriptor: str = "hybrid_alf_rmsd"
    ferebus_kernel: str = "rbfc_per"
    acquisition_gradient_mode: str = "cartesian_fd"
    acquisition_max_subspace_dim: int = 6

    def __call__(self):
        _ensure_selected_campaign_loaded_for_editor()
        _refresh_config_lock_options()
        return super().__call__()


edit_campaign_config_menu_options = EditCampaignConfigMenuOptions()


def _get_config_value(path: str):
    return get_attr_path(_campaign_config, path)


def _set_config_value(path: str, value):
    old = get_attr_path(_campaign_config, path)
    set_attr_path(_campaign_config, path, value)
    if old != value:
        _dirty_paths.add(path)
    _sync_options_from_config()


def _edit_field(spec: _FieldSpec):
    status = _config_lock_change_status(spec.path)
    if (
        status.startswith("locked")
        or status.startswith("blocked")
        or status.startswith("unclassified")
    ):
        print("Config-lock warning for " + spec.path + ": " + status)
        answer = user_input_free_flow(
            "Type YES to edit this field anyway: ",
            "",
        )
        if str(answer).strip() != "YES":
            print("Edit cancelled.")
            return
    old = _get_config_value(spec.path)
    _shared_edit_field(spec, _get_config_value, _set_config_value)
    new = _get_config_value(spec.path)
    if old != new:
        print("Set " + spec.path + ":")
        print("  old: " + str(old))
        print("  new: " + str(new))
        print("Pending changes are not saved yet. Use Save to disk.")


def _make_block_menu(title: str, subtitle: str, fields):
    return make_field_menu(
        title,
        subtitle,
        fields,
        _get_config_value,
        _set_config_value,
        prologue_text="Current values for this campaign.yaml block:\n",
        status_for_path=_config_lock_change_status,
        include_parent_menu_options=False,
    )


def _spec(path: str, input_kind: str, choices=None, transform=None, prompt=None):
    return _shared_spec(path, input_kind, choices, transform, prompt)


def _read_only_spec(path: str):
    return _FieldSpec(path=path, read_only=True)


def _auto_or_int(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() == "auto":
        return "auto"
    return int(text)


def _sync_options_from_config():
    edit_campaign_config_menu_options.loaded_from = (
        str(_loaded_from_path)
        if _loaded_from_path is not None
        else "DEFAULTS ONLY -- no campaign.yaml loaded"
    )
    edit_campaign_config_menu_options.selected_campaign = (
        str(_editor_selected_campaign_dir)
        if _editor_selected_campaign_dir is not None
        else "(none)"
    )
    edit_campaign_config_menu_options.unsaved_changes = (
        "yes" if has_unsaved_config_changes() else "no"
    )
    dirty = dirty_paths()
    edit_campaign_config_menu_options.dirty_fields = (
        ", ".join(dirty[:8]) + (" ..." if len(dirty) > 8 else "")
        if dirty
        else ""
    )
    edit_campaign_config_menu_options.last_save_path = (
        str(_last_save_path) if _last_save_path is not None else ""
    )
    edit_campaign_config_menu_options.last_error = (
        _last_error[:160] if _last_error else ""
    )
    edit_campaign_config_menu_options.system_name = _campaign_config.campaign.system_name
    edit_campaign_config_menu_options.max_iterations = _campaign_config.campaign.max_iterations
    edit_campaign_config_menu_options.n_seeds_per_iteration = (
        _campaign_config.seed_selection.n_seeds_per_iteration
    )
    edit_campaign_config_menu_options.phase_b_descriptor = _campaign_config.phase_b.descriptor
    edit_campaign_config_menu_options.ferebus_kernel = _campaign_config.ferebus.kernel
    edit_campaign_config_menu_options.acquisition_gradient_mode = (
        _campaign_config.acquisition.gradient.mode
    )
    edit_campaign_config_menu_options.acquisition_max_subspace_dim = (
        _campaign_config.acquisition.subspace.max_subspace_dim
    )


def _campaign_yaml_path():
    return selected_campaign_dir() / "campaign.yaml"


def _confirm_discard_dirty(action: str) -> bool:
    if not has_unsaved_config_changes():
        return True
    print("There are unsaved campaign.yaml edits:")
    for path in dirty_paths()[:20]:
        print("  " + path)
    answer = user_input_free_flow(
        "Type YES to discard these edits and " + action + ": ",
        "",
    )
    return str(answer).strip() == "YES"


def _pending_sparse_yaml() -> str:
    paths = dirty_paths()
    if not paths:
        return "No pending campaign.yaml field changes.\n"
    lines = ["Pending campaign.yaml field changes:\n"]
    for path in paths:
        lines.append(_format_config_change_line(path) + "\n")
    return "".join(lines)


def _get_snapshot_value(path: str):
    if _loaded_snapshot is None:
        raise KeyError(path)
    cursor = _loaded_snapshot
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(path)
        cursor = cursor[part]
    return cursor


def _format_config_change_line(path: str) -> str:
    if path == "campaign.yaml":
        return "  campaign.yaml: <not present> -> generated template"
    try:
        old = format_field_value(_get_snapshot_value(path))
    except KeyError:
        old = "<not present>"
    try:
        new = format_field_value(_get_config_value(path))
    except Exception:
        new = "<not available>"
    return "  " + path + ": " + old + " -> " + new


def load_config_for_campaign_dir(
    campaign_dir: Path,
    *,
    quiet: bool = False,
    prompt_if_dirty: bool = False,
) -> bool:
    global _last_error
    selected = Path(campaign_dir).expanduser().absolute()
    if not selected.exists():
        _last_error = "Campaign directory does not exist: " + str(selected)
        if not quiet:
            print(_last_error)
        _sync_options_from_config()
        return False
    if not selected.is_dir():
        _last_error = "Campaign path is not a directory: " + str(selected)
        if not quiet:
            print(_last_error)
        _sync_options_from_config()
        return False
    if prompt_if_dirty and not _confirm_discard_dirty("load " + str(selected)):
        if not quiet:
            print("Load cancelled; current editor state was kept.")
        _sync_options_from_config()
        return False
    yaml_path = selected / "campaign.yaml"
    if yaml_path.is_file():
        try:
            cfg = CampaignConfig.from_yaml(yaml_path)
        except Exception as exc:
            _last_error = "Failed to load " + str(yaml_path) + ": " + str(exc)
            if not quiet:
                print(_last_error)
            _sync_options_from_config()
            return False
        _replace_campaign_config(cfg, loaded_from=yaml_path, selected_dir=selected)
        if not quiet:
            print("Loaded " + str(yaml_path))
        return True
    _mark_new_default_campaign(selected)
    if not quiet:
        print("No campaign.yaml at " + str(yaml_path))
        print("Loaded defaults for a new unsaved campaign.yaml.")
    return True


def _ensure_selected_campaign_loaded_for_editor() -> None:
    selected = selected_campaign_dir_or_none()
    if selected is None:
        return
    if _editor_selected_campaign_dir == selected:
        return
    if has_unsaved_config_changes():
        return
    load_config_for_campaign_dir(selected, quiet=True, prompt_if_dirty=False)


def _pause():
    user_input_free_flow("Press enter to return to the menu: ", "")


class EditCampaignConfigFunctions:
    @staticmethod
    def show_current_config():
        print("Dense/internal diagnostic in-memory config snapshot")
        print(
            "Diagnostic view only: this includes internal and hidden fields. "
            "Normal sampling control is campaign.sampling_aggressiveness; "
            "manual edits to derived lower-level fields may be rejected by the "
            "config lock after ARIADNE outputs exist."
        )
        print("Loaded from: " + edit_campaign_config_menu_options.loaded_from)
        print("Selected campaign: " + edit_campaign_config_menu_options.selected_campaign)
        print("Unsaved changes: " + edit_campaign_config_menu_options.unsaved_changes)
        print(json.dumps(_campaign_config.to_dict(), indent=2, sort_keys=True))
        _pause()

    @staticmethod
    def show_sampling_protocol_summary():
        print("Summary source: current in-memory editor config.")
        print(format_sampling_protocol_summary(_campaign_config))
        _pause()

    @staticmethod
    def show_unsaved_changes():
        if not has_unsaved_config_changes():
            print("No unsaved changes.")
        else:
            print("Unsaved fields:")
            for path in dirty_paths():
                print("  " + path)
        _pause()

    @staticmethod
    def show_config_lock_review():
        print(format_current_config_lock_review())
        _pause()

    @staticmethod
    def show_config_editability_windows():
        print("Config editability windows")
        print("Source: current in-memory editor config.")
        if current_config_lock_review() is not None:
            print("Current lock-review results are included where fields changed.")
        for path, _spec_obj in _iter_campaign_field_specs():
            print("  " + _format_editability_line(path))
        _pause()

    @staticmethod
    def show_pending_yaml_diff():
        print("Pending campaign.yaml field changes:")
        print(_pending_sparse_yaml())
        _pause()

    @staticmethod
    def export_dense_config_snapshot():
        try:
            target = _campaign_yaml_path().with_name("campaign.dense.yaml")
        except CampaignSelectionError as exc:
            print_campaign_selection_error(exc)
            _pause()
            return
        try:
            _campaign_config.to_yaml_dense(target)
        except Exception as exc:
            print("Failed to write dense config snapshot: " + str(exc))
            _pause()
            return
        print(
            "Wrote dense/internal diagnostic config snapshot to "
            + str(target)
            + ". Diagnostic view only: this file includes hidden fields; normal "
            + "sampling control is campaign.sampling_aggressiveness."
        )
        _pause()

    @staticmethod
    def discard_unsaved_changes():
        try:
            yaml_path = _campaign_yaml_path()
        except CampaignSelectionError as exc:
            print_campaign_selection_error(exc)
            _pause()
            return
        if not yaml_path.exists():
            print("No campaign.yaml at " + str(yaml_path) + " -- nothing to reload.")
            _pause()
            return
        if not _confirm_discard_dirty("reload from disk"):
            print("Reload cancelled.")
            _pause()
            return
        load_config_for_campaign_dir(yaml_path.parent, quiet=True, prompt_if_dirty=False)
        print("Reloaded " + str(yaml_path))
        _pause()

    @staticmethod
    def load_from_disk():
        try:
            yaml_path = _campaign_yaml_path()
        except CampaignSelectionError as exc:
            print_campaign_selection_error(exc)
            _pause()
            return
        if not yaml_path.exists():
            print("No campaign.yaml at " + str(yaml_path) + " -- nothing to load.")
            _pause()
            return
        if not _confirm_discard_dirty("load from disk"):
            print("Load cancelled.")
            _pause()
            return
        if not load_config_for_campaign_dir(
            yaml_path.parent, quiet=True, prompt_if_dirty=False
        ):
            _pause()
            return
        _log_info("Campaign config loaded from " + str(yaml_path))
        print("Loaded " + str(yaml_path))
        _pause()

    @staticmethod
    def reset_to_defaults():
        if not _confirm_discard_dirty("reset to defaults"):
            print("Reset cancelled.")
            _pause()
            return
        _replace_campaign_config(
            CampaignConfig(),
            loaded_from=None,
            selected_dir=_editor_selected_campaign_dir,
            clear_dirty=False,
        )
        _dirty_paths.add("campaign.yaml")
        _log_info("Campaign config reset to defaults")
        _sync_options_from_config()
        print("Reset to defaults. Save to disk to write campaign.yaml.")
        _pause()

    @staticmethod
    def validate_current_config():
        try:
            _campaign_config._validate()
        except ConfigValidationError as exc:
            print("Validation failed.")
            print("  " + str(exc))
            _pause()
            return
        print("Current config validates.")
        _pause()

    @staticmethod
    def edit_campaign_identity():
        _campaign_config.system_name = user_input_free_flow(
            "system_name: ", _campaign_config.system_name,
        )
        _sync_options_from_config()

    @staticmethod
    def edit_iteration_control():
        _campaign_config.max_iterations = user_input_int(
            "max_iterations: ", _campaign_config.max_iterations,
        )
        _campaign_config.poll_interval_seconds = user_input_int(
            "poll_interval_seconds: ", _campaign_config.poll_interval_seconds,
        )
        _campaign_config.poll_interval_idle_seconds = user_input_int(
            "poll_interval_idle_seconds: ", _campaign_config.poll_interval_idle_seconds,
        )
        _campaign_config.poll_sacct_empty_max_ticks = user_input_int(
            "poll_sacct_empty_max_ticks (0 disables empty-sacct escalation): ",
            _campaign_config.poll_sacct_empty_max_ticks,
        )
        _sync_options_from_config()

    @staticmethod
    def edit_resources():
        _unsupported_sequential_field_editor()

    @staticmethod
    def edit_gaussian():
        _unsupported_sequential_field_editor()

    @staticmethod
    def edit_seed_selection():
        s = _campaign_config.seed_selection
        s.n_seeds_per_iteration = user_input_int(
            "seed_selection.n_seeds_per_iteration: ", s.n_seeds_per_iteration,
        )
        s.bulk_fraction = user_input_float(
            "seed_selection.bulk_fraction (0.0-1.0): ", s.bulk_fraction,
        )
        chosen = user_input_restricted(
            sorted(VALID_SEED_SELECTION_STRATEGIES),
            "seed_selection.strategy: ",
            s.strategy,
        )
        if chosen is not None:
            s.strategy = chosen
        s.variance_chunk_size = user_input_int(
            "seed_selection.variance_chunk_size: ", s.variance_chunk_size,
        )
        s.d_optimal_pool_multiplier = user_input_int(
            "seed_selection.d_optimal_pool_multiplier: ",
            s.d_optimal_pool_multiplier,
        )
        s.d_optimal_jitter = user_input_float(
            "seed_selection.d_optimal_jitter: ", s.d_optimal_jitter,
        )
        s.d_optimal_novelty_floor = user_input_float(
            "seed_selection.d_optimal_novelty_floor: ", s.d_optimal_novelty_floor,
        )
        s.d_optimal_score_power = user_input_float(
            "seed_selection.d_optimal_score_power: ", s.d_optimal_score_power,
        )
        _sync_options_from_config()

    @staticmethod
    def edit_anti_overlap():
        a = _campaign_config.anti_overlap
        a.min_post_ariadne_whitened_distance = user_input_float(
            "anti_overlap.min_post_ariadne_whitened_distance: ",
            a.min_post_ariadne_whitened_distance,
        )
        a.max_post_ariadne_whitened_distance = user_input_float(
            "anti_overlap.max_post_ariadne_whitened_distance: ",
            a.max_post_ariadne_whitened_distance,
        )
        a.enforce_post_ariadne = user_input_bool(
            "anti_overlap.enforce_post_ariadne: ", a.enforce_post_ariadne,
        )
        _sync_options_from_config()

    @staticmethod
    def edit_phase_b():
        pb = _campaign_config.phase_b
        chosen = user_input_restricted(
            sorted(VALID_DESCRIPTORS), "phase_b.descriptor: ", pb.descriptor,
        )
        if chosen is not None:
            pb.descriptor = chosen
        pb.beta = user_input_float("phase_b.beta (0.0-1.0): ", pb.beta)
        _sync_options_from_config()

    @staticmethod
    def edit_split():
        _unsupported_sequential_field_editor()

    @staticmethod
    def edit_ferebus():
        _unsupported_sequential_field_editor()

    @staticmethod
    def edit_acquisition_core():
        a = _campaign_config.acquisition
        a.property_name = user_input_free_flow(
            "acquisition.property_name: ", a.property_name,
        )
        a.use_scaled_posterior_covariance = user_input_bool(
            "acquisition.use_scaled_posterior_covariance: ",
            a.use_scaled_posterior_covariance,
        )
        a.allow_uniform_posterior_fallback = user_input_bool(
            "acquisition.allow_uniform_posterior_fallback: ",
            a.allow_uniform_posterior_fallback,
        )
        _sync_options_from_config()

    @staticmethod
    def edit_robustness():
        _campaign_config.failure_threshold_fraction = user_input_float(
            "failure_threshold_fraction (0.0-1.0): ",
            _campaign_config.failure_threshold_fraction,
        )
        _campaign_config.max_force_per_atom_ha_per_ang = user_input_float(
            "max_force_per_atom_ha_per_ang (deprecated acquisition-gradient alias): ",
            _campaign_config.max_force_per_atom_ha_per_ang,
        )
        _sync_options_from_config()

    @staticmethod
    def edit_acquisition_subspace():
        sb = _campaign_config.acquisition.subspace
        sb.neighbour_count = user_input_int(
            "acquisition.subspace.neighbour_count: ", sb.neighbour_count,
        )
        sb.variance_capture = user_input_float(
            "acquisition.subspace.variance_capture (0.0-1.0): ", sb.variance_capture,
        )
        sb.min_subspace_dim = user_input_int(
            "acquisition.subspace.min_subspace_dim: ", sb.min_subspace_dim,
        )
        sb.max_subspace_dim = user_input_int(
            "acquisition.subspace.max_subspace_dim: ", sb.max_subspace_dim,
        )
        chosen = user_input_restricted(
            sorted(VALID_MODE_WEIGHTING_POLICIES),
            "acquisition.subspace.mode_weighting_policy: ",
            sb.mode_weighting_policy,
        )
        if chosen is not None:
            sb.mode_weighting_policy = chosen
        sb.canonicalise_basis = user_input_bool(
            "acquisition.subspace.canonicalise_basis: ", sb.canonicalise_basis,
        )
        _sync_options_from_config()

    @staticmethod
    def edit_acquisition_weights():
        w = _campaign_config.acquisition.weights
        w.lambda_force = user_input_float(
            "acquisition.weights.lambda_force: ", w.lambda_force,
        )
        w.lambda_frequency = user_input_float(
            "acquisition.weights.lambda_frequency: ", w.lambda_frequency,
        )
        w.lambda_anharmonic = user_input_float(
            "acquisition.weights.lambda_anharmonic: ", w.lambda_anharmonic,
        )
        w.lambda_energy = user_input_float(
            "acquisition.weights.lambda_energy: ", w.lambda_energy,
        )
        w.lambda_distance = user_input_float(
            "acquisition.weights.lambda_distance: ", w.lambda_distance,
        )
        _sync_options_from_config()

    @staticmethod
    def edit_acquisition_gradient():
        g = _campaign_config.acquisition.gradient
        chosen = user_input_restricted(
            sorted(VALID_GRADIENT_MODES), "acquisition.gradient.mode: ", g.mode,
        )
        if chosen is not None:
            g.mode = chosen
        g.cartesian_step = user_input_float(
            "acquisition.gradient.cartesian_step: ", g.cartesian_step,
        )
        g.active_step = user_input_float(
            "acquisition.gradient.active_step: ", g.active_step,
        )
        _sync_options_from_config()

    @staticmethod
    def edit_acquisition_barrier():
        b = _campaign_config.acquisition.barrier
        b.use_connectivity_barrier = user_input_bool(
            "acquisition.barrier.use_connectivity_barrier: ", b.use_connectivity_barrier,
        )
        b.nonbonded_clash_scale = user_input_float(
            "acquisition.barrier.nonbonded_clash_scale: ", b.nonbonded_clash_scale,
        )
        b.clash_delta = user_input_float(
            "acquisition.barrier.clash_delta: ", b.clash_delta,
        )
        b.clash_lambda = user_input_float(
            "acquisition.barrier.clash_lambda: ", b.clash_lambda,
        )
        b.bond_lower_scale = user_input_float(
            "acquisition.barrier.bond_lower_scale: ", b.bond_lower_scale,
        )
        b.bond_upper_scale = user_input_float(
            "acquisition.barrier.bond_upper_scale: ", b.bond_upper_scale,
        )
        b.bond_lambda = user_input_float(
            "acquisition.barrier.bond_lambda: ", b.bond_lambda,
        )
        b.energy_cap_quantile = user_input_float(
            "acquisition.barrier.energy_cap_quantile (0.0-1.0): ", b.energy_cap_quantile,
        )
        b.energy_cap_lambda = user_input_float(
            "acquisition.barrier.energy_cap_lambda: ", b.energy_cap_lambda,
        )
        _sync_options_from_config()

    @staticmethod
    def edit_acquisition_stencils():
        st = _campaign_config.acquisition.stencils
        st.step_scale = user_input_float(
            "acquisition.stencils.step_scale: ", st.step_scale,
        )
        st.min_step = user_input_float(
            "acquisition.stencils.min_step: ", st.min_step,
        )
        st.max_step = user_input_float(
            "acquisition.stencils.max_step: ", st.max_step,
        )
        st.curvature_floor = user_input_float(
            "acquisition.stencils.curvature_floor: ", st.curvature_floor,
        )
        st.softplus_scale = user_input_float(
            "acquisition.stencils.softplus_scale: ", st.softplus_scale,
        )
        st.autotune_from_cubic = user_input_bool(
            "acquisition.stencils.autotune_from_cubic: ",
            st.autotune_from_cubic,
        )
        _sync_options_from_config()

    @staticmethod
    def edit_acquisition_references():
        rs = _campaign_config.acquisition.references
        rs.max_reference_samples = user_input_int(
            "acquisition.references.max_reference_samples: ", rs.max_reference_samples,
        )
        rs.floor = user_input_float(
            "acquisition.references.floor: ", rs.floor,
        )
        chosen = user_input_restricted(
            ["every_iteration", "every_n_iterations", "never"],
            "acquisition.references.refresh_policy: ",
            rs.refresh_policy,
        )
        if chosen is not None:
            rs.refresh_policy = chosen
        rs.refresh_period = user_input_int(
            "acquisition.references.refresh_period (only used by every_n_iterations): ",
            rs.refresh_period,
        )
        _sync_options_from_config()

    @staticmethod
    def edit_stop():
        st = _campaign_config.stop
        st.alpha0_streak_threshold = user_input_float(
            "stop.alpha0_streak_threshold: ", st.alpha0_streak_threshold,
        )
        st.alpha0_streak_length = user_input_int(
            "stop.alpha0_streak_length (0 disables this rule): ",
            st.alpha0_streak_length,
        )
        st.rel_alpha_improvement_min = user_input_float(
            "stop.rel_alpha_improvement_min: ", st.rel_alpha_improvement_min,
        )
        st.rel_alpha_improvement_window = user_input_int(
            "stop.rel_alpha_improvement_window (0 disables this rule): ",
            st.rel_alpha_improvement_window,
        )
        st.min_iterations_before_stop = user_input_int(
            "stop.min_iterations_before_stop: ", st.min_iterations_before_stop,
        )
        _sync_options_from_config()

    @staticmethod
    def save_to_disk():
        global _last_save_path, _last_error
        try:
            _campaign_config._validate()
        except ConfigValidationError as exc:
            print("Validation failed; campaign.yaml NOT written.")
            print("  " + str(exc))
            _last_error = "Validation failed: " + str(exc)
            _sync_options_from_config()
            _pause()
            return
        campaign_dir = _editor_selected_campaign_dir or selected_campaign_dir_or_none()
        started = bool(
            campaign_dir is not None
            and _state_path_for_campaign(campaign_dir).is_file()
        )
        review = current_config_lock_review()
        if started and review is None:
            print("campaign.yaml NOT written because config-lock review is unavailable.")
            print(format_current_config_lock_review())
            _last_error = "Config lock review unavailable"
            _sync_options_from_config()
            _pause()
            return
        if review is not None and review.blocked_changes:
            print(
                "campaign.yaml NOT written because these changes are "
                "locked after campaign start:"
            )
            print(format_current_config_lock_review())
            _last_error = "Config lock blocked save"
            _sync_options_from_config()
            _pause()
            return
        try:
            target = _campaign_yaml_path()
        except CampaignSelectionError as exc:
            print_campaign_selection_error(exc)
            _pause()
            return
        if not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        changed = dirty_paths()
        changed_lines = [_format_config_change_line(path) for path in changed]
        try:
            from ichor.hpc.active_learning.campaign_yaml import (
                patch_campaign_yaml_fields,
            )

            reloaded = patch_campaign_yaml_fields(
                target,
                _campaign_config,
                changed,
            )
        except Exception as exc:
            print(
                "Failed to patch and verify "
                + str(target)
                + "; existing campaign.yaml was left untouched: "
                + str(exc)
            )
            _last_error = "Patch save failed: " + str(exc)
            _sync_options_from_config()
            _pause()
            return
        _replace_campaign_config(
            reloaded,
            loaded_from=target,
            selected_dir=target.parent,
            clear_dirty=True,
        )
        _last_save_path = target
        _last_error = ""
        _sync_options_from_config()
        _log_info("Campaign config saved to " + str(target))
        print("Wrote and verified " + str(target))
        if changed:
            print("Changed fields saved:")
            for line in changed_lines:
                print(line)
        print("Note: unmodified campaign.yaml fields and comments are preserved where possible.")
        _pause()


def _unsupported_sequential_field_editor():
    raise RuntimeError(
        "Sequential campaign config editors are no longer supported. "
        "Use the block field menus so current values remain visible and every "
        "campaign.yaml field is edited through one authoritative path."
    )


for _legacy_editor_name in (
    "edit_campaign_identity",
    "edit_iteration_control",
    "edit_resources",
    "edit_gaussian",
    "edit_seed_selection",
    "edit_anti_overlap",
    "edit_phase_b",
    "edit_split",
    "edit_ferebus",
    "edit_acquisition_core",
    "edit_robustness",
    "edit_acquisition_subspace",
    "edit_acquisition_weights",
    "edit_acquisition_gradient",
    "edit_acquisition_barrier",
    "edit_acquisition_stencils",
    "edit_acquisition_references",
    "edit_stop",
):
    setattr(
        EditCampaignConfigFunctions,
        _legacy_editor_name,
        staticmethod(_unsupported_sequential_field_editor),
    )


edit_campaign_config_menu = ConsoleMenu(
    this_menu_options=edit_campaign_config_menu_options,
    title=EDIT_CAMPAIGN_CONFIG_MENU_DESCRIPTION.title,
    subtitle=EDIT_CAMPAIGN_CONFIG_MENU_DESCRIPTION.subtitle,
    prologue_text=EDIT_CAMPAIGN_CONFIG_MENU_DESCRIPTION.prologue_description_text,
    epilogue_text=EDIT_CAMPAIGN_CONFIG_MENU_DESCRIPTION.epilogue_description_text,
    show_exit_option=EDIT_CAMPAIGN_CONFIG_MENU_DESCRIPTION.show_exit_option,
)


_BLOCK_MENUS_BY_LABEL = {
    "Edit campaign": _make_block_menu(
        "Edit Campaign",
        "Campaign identity, operator inputs, length, and sampling control.",
        [
            _read_only_spec("schema_version"),
            _spec("campaign.system_name", "str"),
            _spec("campaign.max_iterations", "int"),
            _spec("campaign.random_seed", "int"),
            _spec("campaign.custom_bootstrap", "bool"),
            _spec(
                "campaign.sampling_aggressiveness",
                "int",
                prompt=(
                    "campaign.sampling_aggressiveness "
                    "(1 conservative, 5 balanced, 10 exploratory): "
                ),
            ),
        ],
    ),
    "Edit point_allocation": _make_block_menu(
        "Edit Point Allocation",
        "Exact bootstrap and per-iteration training/validation slot counts.",
        [
            _spec("point_allocation.bootstrap_training_size", "int"),
            _spec("point_allocation.bootstrap_internal_validation_size", "int"),
            _spec("point_allocation.bootstrap_external_validation_size", "int"),
            _spec("point_allocation.batch_training_size", "int"),
            _spec("point_allocation.batch_internal_validation_size", "int"),
        ],
    ),
    "Edit resource defaults": _make_block_menu(
        "Edit Resource Defaults",
        "Default SLURM resources inherited by backend-specific resource blocks.",
        [
            _spec("resources.defaults.partition", "str"),
            _spec("resources.defaults.walltime_hours", "float"),
            _spec("resources.defaults.cpus_per_task", "str", transform=_auto_or_int, prompt="resources.defaults.cpus_per_task (auto or positive integer): "),
            _spec("resources.defaults.mem_per_cpu", "str", prompt="resources.defaults.mem_per_cpu (auto or SLURM style, e.g. 4G): "),
            _spec("resources.array_concurrency_limit", "optional_int"),
            _spec("resources.fail_on_memory_estimate_exceeds_request", "bool"),
            _spec("resources.memory_estimate_safety_factor", "float"),
            _spec("resources.scheduler_usage_telemetry", "bool"),
            _spec("resources.scheduler_usage_history_limit", "int"),
            _spec("resources.gradient_parallel_backend", "choice", choices=sorted(VALID_GRADIENT_PARALLEL_BACKENDS)),
        ],
    ),
    "Edit POLUS resources": _make_block_menu(
        "Edit POLUS Resources",
        "POLUS backend overrides; null values inherit from resource defaults.",
        [
            _spec("resources.polus.partition", "optional_str"),
            _spec("resources.polus.walltime_hours", "optional_float"),
            _spec("resources.polus.cpus_per_task", "optional_str", transform=_auto_or_int, prompt="resources.polus.cpus_per_task (null, auto, or positive integer): "),
            _spec("resources.polus.mem_per_cpu", "optional_str", prompt="resources.polus.mem_per_cpu (null, auto, or SLURM style, e.g. 4G): "),
            _spec("resources.polus.auto_max_workers", "int"),
            _spec("resources.polus.target_pairs_per_worker", "int"),
            _spec("resources.polus.in_memory_distance_store_fraction", "float"),
        ],
    ),
    "Edit Gaussian runtime resources": _make_block_menu(
        "Edit Gaussian Runtime Resources",
        "Gaussian Slurm resources and Gaussian memory contract.",
        [
            _spec("resources.gaussian.partition", "optional_str"),
            _spec("resources.gaussian.walltime_hours", "optional_float"),
            _spec("resources.gaussian.cpus_per_task", "optional_str", transform=_auto_or_int, prompt="resources.gaussian.cpus_per_task (null, auto, or positive integer): "),
            _spec("resources.gaussian.mem_per_cpu", "optional_str", prompt="resources.gaussian.mem_per_cpu (null, auto, or SLURM style, e.g. 4G): "),
            _spec("resources.gaussian.memory_mode", "choice", choices=sorted(VALID_GAUSSIAN_MEMORY_MODES)),
            _spec("resources.gaussian.link0_mem", "str", prompt="resources.gaussian.link0_mem (Gaussian style, e.g. 8GB): "),
            _spec("resources.gaussian.memory_fraction_of_slurm", "float"),
        ],
    ),
    "Edit AIMAll resources": _make_block_menu(
        "Edit AIMAll Resources",
        "AIMAll backend overrides; null values inherit from resource defaults.",
        [
            _spec("resources.aimall.partition", "optional_str"),
            _spec("resources.aimall.walltime_hours", "optional_float"),
            _spec("resources.aimall.cpus_per_task", "optional_str", transform=_auto_or_int, prompt="resources.aimall.cpus_per_task (null, auto, or positive integer): "),
            _spec("resources.aimall.mem_per_cpu", "optional_str", prompt="resources.aimall.mem_per_cpu (null, auto, or SLURM style, e.g. 4G): "),
        ],
    ),
    "Edit ARIADNE resources": _make_block_menu(
        "Edit ARIADNE Resources",
        "ARIADNE backend overrides; null values inherit from resource defaults.",
        [
            _spec("resources.ariadne.partition", "optional_str"),
            _spec("resources.ariadne.walltime_hours", "optional_float"),
            _spec("resources.ariadne.cpus_per_task", "optional_str", transform=_auto_or_int, prompt="resources.ariadne.cpus_per_task (null, auto, or positive integer): "),
            _spec("resources.ariadne.mem_per_cpu", "optional_str", prompt="resources.ariadne.mem_per_cpu (null, auto, or SLURM style, e.g. 4G): "),
        ],
    ),
    "Edit FEREBUS resources": _make_block_menu(
        "Edit FEREBUS Resources",
        "FEREBUS backend overrides; null values inherit from resource defaults.",
        [
            _spec("resources.ferebus.partition", "optional_str"),
            _spec("resources.ferebus.walltime_hours", "optional_float"),
            _spec("resources.ferebus.cpus_per_task", "optional_str", transform=_auto_or_int, prompt="resources.ferebus.cpus_per_task (null, auto, or positive integer): "),
            _spec("resources.ferebus.mem_per_cpu", "optional_str", prompt="resources.ferebus.mem_per_cpu (null, auto, or SLURM style, e.g. 4G): "),
        ],
    ),
    "Edit Gaussian block": _make_block_menu(
        "Edit Gaussian Block",
        "Ab-initio backend settings for training-data calculations.",
        [
            _spec("gaussian.method", "str"),
            _spec("gaussian.basis_set", "str"),
            _spec("gaussian.charge", "int"),
            _spec("gaussian.spin_multiplicity", "int"),
            _spec(
                "gaussian.extra_route_keywords",
                "csv_list",
                prompt="gaussian.extra_route_keywords (comma-separated; blank for none): ",
            ),
        ],
    ),
    "Edit AIMAll block": _make_block_menu(
        "Edit AIMAll Block",
        "AIMAll invocation controls for live QM post-processing.",
        [
            _spec("aimall.encomp", "int"),
            _spec("aimall.nogui", "bool"),
            _spec("aimall.naat", "str", transform=_auto_or_int, prompt="aimall.naat (auto or positive integer): "),
            _spec("aimall.boaq", "choice", choices=sorted(VALID_AIMALL_BOAQ_VALUES)),
            _spec("aimall.iasmesh", "choice", choices=sorted(VALID_AIMALL_IASMESH_VALUES)),
        ],
    ),
    "Edit seed_selection": _make_block_menu(
        "Edit seed_selection",
        "Seed count and uncertainty-evaluation controls.",
        [
            _spec("seed_selection.n_seeds_per_iteration", "int"),
            _spec("seed_selection.bulk_fraction", "float", prompt="seed_selection.bulk_fraction (0.0-1.0): "),
            _spec("seed_selection.variance_chunk_size", "int"),
            _spec("seed_selection.strategy", "choice", choices=sorted(VALID_SEED_SELECTION_STRATEGIES)),
            _spec("seed_selection.d_optimal_pool_multiplier", "int"),
            _spec("seed_selection.d_optimal_jitter", "float"),
            _spec("seed_selection.d_optimal_novelty_floor", "float"),
            _spec("seed_selection.d_optimal_score_power", "float"),
            _spec(
                "seed_selection.d_optimal_degenerate_policy",
                "choice",
                choices=sorted(VALID_D_OPTIMAL_DEGENERATE_POLICIES),
            ),
            _spec("seed_selection.exclude_committed_seed_frames", "bool"),
            _spec("seed_selection.recent_seed_cooldown_iterations", "int"),
        ],
    ),
    "Edit anti_overlap": _make_block_menu(
        "Edit anti_overlap",
        "Candidate anti-overlap and ARIADNE movement filters.",
        [
            _spec("anti_overlap.min_post_ariadne_whitened_distance", "float"),
            _spec("anti_overlap.max_post_ariadne_whitened_distance", "float"),
            _spec("anti_overlap.enforce_post_ariadne", "bool"),
        ],
    ),
    "Edit phase_b": _make_block_menu(
        "Edit phase_b",
        "Phase-B descriptor/FPS selection controls.",
        [
            _spec("phase_b.descriptor", "choice", choices=sorted(VALID_DESCRIPTORS)),
            _spec("phase_b.beta", "float", prompt="phase_b.beta (0.0-1.0): "),
        ],
    ),
    "Edit geometry_novelty": _make_block_menu(
        "Edit geometry_novelty",
        "Dimensionless Phase-B geometry novelty scaling controls.",
        [
            _spec("geometry_novelty.enabled", "bool"),
            _spec(
                "geometry_novelty.scale_source",
                "choice",
                choices=sorted(VALID_GEOMETRY_NOVELTY_SCALE_SOURCES),
            ),
            _spec(
                "geometry_novelty.statistic",
                "choice",
                choices=sorted(VALID_GEOMETRY_NOVELTY_STATISTICS),
            ),
            _spec("geometry_novelty.scale_floor_angstrom", "float"),
            _spec("geometry_novelty.history_window_iterations", "int"),
            _spec("geometry_novelty.fallback_scale_angstrom", "float"),
        ],
    ),
    "Edit FEREBUS block": _make_block_menu(
        "Edit FEREBUS Block",
        "FEREBUS model-training controls; dataset slots come from point_allocation.",
        [
            _spec("ferebus.kernel", "str", prompt="ferebus.kernel (e.g. rbfc_per, rbf_per): "),
            _spec("ferebus.loss", "str", prompt="ferebus.loss (e.g. huber, mse, mae): "),
            _spec("ferebus.nagents", "int"),
            _spec("ferebus.maxiter", "int"),
            _spec("ferebus.is_constant_noise", "bool"),
            _read_only_spec("ferebus.prior_mean_type"),
            _spec(
                "ferebus.prior_mean_level_of_theory",
                "choice",
                choices=["auto", *sorted(SUPPORTED_LEVELS)],
            ),
            _spec("ferebus.prior_mean_iqa_deviation_factor", "float"),
            _spec("ferebus.feature_scaling", "bool"),
            _read_only_spec("ferebus.property_scaling"),
            _spec("ferebus.full_ARD", "bool"),
            _spec("ferebus.properties", "csv_list", prompt="ferebus.properties (comma-separated, e.g. iqa,q00): "),
        ],
    ),
    "Edit acquisition core": _make_block_menu(
        "Edit Acquisition Core",
        "Top-level adversarial acquisition settings.",
        [
            _spec("acquisition.property_name", "str"),
            _spec("acquisition.use_scaled_posterior_covariance", "bool"),
            _spec("acquisition.allow_uniform_posterior_fallback", "bool"),
        ],
    ),
    "Edit acquisition.subspace": _make_block_menu(
        "Edit acquisition.subspace",
        "Local subspace construction for adversarial acquisition.",
        [
            _spec("acquisition.subspace.neighbour_count", "int"),
            _spec("acquisition.subspace.neighbour_deduplicate_rmsd", "float"),
            _spec("acquisition.subspace.variance_capture", "float", prompt="acquisition.subspace.variance_capture (0.0-1.0): "),
            _spec("acquisition.subspace.min_subspace_dim", "int"),
            _spec("acquisition.subspace.max_subspace_dim", "int"),
            _spec("acquisition.subspace.gaussian_weight_sigma", "optional_float"),
            _spec("acquisition.subspace.covariance_regularization", "float"),
            _spec("acquisition.subspace.canonicalise_basis", "bool"),
            _spec("acquisition.subspace.degeneracy_tolerance", "float"),
            _spec("acquisition.subspace.mode_weighting_policy", "choice", choices=sorted(VALID_MODE_WEIGHTING_POLICIES)),
        ],
    ),
    "Edit acquisition.weights": _make_block_menu(
        "Edit acquisition.weights",
        "Weights in the adversarial acquisition objective.",
        [
            _spec("acquisition.weights.lambda_force", "float"),
            _spec("acquisition.weights.lambda_frequency", "float"),
            _spec("acquisition.weights.lambda_anharmonic", "float"),
            _spec("acquisition.weights.lambda_energy", "float"),
            _spec("acquisition.weights.lambda_distance", "float"),
        ],
    ),
    "Edit acquisition.spectral": _make_block_menu(
        "Edit acquisition.spectral",
        "Observable-oriented spectral frequency acquisition settings.",
        [
            _spec("acquisition.spectral.enabled", "bool"),
            _spec("acquisition.spectral.mode", "choice", choices=sorted(VALID_SPECTRAL_MODES)),
            _spec("acquisition.spectral.mode_weighting", "choice", choices=sorted(VALID_MODE_WEIGHTING_POLICIES)),
            _spec("acquisition.spectral.lambda_spectral", "float"),
            _spec("acquisition.spectral.omega_floor", "float"),
            _spec("acquisition.spectral.low_frequency_power", "float"),
            _spec("acquisition.spectral.max_modes", "optional_int"),
        ],
    ),
    "Edit acquisition.calibrated_energy": _make_block_menu(
        "Edit acquisition.calibrated_energy",
        "Banded calibrated IQA-error utility settings.",
        [
            _spec("acquisition.calibrated_energy.utility", "choice", choices=sorted(VALID_CALIBRATED_ENERGY_UTILITIES)),
            _spec("acquisition.calibrated_energy.band_low_ha", "optional_float"),
            _spec("acquisition.calibrated_energy.band_high_ha", "optional_float"),
            _spec("acquisition.calibrated_energy.low_softness_ha", "optional_float"),
            _spec("acquisition.calibrated_energy.high_softness_ha", "optional_float"),
            _spec("acquisition.calibrated_energy.fallback_to_raw_variance", "bool"),
        ],
    ),
    "Edit acquisition.fullspace_confinement": _make_block_menu(
        "Edit acquisition.fullspace_confinement",
        "Full-space geometric confinement outside the local active subspace.",
        [
            _spec("acquisition.fullspace_confinement.enabled", "bool"),
            _spec("acquisition.fullspace_confinement.lambda_residual", "float"),
            _spec("acquisition.fullspace_confinement.lambda_rmsd", "float"),
        ],
    ),
    "Edit acquisition.size_normalisation": _make_block_menu(
        "Edit acquisition.size_normalisation",
        "System-size intensive acquisition term normalisation.",
        [
            _spec("acquisition.size_normalisation.enabled", "bool"),
            _spec("acquisition.size_normalisation.energy_mode", "choice", choices=sorted(VALID_SIZE_NORMALISATION_ENERGY_MODES)),
            _spec("acquisition.size_normalisation.whitened_distance_mode", "choice", choices=sorted(VALID_SIZE_NORMALISATION_DISTANCE_MODES)),
            _spec("acquisition.size_normalisation.chemistry_barrier_mode", "choice", choices=sorted(VALID_SIZE_NORMALISATION_BARRIER_MODES)),
        ],
    ),
    "Edit acquisition.driver": _make_block_menu(
        "Edit acquisition.driver",
        "Optional cheap optimiser-driving objective; full scoring still selects landings.",
        [
            _spec("acquisition.driver.enabled", "bool"),
            _spec("acquisition.driver.objective", "choice", choices=sorted(VALID_ACQUISITION_DRIVER_OBJECTIVES)),
            _spec("acquisition.driver.gradient_backend", "choice", choices=sorted(VALID_ACQUISITION_DRIVER_GRADIENT_BACKENDS)),
            _spec("acquisition.driver.include_stencils", "bool"),
            _spec("acquisition.driver.analytic_movement", "bool"),
            _spec("acquisition.driver.analytic_whitened_distance", "bool"),
            _spec("acquisition.driver.analytic_pair_barriers", "bool"),
            _spec("acquisition.driver.analytic_fullspace_rmsd", "bool"),
            _spec("acquisition.driver.finite_difference_energy", "bool"),
            _spec("acquisition.driver.analytic_validation", "bool"),
            _spec("acquisition.driver.analytic_validation_tol_cosine", "float"),
            _spec("acquisition.driver.lambda_energy", "float"),
            _spec("acquisition.driver.lambda_movement", "float"),
            _spec("acquisition.driver.lambda_distance", "float"),
            _spec("acquisition.driver.lambda_fullspace", "float"),
            _spec("acquisition.driver.lambda_chemistry", "float"),
        ],
    ),
    "Edit acquisition.gradient": _make_block_menu(
        "Edit acquisition.gradient",
        "Finite-difference gradient controls.",
        [
            _spec("acquisition.gradient.mode", "choice", choices=sorted(VALID_GRADIENT_MODES)),
            _spec("acquisition.gradient.cartesian_step", "float"),
            _spec("acquisition.gradient.active_step", "float"),
            _spec("acquisition.gradient.regularization", "float"),
            _spec("acquisition.gradient.cartesian_step_floor", "float"),
            _spec("acquisition.gradient.ghost_mass_threshold", "float"),
            _spec("acquisition.gradient.max_acquisition_grad_per_ang", "float"),
        ],
    ),
    "Edit acquisition.barrier": _make_block_menu(
        "Edit acquisition.barrier",
        "Geometry and energy risk penalties in the acquisition objective.",
        [
            _spec("acquisition.barrier.use_connectivity_barrier", "bool"),
            _spec("acquisition.barrier.nonbonded_clash_scale", "float"),
            _spec("acquisition.barrier.clash_delta", "float"),
            _spec("acquisition.barrier.clash_lambda", "float"),
            _spec("acquisition.barrier.nonbonded_expansion_scale", "float"),
            _spec("acquisition.barrier.nonbonded_expansion_delta", "float"),
            _spec("acquisition.barrier.nonbonded_expansion_lambda", "float"),
            _spec("acquisition.barrier.bond_lower_scale", "float"),
            _spec("acquisition.barrier.bond_upper_scale", "float"),
            _spec("acquisition.barrier.bond_delta", "float"),
            _spec("acquisition.barrier.bond_lambda", "float"),
            _spec("acquisition.barrier.angle_lower_scale", "float"),
            _spec("acquisition.barrier.angle_upper_scale", "float"),
            _spec("acquisition.barrier.angle_delta", "float"),
            _spec("acquisition.barrier.angle_lambda", "float"),
            _spec("acquisition.barrier.energy_cap_quantile", "float", prompt="acquisition.barrier.energy_cap_quantile (0.0-1.0): "),
            _spec("acquisition.barrier.energy_cap_floor", "float"),
            _spec("acquisition.barrier.energy_cap_delta", "float"),
            _spec("acquisition.barrier.energy_cap_lambda", "float"),
            _spec("acquisition.barrier.softplus_cap", "optional_float"),
        ],
    ),
    "Edit acquisition.stencils": _make_block_menu(
        "Edit acquisition.stencils",
        "Stencil step and curvature controls for finite-difference diagnostics.",
        [
            _spec("acquisition.stencils.step_scale", "float"),
            _spec("acquisition.stencils.min_step", "float"),
            _spec("acquisition.stencils.max_step", "float"),
            _spec("acquisition.stencils.jitter", "float"),
            _spec("acquisition.stencils.curvature_floor", "float"),
            _spec("acquisition.stencils.softplus_scale", "float"),
            _spec("acquisition.stencils.autotune_from_cubic", "bool"),
            _spec("acquisition.stencils.negative_curvature_policy", "choice", choices=sorted(VALID_NEGATIVE_CURVATURE_POLICIES)),
            _spec("acquisition.stencils.lambda_negative_curvature", "float"),
            _spec("acquisition.stencils.weak_mode_gating_enabled", "bool"),
            _spec("acquisition.stencils.weak_mode_omega_low_fraction", "float"),
            _spec("acquisition.stencils.weak_mode_omega_high_fraction", "float"),
            _spec("acquisition.stencils.weak_mode_abs_omega_floor", "float"),
            _spec("acquisition.stencils.weak_mode_penalty", "float"),
            _spec("acquisition.stencils.max_anharmonic_mode_score", "float"),
            _spec("acquisition.stencils.max_anharmonic_total_score", "float"),
        ],
    ),
    "Edit acquisition.references": _make_block_menu(
        "Edit acquisition.references",
        "Reference scale refresh controls for acquisition normalisation.",
        [
            _spec("acquisition.references.max_reference_samples", "int"),
            _spec("acquisition.references.floor", "float"),
            _spec("acquisition.references.refresh_policy", "choice", choices=["every_iteration", "every_n_iterations", "never"]),
            _spec("acquisition.references.refresh_period", "int", prompt="acquisition.references.refresh_period (only used by every_n_iterations): "),
        ],
    ),
    "Edit stop": _make_block_menu(
        "Edit stop",
        "Automatic stopping criteria.",
        [
            _spec("stop.alpha0_streak_threshold", "float"),
            _spec("stop.alpha0_streak_length", "int", prompt="stop.alpha0_streak_length (0 disables this rule): "),
            _spec("stop.rel_alpha_improvement_min", "float"),
            _spec("stop.rel_alpha_improvement_window", "int", prompt="stop.rel_alpha_improvement_window (0 disables this rule): "),
            _spec("stop.min_iterations_before_stop", "int"),
        ],
    ),
    "Edit adversarial_safety": _make_block_menu(
        "Edit adversarial_safety",
        "Safe adversarial landing policy and thresholds.",
        [
            _spec("adversarial_safety.enabled", "bool"),
            _spec("adversarial_safety.reject_unsafe_landings", "bool"),
            _spec("adversarial_safety.salvage_safe_iterate", "bool"),
            _spec("adversarial_safety.backtrack_to_safe_landing", "bool"),
            _spec("adversarial_safety.backtrack_points", "int"),
            _spec("adversarial_safety.allow_seed_fallback", "bool"),
            _spec("adversarial_safety.accept_legacy_missing_landing_safety", "bool"),
            _spec("adversarial_safety.min_whitened_distance", "float"),
            _spec("adversarial_safety.max_whitened_distance", "float"),
            _spec("adversarial_safety.enforce_min_whitened_distance", "bool"),
            _spec("adversarial_safety.max_predicted_energy_delta_ha", "optional_float"),
            _spec("adversarial_safety.max_energy_variance", "optional_float"),
            _spec("adversarial_safety.max_chemistry_penalty", "optional_float"),
            _spec("adversarial_safety.phase_b_filter_enabled", "bool"),
            _spec("adversarial_safety.enforce_movement_band", "bool"),
            _spec("adversarial_safety.under_move_retry", "bool"),
            _spec("adversarial_safety.reject_under_moved_after_retry", "bool"),
            _spec("adversarial_safety.reject_over_moved", "bool"),
        ],
    ),
    "Edit error_calibration": _make_block_menu(
        "Edit error_calibration",
        "Empirical mapping from raw uncertainty to realised IQA error.",
        [
            _spec("error_calibration.enabled", "bool"),
            _spec("error_calibration.mode", "choice", choices=sorted(VALID_ERROR_CALIBRATION_MODES)),
            _spec("error_calibration.min_records_to_apply", "int"),
            _spec("error_calibration.min_model_versions_to_apply", "int"),
            _spec("error_calibration.n_bins", "int"),
            _spec("error_calibration.min_bin_records", "int"),
            _spec("error_calibration.max_records", "int"),
            _spec("error_calibration.max_model_age_iterations", "int"),
            _spec("error_calibration.monotone_estimator", "bool"),
            _spec("error_calibration.quantile", "float"),
            _spec("error_calibration.apply_strength", "float"),
            _spec("error_calibration.group_by_atom_type", "bool"),
            _spec("error_calibration.group_by_landing_policy", "bool"),
            _spec("error_calibration.model_version_policy", "choice", choices=sorted(VALID_ERROR_CALIBRATION_MODEL_VERSION_POLICIES)),
            _spec("error_calibration.aggressiveness_match_required", "bool"),
            _spec("error_calibration.output_units", "choice", choices=["ha"]),
        ],
    ),
    "Edit quality_gates": _make_block_menu(
        "Edit quality_gates",
        "Science-quality gates for QM/AIMAll, FEREBUS, and ARIADNE outputs.",
        [
            _spec("quality_gates.require_readable_aimall_geometry", "bool"),
            _spec("quality_gates.require_finite_iqa", "bool"),
            _spec("quality_gates.require_finite_integration_error", "bool"),
            _spec("quality_gates.max_abs_integration_error", "optional_float"),
            _spec("quality_gates.iqa_energy_recovery_tolerance_ha", "optional_float"),
            _spec("quality_gates.ferebus_min_ext_r2", "optional_float"),
            _spec("quality_gates.ferebus_max_ext_rmse_ha", "optional_float"),
            _spec("quality_gates.ferebus_max_condition_number", "optional_float"),
            _spec("quality_gates.ferebus_max_aggregate_ext_rmse_increase_fraction", "float"),
            _spec("quality_gates.ferebus_max_task_ext_rmse_increase_fraction", "float"),
            _spec("quality_gates.ferebus_regression_abs_tolerance_ha", "float"),
            _spec("quality_gates.ariadne_max_displacement_ang", "optional_float"),
            _spec("quality_gates.ariadne_min_pair_distance_ang", "optional_float"),
        ],
    ),
    "Edit runtime": _make_block_menu(
        "Edit runtime",
        "Daemon runtime resilience and retry controls.",
        [
            _spec("runtime.poll_interval_seconds", "int"),
            _spec("runtime.poll_interval_idle_seconds", "int"),
            _spec("runtime.poll_sacct_empty_max_ticks", "int", prompt="runtime.poll_sacct_empty_max_ticks (0 disables empty-sacct escalation): "),
            _spec("runtime.failure_threshold_fraction", "float", prompt="runtime.failure_threshold_fraction (0.0-1.0): "),
            _spec("runtime.lease_stale_seconds", "int"),
            _spec("runtime.postprocess_settle_attempts", "int"),
            _spec("runtime.postprocess_settle_seconds", "int"),
            _spec("runtime.transient_phase_retry_max", "int"),
            _spec("runtime.poll_sacct_unknown_max_ticks", "int"),
            _spec("runtime.poll_sacct_error_max_ticks", "int"),
            _spec("runtime.poll_sacct_missing_max_ticks", "int"),
            _spec("runtime.poll_squeue_inconclusive_max_ticks", "int"),
            _spec("runtime.halt_on_tick_exception", "bool"),
            _spec("runtime.scheduler_command_timeout_seconds", "int"),
            _spec("runtime.cancellation_confirmation_timeout_seconds", "int"),
            _spec("runtime.journal_max_bytes", "int"),
            _spec("runtime.journal_retained_files", "int"),
            _spec("runtime.lease_heartbeat_seconds", "int"),
            _spec("runtime.lease_heartbeat_failure_max", "int"),
            _spec("runtime.clock_skew_tolerance_seconds", "int"),
            _spec("runtime.background_readiness_timeout_seconds", "int"),
            _spec("runtime.ledger_lock_timeout_seconds", "int"),
        ],
    ),
    "Edit retention": _make_block_menu(
        "Edit retention",
        "Durable campaign checkpoint policy.",
        [
            _spec("retention.checkpoint_destination", "optional_str"),
            _spec("retention.checkpoint_every_iterations", "int"),
            _spec("retention.checkpoint_required", "bool"),
            _spec("retention.checkpoint_verify_after_write", "bool"),
        ],
    ),
}


def _block_submenu_item(label: str):
    return SubmenuItem(label, _BLOCK_MENUS_BY_LABEL[label], edit_campaign_config_menu)


_ACQUISITION_BLOCK_LABELS = (
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
)


edit_acquisition_config_menu = ConsoleMenu(
    title="Edit acquisition",
    subtitle=(
        "Edit the adversarial acquisition objective, local subspace, "
        "movement, gradient, barriers and reference-scale settings.\n"
    ),
    prologue_text="Acquisition config blocks:\n",
    include_parent_menu_options=False,
)


add_items_to_menu(
    edit_acquisition_config_menu,
    [
        SubmenuItem(label, _BLOCK_MENUS_BY_LABEL[label], edit_acquisition_config_menu)
        for label in _ACQUISITION_BLOCK_LABELS
    ],
)


edit_campaign_config_menu_items = [
    FunctionItem(
        "Show dense/internal diagnostic config",
        EditCampaignConfigFunctions.show_current_config,
    ),
    FunctionItem(
        "Show sampling protocol summary",
        EditCampaignConfigFunctions.show_sampling_protocol_summary,
    ),
    FunctionItem("Load from disk", EditCampaignConfigFunctions.load_from_disk),
    FunctionItem("Reset to defaults", EditCampaignConfigFunctions.reset_to_defaults),
    FunctionItem("Validate current config", EditCampaignConfigFunctions.validate_current_config),
    _block_submenu_item("Edit campaign"),
    _block_submenu_item("Edit point_allocation"),
    _block_submenu_item("Edit resource defaults"),
    _block_submenu_item("Edit POLUS resources"),
    _block_submenu_item("Edit Gaussian runtime resources"),
    _block_submenu_item("Edit AIMAll resources"),
    _block_submenu_item("Edit ARIADNE resources"),
    _block_submenu_item("Edit FEREBUS resources"),
    _block_submenu_item("Edit Gaussian block"),
    _block_submenu_item("Edit AIMAll block"),
    _block_submenu_item("Edit seed_selection"),
    _block_submenu_item("Edit anti_overlap"),
    _block_submenu_item("Edit phase_b"),
    _block_submenu_item("Edit geometry_novelty"),
    _block_submenu_item("Edit FEREBUS block"),
    SubmenuItem(
        "Edit acquisition",
        edit_acquisition_config_menu,
        edit_campaign_config_menu,
    ),
    SubmenuItem(
        EDIT_ARIADNE_BLOCK_MENU_DESCRIPTION.title,
        edit_ariadne_block_menu,
        edit_campaign_config_menu,
    ),
    _block_submenu_item("Edit stop"),
    _block_submenu_item("Edit adversarial_safety"),
    _block_submenu_item("Edit error_calibration"),
    _block_submenu_item("Edit quality_gates"),
    _block_submenu_item("Edit runtime"),
    _block_submenu_item("Edit retention"),
    FunctionItem(
        "Show unsaved changes",
        EditCampaignConfigFunctions.show_unsaved_changes,
    ),
    FunctionItem(
        "Show config lock review",
        EditCampaignConfigFunctions.show_config_lock_review,
    ),
    FunctionItem(
        "Show config editability windows",
        EditCampaignConfigFunctions.show_config_editability_windows,
    ),
    FunctionItem(
        "Show pending config changes",
        EditCampaignConfigFunctions.show_pending_yaml_diff,
    ),
    FunctionItem(
        "Discard unsaved changes / reload from disk",
        EditCampaignConfigFunctions.discard_unsaved_changes,
    ),
    FunctionItem(
        "Export dense/internal diagnostic snapshot",
        EditCampaignConfigFunctions.export_dense_config_snapshot,
    ),
    FunctionItem("Save to disk", EditCampaignConfigFunctions.save_to_disk),
]


add_items_to_menu(edit_campaign_config_menu, edit_campaign_config_menu_items)


_sync_options_from_config()
