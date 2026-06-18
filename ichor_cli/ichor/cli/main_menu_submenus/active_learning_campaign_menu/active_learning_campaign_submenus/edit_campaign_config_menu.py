"""Edit campaign config menu -- schema v2.

Field-grouped editor over a single CampaignConfig instance held in module
state. The grouping mirrors the nested block structure of schema v2: one
menu item per top-level block plus a separate submenu tree for deeper
acquisition surface.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ichor.cli.global_menu_variables
import ichor.hpc.global_variables
from consolemenu.items import FunctionItem, SubmenuItem
from ichor.cli.console_menu import ConsoleMenu, add_items_to_menu
from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
    CampaignSelectionError,
    print_campaign_selection_error,
    selected_campaign_dir,
)
from ichor.cli.main_menu_submenus.active_learning_campaign_menu.field_menu import (
    FieldSpec as _FieldSpec,
    edit_field as _shared_edit_field,
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
    VALID_CALIBRATED_ENERGY_UTILITIES,
    VALID_BATCH_POLICIES,
    VALID_DESCRIPTORS,
    VALID_ERROR_CALIBRATION_MODES,
    VALID_ERROR_CALIBRATION_MODEL_VERSION_POLICIES,
    VALID_FULLSPACE_RESIDUAL_SCALES,
    VALID_GAUSSIAN_MEMORY_MODES,
    VALID_GRADIENT_MODES,
    VALID_GRADIENT_PARALLEL_BACKENDS,
    VALID_MODE_WEIGHTING_POLICIES,
    VALID_NEGATIVE_CURVATURE_POLICIES,
    VALID_SEED_SELECTION_STRATEGIES,
    VALID_SPECTRAL_MODES,
    VALID_SPLITS,
    VALID_WARMSTART,
)


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


def get_campaign_config() -> CampaignConfig:
    return _campaign_config


def _replace_campaign_config(cfg, loaded_from):
    global _campaign_config, _loaded_from_path
    _campaign_config = cfg
    _loaded_from_path = loaded_from
    _sync_options_from_config()


@dataclass
class EditCampaignConfigMenuOptions(MenuOptions):
    loaded_from: str = "defaults"
    system_name: str = "SYSTEM"
    max_iterations: int = 50
    n_seeds_per_iteration: int = 50
    phase_b_descriptor: str = "hybrid_alf_rmsd"
    split_strategy: str = "stratified_with_holdout"
    ferebus_warmstart: str = "adaptive"
    ferebus_kernel: str = "rbfc_per"
    acquisition_gradient_mode: str = "cartesian_fd"
    acquisition_max_subspace_dim: int = 6


edit_campaign_config_menu_options = EditCampaignConfigMenuOptions()


def _get_config_value(path: str):
    return get_attr_path(_campaign_config, path)


def _set_config_value(path: str, value):
    set_attr_path(_campaign_config, path, value)
    _sync_options_from_config()


def _edit_field(spec: _FieldSpec):
    _shared_edit_field(spec, _get_config_value, _set_config_value)


def _make_block_menu(title: str, subtitle: str, fields):
    return make_field_menu(
        title,
        subtitle,
        fields,
        _get_config_value,
        _set_config_value,
        prologue_text="Current values for this campaign.yaml block:\n",
    )


def _spec(path: str, input_kind: str, choices=None, transform=None, prompt=None):
    return _shared_spec(path, input_kind, choices, transform, prompt)


def _read_only_spec(path: str):
    return _FieldSpec(path=path, read_only=True)


def _auto_or_int(value):
    text = str(value).strip()
    if text.lower() == "auto":
        return "auto"
    return int(text)


def _sync_options_from_config():
    edit_campaign_config_menu_options.loaded_from = (
        str(_loaded_from_path) if _loaded_from_path is not None else "defaults"
    )
    edit_campaign_config_menu_options.system_name = _campaign_config.system_name
    edit_campaign_config_menu_options.max_iterations = _campaign_config.max_iterations
    edit_campaign_config_menu_options.n_seeds_per_iteration = (
        _campaign_config.seed_selection.n_seeds_per_iteration
    )
    edit_campaign_config_menu_options.phase_b_descriptor = _campaign_config.phase_b.descriptor
    edit_campaign_config_menu_options.split_strategy = _campaign_config.split.strategy
    edit_campaign_config_menu_options.ferebus_warmstart = _campaign_config.ferebus.warmstart
    edit_campaign_config_menu_options.ferebus_kernel = _campaign_config.ferebus.kernel
    edit_campaign_config_menu_options.acquisition_gradient_mode = (
        _campaign_config.acquisition.gradient.mode
    )
    edit_campaign_config_menu_options.acquisition_max_subspace_dim = (
        _campaign_config.acquisition.subspace.max_subspace_dim
    )


def _campaign_yaml_path():
    return selected_campaign_dir() / "campaign.yaml"


def _pause():
    user_input_free_flow("Press enter to return to the menu: ", "")


class EditCampaignConfigFunctions:
    @staticmethod
    def show_current_config():
        import json
        print("Loaded from: " + edit_campaign_config_menu_options.loaded_from)
        print(json.dumps(_campaign_config.to_dict(), indent=2, sort_keys=True))
        _pause()

    @staticmethod
    def show_sampling_protocol_summary():
        print(format_sampling_protocol_summary(_campaign_config))
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
        try:
            cfg = CampaignConfig.from_yaml(yaml_path)
        except Exception as exc:
            print("Failed to load " + str(yaml_path) + ": " + str(exc))
            _pause()
            return
        _replace_campaign_config(cfg, loaded_from=yaml_path)
        ichor.hpc.global_variables.LOGGER.info(
            "Campaign config loaded from " + str(yaml_path)
        )
        print("Loaded.")
        _pause()

    @staticmethod
    def reset_to_defaults():
        _replace_campaign_config(CampaignConfig(), loaded_from=None)
        ichor.hpc.global_variables.LOGGER.info("Campaign config reset to defaults")
        print("Reset.")
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
    def edit_trajectory_pool():
        _campaign_config.trajectory_pool.source_path = user_input_free_flow(
            "trajectory_pool.source_path: ",
            _campaign_config.trajectory_pool.source_path,
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
        r = _campaign_config.resources
        r.partition = user_input_free_flow("resources.partition: ", r.partition)
        r.walltime_hours = user_input_int(
            "resources.walltime_hours: ", r.walltime_hours,
        )
        r.mem_per_cpu = user_input_free_flow(
            "resources.mem_per_cpu (SLURM style, e.g. 4G): ", r.mem_per_cpu,
        )
        r.cpus_per_task = user_input_int(
            "resources.cpus_per_task: ", r.cpus_per_task,
        )
        r.ntasks = user_input_int("resources.ntasks: ", r.ntasks)
        r.aimall_cpus_per_task = user_input_int(
            "resources.aimall_cpus_per_task: ", r.aimall_cpus_per_task,
        )
        r.ariadne_cpus_per_task = user_input_int(
            "resources.ariadne_cpus_per_task: ", r.ariadne_cpus_per_task,
        )
        chosen = user_input_restricted(
            sorted(VALID_GRADIENT_PARALLEL_BACKENDS),
            "resources.gradient_parallel_backend: ",
            r.gradient_parallel_backend,
        )
        if chosen is not None:
            r.gradient_parallel_backend = chosen
        _sync_options_from_config()

    @staticmethod
    def edit_gaussian():
        g = _campaign_config.gaussian
        g.method = user_input_free_flow("gaussian.method: ", g.method)
        g.basis_set = user_input_free_flow("gaussian.basis_set: ", g.basis_set)
        g.charge = user_input_int("gaussian.charge: ", g.charge)
        g.spin_multiplicity = user_input_int(
            "gaussian.spin_multiplicity: ", g.spin_multiplicity,
        )
        g.extra_keywords = user_input_free_flow(
            "gaussian.extra_keywords: ", g.extra_keywords,
        )
        g.nproc = user_input_int("gaussian.nproc: ", g.nproc)
        g.mem = user_input_free_flow(
            "gaussian.mem (Gaussian style, e.g. 8GB): ", g.mem,
        )
        _sync_options_from_config()

    @staticmethod
    def edit_initial_subsample():
        _campaign_config.initial_train_size = user_input_int(
            "initial_train_size: ", _campaign_config.initial_train_size,
        )
        _campaign_config.initial_val_size = user_input_int(
            "initial_val_size: ", _campaign_config.initial_val_size,
        )
        _sync_options_from_config()

    @staticmethod
    def edit_batch_sizing():
        b = _campaign_config.batch_sizing
        chosen = user_input_restricted(
            sorted(VALID_BATCH_POLICIES), "batch_sizing.policy: ", b.policy,
        )
        if chosen is not None:
            b.policy = chosen
        b.floor = user_input_int("batch_sizing.floor: ", b.floor)
        b.cap = user_input_int("batch_sizing.cap: ", b.cap)
        _sync_options_from_config()

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
        a.skip_training_seeds = user_input_bool(
            "anti_overlap.skip_training_seeds: ", a.skip_training_seeds,
        )
        a.recent_seeds_cooldown = user_input_int(
            "anti_overlap.recent_seeds_cooldown: ", a.recent_seeds_cooldown,
        )
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
        pb.min_separation = user_input_float(
            "phase_b.min_separation: ", pb.min_separation,
        )
        _sync_options_from_config()

    @staticmethod
    def edit_split():
        s = _campaign_config.split
        chosen = user_input_restricted(
            sorted(VALID_SPLITS), "split.strategy: ", s.strategy,
        )
        if chosen is not None:
            s.strategy = chosen
        s.train_fraction = user_input_float(
            "split.train_fraction (0.0-1.0): ", s.train_fraction,
        )
        s.val_mid_fraction = user_input_float(
            "split.val_mid_fraction (0.0-1.0): ", s.val_mid_fraction,
        )
        s.high_holdout_fraction = user_input_float(
            "split.high_holdout_fraction (0.0-1.0): ", s.high_holdout_fraction,
        )
        _sync_options_from_config()

    @staticmethod
    def edit_ferebus():
        f = _campaign_config.ferebus
        chosen = user_input_restricted(
            sorted(VALID_WARMSTART), "ferebus.warmstart: ", f.warmstart,
        )
        if chosen is not None:
            f.warmstart = chosen
        f.warmstart_streak = user_input_int(
            "ferebus.warmstart_streak: ", f.warmstart_streak,
        )
        f.kernel = user_input_free_flow(
            "ferebus.kernel (e.g. rbfc_per, rbf_per): ", f.kernel,
        )
        f.loss = user_input_free_flow(
            "ferebus.loss (e.g. huber, mse, mae): ", f.loss,
        )
        f.nagents = user_input_int("ferebus.nagents: ", f.nagents)
        f.maxiter = user_input_int("ferebus.maxiter: ", f.maxiter)
        f.is_constant_noise = user_input_bool(
            "ferebus.is_constant_noise: ", f.is_constant_noise,
        )
        f.scaling = user_input_bool("ferebus.scaling: ", f.scaling)
        f.full_ARD = user_input_bool("ferebus.full_ARD: ", f.full_ARD)
        raw_props = user_input_free_flow(
            "ferebus.properties (comma-separated, e.g. iqa,q00): ",
            ",".join(f.properties),
        )
        f.properties = [p.strip() for p in str(raw_props).split(",") if p.strip()]
        f.train_fraction = user_input_float(
            "ferebus.train_fraction: ", f.train_fraction,
        )
        f.int_val_fraction = user_input_float(
            "ferebus.int_val_fraction: ", f.int_val_fraction,
        )
        f.ext_val_fraction = user_input_float(
            "ferebus.ext_val_fraction: ", f.ext_val_fraction,
        )
        _sync_options_from_config()

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
    def edit_outlier_filter():
        of = _campaign_config.outlier_filter
        of.enabled = user_input_bool(
            "outlier_filter.enabled: ", of.enabled,
        )
        of.energy_z_threshold = user_input_float(
            "outlier_filter.energy_z_threshold: ", of.energy_z_threshold,
        )
        of.per_atom_rmsd_z_threshold = user_input_float(
            "outlier_filter.per_atom_rmsd_z_threshold: ",
            of.per_atom_rmsd_z_threshold,
        )
        _sync_options_from_config()

    @staticmethod
    def save_to_disk():
        try:
            _campaign_config._validate()
        except ConfigValidationError as exc:
            print("Validation failed; campaign.yaml NOT written.")
            print("  " + str(exc))
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
        try:
            _campaign_config.to_yaml(target)
        except Exception as exc:
            print("Failed to write " + str(target) + ": " + str(exc))
            _pause()
            return
        global _loaded_from_path
        _loaded_from_path = target
        _sync_options_from_config()
        ichor.hpc.global_variables.LOGGER.info(
            "Campaign config saved to " + str(target)
        )
        print("Wrote " + str(target))
        _pause()


def _unsupported_sequential_field_editor():
    raise RuntimeError(
        "Sequential campaign config editors are no longer supported. "
        "Use the block field menus so current values remain visible and every "
        "campaign.yaml field is edited through one authoritative path."
    )


for _legacy_editor_name in (
    "edit_campaign_identity",
    "edit_trajectory_pool",
    "edit_iteration_control",
    "edit_resources",
    "edit_gaussian",
    "edit_initial_subsample",
    "edit_batch_sizing",
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
    "edit_outlier_filter",
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
    "Edit campaign identity": _make_block_menu(
        "Edit Campaign Identity",
        "Top-level campaign identity fields.",
        [_read_only_spec("schema_version"), _spec("system_name", "str")],
    ),
    "Edit trajectory_pool": _make_block_menu(
        "Edit trajectory_pool",
        "Trajectory source used when importing the campaign pool.",
        [_spec("trajectory_pool.source_path", "str")],
    ),
    "Edit iteration control": _make_block_menu(
        "Edit Iteration Control",
        "Daemon iteration and polling controls.",
        [
            _spec("max_iterations", "int"),
            _spec("poll_interval_seconds", "int"),
            _spec("poll_interval_idle_seconds", "int"),
            _spec("poll_sacct_empty_max_ticks", "int", prompt="poll_sacct_empty_max_ticks (0 disables empty-sacct escalation): "),
        ],
    ),
    "Edit initial sub-sample sizes": _make_block_menu(
        "Edit Initial Sub-Sample Sizes",
        "Initial POLUS train/validation sample sizes.",
        [
            _spec("initial_train_size", "int"),
            _spec("initial_val_size", "int"),
        ],
    ),
    "Edit resources": _make_block_menu(
        "Edit resources",
        "SLURM resources used by live backend phases.",
        [
            _spec("resources.partition", "str"),
            _spec("resources.walltime_hours", "int"),
            _spec("resources.polus_walltime_hours", "optional_int"),
            _spec("resources.gaussian_walltime_hours", "optional_int"),
            _spec("resources.aimall_walltime_hours", "optional_int"),
            _spec("resources.ariadne_walltime_hours", "optional_int"),
            _spec("resources.ferebus_walltime_hours", "optional_int"),
            _spec("resources.mem_per_cpu", "str", prompt="resources.mem_per_cpu (SLURM style, e.g. 4G): "),
            _spec("resources.cpus_per_task", "int"),
            _spec("resources.ntasks", "int"),
            _spec("resources.aimall_cpus_per_task", "int"),
            _spec("resources.ariadne_cpus_per_task", "int"),
            _spec("resources.array_concurrency_limit", "optional_int"),
            _spec("resources.gradient_parallel_backend", "choice", choices=sorted(VALID_GRADIENT_PARALLEL_BACKENDS)),
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
            _spec("gaussian.extra_keywords", "str"),
            _spec("gaussian.nproc", "int"),
            _spec("gaussian.mem", "str", prompt="gaussian.mem (Gaussian style, e.g. 8GB): "),
            _spec("gaussian.memory_mode", "choice", choices=sorted(VALID_GAUSSIAN_MEMORY_MODES)),
            _spec("gaussian.memory_fraction_of_slurm", "float"),
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
    "Edit batch_sizing": _make_block_menu(
        "Edit batch_sizing",
        "Batch size policy for later active-learning iterations.",
        [
            _spec("batch_sizing.policy", "choice", choices=sorted(VALID_BATCH_POLICIES)),
            _spec("batch_sizing.floor", "int"),
            _spec("batch_sizing.cap", "int"),
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
        ],
    ),
    "Edit anti_overlap": _make_block_menu(
        "Edit anti_overlap",
        "Candidate anti-overlap and ARIADNE movement filters.",
        [
            _spec("anti_overlap.skip_training_seeds", "bool"),
            _spec("anti_overlap.recent_seeds_cooldown", "int"),
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
            _spec("phase_b.min_separation", "float"),
        ],
    ),
    "Edit split": _make_block_menu(
        "Edit split",
        "High-level train/validation split strategy.",
        [
            _spec("split.strategy", "choice", choices=sorted(VALID_SPLITS)),
            _spec("split.train_fraction", "float", prompt="split.train_fraction (0.0-1.0): "),
            _spec("split.val_mid_fraction", "float", prompt="split.val_mid_fraction (0.0-1.0): "),
            _spec("split.high_holdout_fraction", "float", prompt="split.high_holdout_fraction (0.0-1.0): "),
        ],
    ),
    "Edit FEREBUS block": _make_block_menu(
        "Edit FEREBUS Block",
        "FEREBUS training and dataset split controls.",
        [
            _spec("ferebus.warmstart", "choice", choices=sorted(VALID_WARMSTART)),
            _spec("ferebus.warmstart_streak", "int"),
            _spec("ferebus.kernel", "str", prompt="ferebus.kernel (e.g. rbfc_per, rbf_per): "),
            _spec("ferebus.loss", "str", prompt="ferebus.loss (e.g. huber, mse, mae): "),
            _spec("ferebus.nagents", "int"),
            _spec("ferebus.maxiter", "int"),
            _spec("ferebus.is_constant_noise", "bool"),
            _spec("ferebus.scaling", "bool"),
            _spec("ferebus.full_ARD", "bool"),
            _spec("ferebus.properties", "csv_list", prompt="ferebus.properties (comma-separated, e.g. iqa,q00): "),
            _spec("ferebus.train_fraction", "float"),
            _spec("ferebus.int_val_fraction", "float"),
            _spec("ferebus.ext_val_fraction", "float"),
        ],
    ),
    "Edit robustness": _make_block_menu(
        "Edit Robustness",
        "Global failure and force sanity thresholds.",
        [
            _spec("failure_threshold_fraction", "float", prompt="failure_threshold_fraction (0.0-1.0): "),
            _spec("max_acquisition_grad_per_ang", "optional_float", prompt="max_acquisition_grad_per_ang (acquisition units/Angstrom, null uses deprecated alias): "),
            _spec("max_force_per_atom_ha_per_ang", "float", prompt="max_force_per_atom_ha_per_ang (deprecated acquisition-gradient alias): "),
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
            _spec("acquisition.fullspace_confinement.residual_scale", "choice", choices=sorted(VALID_FULLSPACE_RESIDUAL_SCALES)),
            _spec("acquisition.fullspace_confinement.fixed_residual_scale_ang", "optional_float"),
            _spec("acquisition.fullspace_confinement.rmsd_scale_ang", "float"),
            _spec("acquisition.fullspace_confinement.min_residual_scale_ang", "float"),
            _spec("acquisition.fullspace_confinement.failure_penalty", "float"),
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
    "Edit outlier_filter": _make_block_menu(
        "Edit outlier_filter",
        "Pre-Phase-A trajectory outlier filter thresholds.",
        [
            _spec("outlier_filter.enabled", "bool"),
            _spec("outlier_filter.energy_z_threshold", "float"),
            _spec("outlier_filter.per_atom_rmsd_z_threshold", "float"),
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
            _spec("adversarial_safety.min_whitened_distance", "float"),
            _spec("adversarial_safety.max_whitened_distance", "float"),
            _spec("adversarial_safety.enforce_min_whitened_distance", "bool"),
            _spec("adversarial_safety.max_predicted_energy_delta_ha", "optional_float"),
            _spec("adversarial_safety.max_energy_variance", "optional_float"),
            _spec("adversarial_safety.max_chemistry_penalty", "optional_float"),
            _spec("adversarial_safety.phase_b_filter_enabled", "bool"),
        ],
    ),
    "Edit error_calibration": _make_block_menu(
        "Edit error_calibration",
        "Empirical mapping from raw uncertainty to realised IQA error.",
        [
            _spec("error_calibration.enabled", "bool"),
            _spec("error_calibration.mode", "choice", choices=sorted(VALID_ERROR_CALIBRATION_MODES)),
            _spec("error_calibration.min_records_to_apply", "int"),
            _spec("error_calibration.n_bins", "int"),
            _spec("error_calibration.min_bin_records", "int"),
            _spec("error_calibration.apply_strength", "float"),
            _spec("error_calibration.group_by_atom_type", "bool"),
            _spec("error_calibration.group_by_landing_policy", "bool"),
            _spec("error_calibration.model_version_policy", "choice", choices=sorted(VALID_ERROR_CALIBRATION_MODEL_VERSION_POLICIES)),
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
            _spec("quality_gates.ariadne_max_displacement_ang", "optional_float"),
            _spec("quality_gates.ariadne_min_pair_distance_ang", "optional_float"),
        ],
    ),
    "Edit runtime": _make_block_menu(
        "Edit runtime",
        "Daemon runtime resilience and retry controls.",
        [
            _spec("runtime.lease_stale_seconds", "int"),
            _spec("runtime.postprocess_settle_attempts", "int"),
            _spec("runtime.postprocess_settle_seconds", "int"),
            _spec("runtime.transient_phase_retry_max", "int"),
            _spec("runtime.poll_sacct_unknown_max_ticks", "int"),
            _spec("runtime.poll_sacct_missing_max_ticks", "int"),
        ],
    ),
}


def _block_submenu_item(label: str):
    return SubmenuItem(label, _BLOCK_MENUS_BY_LABEL[label], edit_campaign_config_menu)


edit_campaign_config_menu_items = [
    FunctionItem("Show current config", EditCampaignConfigFunctions.show_current_config),
    FunctionItem(
        "Show sampling protocol summary",
        EditCampaignConfigFunctions.show_sampling_protocol_summary,
    ),
    FunctionItem("Load from disk", EditCampaignConfigFunctions.load_from_disk),
    FunctionItem("Reset to defaults", EditCampaignConfigFunctions.reset_to_defaults),
    FunctionItem("Validate current config", EditCampaignConfigFunctions.validate_current_config),
    _block_submenu_item("Edit campaign identity"),
    _block_submenu_item("Edit trajectory_pool"),
    _block_submenu_item("Edit iteration control"),
    _block_submenu_item("Edit initial sub-sample sizes"),
    _block_submenu_item("Edit resources"),
    _block_submenu_item("Edit Gaussian block"),
    _block_submenu_item("Edit AIMAll block"),
    _block_submenu_item("Edit batch_sizing"),
    _block_submenu_item("Edit seed_selection"),
    _block_submenu_item("Edit anti_overlap"),
    _block_submenu_item("Edit phase_b"),
    _block_submenu_item("Edit split"),
    _block_submenu_item("Edit FEREBUS block"),
    _block_submenu_item("Edit robustness"),
    _block_submenu_item("Edit acquisition core"),
    _block_submenu_item("Edit acquisition.subspace"),
    _block_submenu_item("Edit acquisition.weights"),
    _block_submenu_item("Edit acquisition.spectral"),
    _block_submenu_item("Edit acquisition.calibrated_energy"),
    _block_submenu_item("Edit acquisition.fullspace_confinement"),
    _block_submenu_item("Edit acquisition.gradient"),
    _block_submenu_item("Edit acquisition.barrier"),
    _block_submenu_item("Edit acquisition.stencils"),
    _block_submenu_item("Edit acquisition.references"),
    _block_submenu_item("Edit stop"),
    _block_submenu_item("Edit outlier_filter"),
    _block_submenu_item("Edit adversarial_safety"),
    _block_submenu_item("Edit error_calibration"),
    _block_submenu_item("Edit quality_gates"),
    _block_submenu_item("Edit runtime"),
    SubmenuItem(
        EDIT_ARIADNE_BLOCK_MENU_DESCRIPTION.title,
        edit_ariadne_block_menu,
        edit_campaign_config_menu,
    ),
    FunctionItem("Save to disk", EditCampaignConfigFunctions.save_to_disk),
]


add_items_to_menu(edit_campaign_config_menu, edit_campaign_config_menu_items)


_sync_options_from_config()
