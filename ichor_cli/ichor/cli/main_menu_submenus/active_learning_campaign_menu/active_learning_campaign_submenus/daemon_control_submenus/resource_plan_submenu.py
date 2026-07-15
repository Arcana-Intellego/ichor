"""Current-value options for the read-only daemon resource plan."""
from __future__ import annotations

from dataclasses import dataclass

from consolemenu.items import FunctionItem
from ichor.cli.main_menu_submenus.active_learning_campaign_menu.campaign_context import (
    CampaignSelectionError,
    campaign_dir_ns,
    print_campaign_selection_error,
)
from ichor.cli.main_menu_submenus.active_learning_campaign_menu.field_menu import (
    make_field_menu,
    spec,
)
from ichor.cli.menu_description import MenuDescription
from ichor.cli.useful_functions import user_input_free_flow


RESOURCE_PLAN_MENU_DESCRIPTION = MenuDescription(
    "Resource Plan",
    subtitle="Inspect immutable or prospective phase resources without changing campaign state.\n",
)


@dataclass
class ResourcePlanMenuOptions:
    scope: str = "current"
    phase: str = "PHASE_A_DIVERSITY"
    iteration: int | None = None
    json: bool = False


resource_plan_options = ResourcePlanMenuOptions()


def _get(path: str):
    return getattr(resource_plan_options, path)


def _set(path: str, value):
    setattr(resource_plan_options, path, value)


def _run_resource_plan():
    from ichor.hpc.active_learning.cli import cmd_resource_plan

    try:
        ns = campaign_dir_ns()
    except CampaignSelectionError as exc:
        print_campaign_selection_error(exc)
        user_input_free_flow("Press enter to return to the menu: ", "")
        return
    ns.all = resource_plan_options.scope == "all"
    ns.phase = (
        resource_plan_options.phase
        if resource_plan_options.scope == "phase"
        else None
    )
    ns.iteration = resource_plan_options.iteration
    ns.json = bool(resource_plan_options.json)
    rc = cmd_resource_plan(ns)
    if rc != 0:
        print("resource-plan returned exit code " + str(rc))
    user_input_free_flow("Press enter to return to the menu: ", "")


def resource_plan_menu(parent=None):
    from ichor.hpc.active_learning.daemon.state import CampaignPhase

    return make_field_menu(
        RESOURCE_PLAN_MENU_DESCRIPTION.title,
        RESOURCE_PLAN_MENU_DESCRIPTION.subtitle,
        [
            spec("scope", "choice", choices=["current", "phase", "all"]),
            spec(
                "phase",
                "choice",
                choices=[phase.value for phase in CampaignPhase],
            ),
            spec("iteration", "optional_int"),
            spec("json", "bool"),
        ],
        _get,
        _set,
        extra_items=[FunctionItem("Show resource plan", _run_resource_plan)],
    )


RESOURCE_PLAN_MENU = resource_plan_menu()
