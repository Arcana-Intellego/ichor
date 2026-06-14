"""Shared current-value field menus for active-learning CLI screens."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from consolemenu.items import FunctionItem
from ichor.cli.console_menu import ConsoleMenu, add_items_to_menu
from ichor.cli.menu_options import MenuOptions
from ichor.cli.useful_functions import (
    user_input_bool,
    user_input_float,
    user_input_free_flow,
    user_input_int,
    user_input_restricted,
)


@dataclass(frozen=True)
class FieldSpec:
    path: str
    input_kind: str = "str"
    item_text: Optional[str] = None
    prompt: Optional[str] = None
    choices: Optional[Sequence[str]] = None
    transform: Optional[Callable[[Any], Any]] = None
    formatter: Optional[Callable[[Any], str]] = None
    read_only: bool = False
    display_path: Optional[str] = None


class FieldMenuOptions(MenuOptions):
    def __init__(
        self,
        fields: Sequence[FieldSpec],
        get_value: Callable[[str], Any],
        title: str = "",
    ):
        self.title = title
        self.fields = list(fields)
        self.get_value = get_value

    def __str__(self):
        lines = []
        for spec in self.fields:
            value = self.get_value(spec.path)
            rendered = (
                spec.formatter(value)
                if spec.formatter is not None
                else format_field_value(value)
            )
            label = spec.display_path or spec.path
            if spec.read_only:
                rendered += " (read-only)"
            lines.append("-- " + label + ": " + rendered + "\n")
        return "".join(lines)


def format_field_value(value):
    if isinstance(value, list):
        return ",".join(str(v) for v in value) if value else "<blank>"
    if value is None:
        return "null"
    if isinstance(value, str) and value == "":
        return "<blank>"
    return str(value)


def _csv_default(value):
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return "" if value is None else str(value)


def get_attr_path(root, path: str):
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def set_attr_path(root, path: str, value):
    parts = path.split(".")
    obj = root
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def user_input_optional_float(prompt: str, default):
    while True:
        try:
            raw = input(prompt + " (float, blank keeps current, null clears): ")
        except EOFError:
            return default
        if raw == "":
            return default
        if raw.strip().lower() in {"none", "null"}:
            return None
        try:
            return float(raw)
        except ValueError:
            pass


def user_input_optional_int(prompt: str, default):
    while True:
        try:
            raw = input(prompt + " (int, blank keeps current, null clears): ")
        except EOFError:
            return default
        if raw == "":
            return default
        if raw.strip().lower() in {"none", "null"}:
            return None
        try:
            return int(raw)
        except ValueError:
            pass


def user_input_clearable_str(prompt: str, default):
    while True:
        try:
            raw = input(prompt + " (blank keeps current, null clears): ")
        except EOFError:
            return default
        if raw == "":
            return default
        if raw.strip().lower() in {"none", "null"}:
            return ""
        return raw


def edit_field(
    spec: FieldSpec,
    get_value: Callable[[str], Any],
    set_value: Callable[[str, Any], None],
):
    if spec.read_only:
        return
    current = get_value(spec.path)
    prompt = spec.prompt or (spec.path + ": ")
    if spec.input_kind == "str":
        value = user_input_free_flow(prompt, current)
    elif spec.input_kind == "clearable_str":
        value = user_input_clearable_str(prompt, current)
    elif spec.input_kind == "int":
        value = user_input_int(prompt, current)
    elif spec.input_kind == "float":
        value = user_input_float(prompt, current)
    elif spec.input_kind == "optional_float":
        value = user_input_optional_float(prompt, current)
    elif spec.input_kind == "optional_int":
        value = user_input_optional_int(prompt, current)
    elif spec.input_kind == "bool":
        value = user_input_bool(prompt, current)
    elif spec.input_kind == "choice":
        value = user_input_restricted(list(spec.choices or []), prompt, current)
    elif spec.input_kind == "csv_list":
        raw = user_input_free_flow(prompt, _csv_default(current))
        value = [part.strip() for part in str(raw).split(",") if part.strip()]
    else:
        raise ValueError("unknown field menu input kind: " + spec.input_kind)
    if spec.transform is not None and value is not None:
        value = spec.transform(value)
    set_value(spec.path, value)


def make_field_item(
    spec: FieldSpec,
    get_value: Callable[[str], Any],
    set_value: Callable[[str, Any], None],
):
    text = spec.item_text or ("Set " + spec.path.split(".")[-1])
    return FunctionItem(text, lambda spec=spec: edit_field(spec, get_value, set_value))


def make_field_menu(
    title: str,
    subtitle: str,
    fields: Sequence[FieldSpec],
    get_value: Callable[[str], Any],
    set_value: Optional[Callable[[str, Any], None]] = None,
    prologue_text: str = "Current values:\n",
    extra_items: Optional[Sequence] = None,
):
    menu = ConsoleMenu(
        this_menu_options=FieldMenuOptions(fields, get_value, title=title),
        title=title,
        subtitle=subtitle,
        prologue_text=prologue_text,
    )
    items = []
    if set_value is not None:
        items.extend(
            make_field_item(spec, get_value, set_value)
            for spec in fields
            if not spec.read_only
        )
    if extra_items:
        items.extend(extra_items)
    add_items_to_menu(menu, items)
    return menu


def spec(
    path: str,
    input_kind: str,
    choices=None,
    transform=None,
    prompt=None,
    item_text=None,
    formatter=None,
    display_path=None,
):
    return FieldSpec(
        path=path,
        input_kind=input_kind,
        item_text=item_text,
        choices=choices,
        transform=transform,
        prompt=prompt or (path + ": "),
        formatter=formatter,
        display_path=display_path,
    )
