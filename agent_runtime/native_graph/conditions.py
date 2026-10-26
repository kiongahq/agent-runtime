"""Deterministic condition grammar: ``operator:argument``.

No expression evaluation: each operator is a fixed comparison. ``matches`` uses
Python ``re`` with a 200-character pattern cap and is applied to at most 32 KiB of
input, which bounds backtracking cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

MAX_PATTERN = 200
MAX_SUBJECT = 32 * 1024
WITH_ARGUMENT = {"contains", "not_contains", "equals", "matches", "longer_than"}
WITHOUT_ARGUMENT = {"empty", "not_empty"}


class ConditionError(ValueError):
    pass


@dataclass(frozen=True)
class Condition:
    op: str
    argument: str
    test: Callable[[str], bool]

    def __call__(self, value: str) -> bool:
        return self.test(value[:MAX_SUBJECT])


def parse(expression: str) -> Condition:
    text = (expression or "").strip()
    op, separator, argument = text.partition(":")
    op = op.strip()
    if op in WITHOUT_ARGUMENT:
        if separator and argument.strip():
            raise ConditionError(f"{op} takes no argument")
        if op == "empty":
            return Condition(op, "", lambda value: not value.strip())
        return Condition(op, "", lambda value: bool(value.strip()))
    if op not in WITH_ARGUMENT:
        raise ConditionError(
            "Use operator:argument with contains, not_contains, equals, matches, "
            "longer_than, empty or not_empty"
        )
    if not separator or argument == "":
        raise ConditionError(f"{op} needs an argument")
    if op == "contains":
        needle = argument.lower()
        return Condition(op, argument, lambda value: needle in value.lower())
    if op == "not_contains":
        needle = argument.lower()
        return Condition(op, argument, lambda value: needle not in value.lower())
    if op == "equals":
        target = argument.strip().lower()
        return Condition(op, argument, lambda value: value.strip().lower() == target)
    if op == "longer_than":
        try:
            limit = int(argument.strip())
        except ValueError:
            raise ConditionError("longer_than needs an integer argument") from None
        if limit < 0:
            raise ConditionError("longer_than needs a non-negative integer")
        return Condition(op, argument, lambda value: len(value) > limit)
    if len(argument) > MAX_PATTERN:
        raise ConditionError(f"matches patterns are limited to {MAX_PATTERN} characters")
    try:
        pattern = re.compile(argument)
    except re.error as error:
        raise ConditionError(f"invalid regular expression: {error}") from None
    return Condition(op, argument, lambda value: pattern.search(value) is not None)
