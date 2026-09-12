"""Shared immutable model base."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    """Immutable, strict base for every value object in the project.

    ``frozen=True`` because these objects are passed between pipeline stages
    and an in-place mutation would be invisible at the call site.

    ``extra="forbid"`` matters more than it looks: a silently ignored key is
    indistinguishable from a knob that does nothing.

    ``protected_namespaces=()`` disables pydantic's warning about fields
    beginning with ``model_``. Here ``model_id`` means a retrieval model, and
    renaming the field to dodge a framework warning would be the tail wagging
    the dog.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())
