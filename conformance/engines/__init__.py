"""
Engine registry.

Every engine module exposes the same small interface:

    NAME         short key, used in file paths and URLs
    DISPLAY      human-readable name
    HOMEPAGE     project URL
    ROUTES       {route: description}. A route is one way a user can ask the
                 engine for accelerations; an engine with two documented ways
                 is scored on both, separately.
    FAMILIES     {route: set of case families that route models}
    NOT_MODELLED {family: reason} for families no route models, so the
                 published page can say why a row is empty
    version()    installed version string
    build(route, case) -> fn
                 fn(state) returns the generalised accelerations, or a pair
                 (accelerations, state_actually_used) when the engine moves the
                 state it was given (Chrono's assembly does). Raising is how an
                 engine refuses, and refusing is scored as refusing.

Adapters translate conventions (state ordering, absolute versus relative
angles, axis directions) and nothing else. They must not repair an engine's
answer: an adapter that fixes what the engine got wrong would hide exactly
what this service exists to show.
"""

from __future__ import annotations

import importlib

ENGINES = {
    "sympy": "conformance.engines.sympy_engine",
    "mechanicsdsl": "conformance.engines.mechanicsdsl_engine",
    "drake": "conformance.engines.drake_engine",
    "chrono": "conformance.engines.chrono_engine",
}


def load(name: str):
    if name not in ENGINES:
        raise KeyError(f"unknown engine {name!r}; known: {sorted(ENGINES)}")
    return importlib.import_module(ENGINES[name])
