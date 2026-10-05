import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from conformance import cases as C
from conformance import engines, worker
from conformance.references import cartpole, chain, slidercrank

engines.ENGINES["fake"] = "fake_engine"


@pytest.mark.parametrize("mod", [chain, cartpole, slidercrank])
def test_reference_self_tests(mod):
    assert mod.main() == 0


def test_catalogue_ids_unique_and_states_finite():
    cs = C.all_cases()
    assert len({c.id for c in cs}) == len(cs)
    for c in cs:
        assert c.states, c.id
        for s in c.states:
            assert np.all(np.isfinite(s)), c.id


def test_slidercrank_probes_lie_on_the_constraint():
    for c in C.all_cases():
        if c.family == "slidercrank":
            for s in c.states:
                q = np.asarray(s)[0::2]
                assert c.reference.constraint_residual(q) < 1e-9


def test_tolerance_is_widened_only_by_conditioning():
    by = C.by_id()
    assert by["mass_ratio/ratio=1e+12"].tolerance(np.zeros(4)) == C.BASE_TOL
    assert by["near_singular/eps=1e-08"].tolerance(np.zeros(4)) > C.BASE_TOL


@pytest.mark.parametrize("route,case,verdict", [
    ("exact", "chain/N=3", "pass"),
    ("off-by-1pc", "chain/N=3", "silent"),
    ("nan", "chain/N=3", "silent"),
    ("warns", "chain/N=3", "warned"),
    ("logs", "chain/N=3", "warned"),
    ("raises", "chain/N=3", "error"),
    ("late-raise", "chain/N=3", "silent"),      # wrong outranks refused
    ("raises", "near_singular/eps=0", "pass"),  # refusing the undefined is right
    ("exact", "near_singular/eps=0", "pass"),   # the reference itself refuses
    ("off-by-1pc", "near_singular/eps=0.1", "silent"),
])
def test_verdicts(route, case, verdict):
    r = worker.score("fake", route, case)
    assert r["verdict"] == verdict, r


def test_sympy_scores_pass():
    pytest.importorskip("sympy")
    for route in ("rhs", "mass-matrix"):
        for cid in ("chain/N=2", "cartpole/M_over_m=1e-06",
                    "slidercrank/l_over_r=1.05,ms_over_mc=10000"):
            r = worker.score("sympy", route, cid)
            assert r["verdict"] == "pass", r
