"""
The fixed catalogue of conformance cases.

A case is one mechanical system with fixed parameters, plus a fixed list of
probe states. At every probe state the engine is asked for the generalised
accelerations and the answer is compared with the closed-form reference.

The catalogue is deliberately small enough to run in a few minutes of free CI
time per engine release, and deliberately includes the configurations that
break solvers: extreme mass ratios, a mass matrix approaching singularity,
aligned and fully extended chains, and slider-crank dead centre.

Changing this file changes what every published result means. Bump
CATALOGUE_VERSION when you do, so old and new results are never compared as
if they were the same test.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .references import cartpole as RCP
from .references import chain as RCH
from .references import slidercrank as RSC

CATALOGUE_VERSION = "1"

# The study's metric: max_i |a_i - r_i| / max(|r_i|, 1).
BASE_TOL = 1e-8
# Floor on the tolerance from the reference's own conditioning; see tolerance().
COND_FACTOR = 1e-14


def relerr(a, r) -> float:
    a = np.asarray(a, dtype=float)
    r = np.asarray(r, dtype=float)
    return float(np.max(np.abs(a - r) / np.maximum(np.abs(r), 1.0)))


def scaled_condition(M: np.ndarray) -> float:
    """Condition number of M after symmetric diagonal (Jacobi) scaling.

    Plain cond(M) punishes a harmless change of units: diag(1, 1e12) has
    cond 1e12 but is solved exactly. Scaling by the diagonal first leaves only
    the conditioning that genuinely limits how accurately anyone, the
    reference included, can solve M a = f.
    """
    d = np.sqrt(np.abs(np.diag(M)))
    d[d == 0] = 1.0
    S = M / np.outer(d, d)
    return float(np.linalg.cond(S))


@dataclass
class Case:
    id: str                       # stable identifier, used as a key in results
    family: str                   # chain | near_singular | mass_ratio | cartpole | slidercrank
    params: Dict[str, float]
    description: str
    reference: object             # has .accel(state) -> ndarray
    states: List[np.ndarray]
    expects_refusal: bool = False # genuinely degenerate: the right answer is an error
    conditioning: Optional[Callable[[np.ndarray], float]] = None
    tags: List[str] = field(default_factory=list)

    def tolerance(self, state) -> float:
        """BASE_TOL, widened only where the reference itself is ill-conditioned.

        A backward-stable solve of a system with scaled condition number k is
        accurate to roughly k * 1e-16. Demanding 1e-8 agreement where k = 1e9
        would fail correct engines for the reference's limitations, so the
        tolerance is max(1e-8, 1e-14 * k): two orders of magnitude of headroom
        over the rounding bound.
        """
        if self.conditioning is None:
            return BASE_TOL
        return max(BASE_TOL, COND_FACTOR * self.conditioning(state))

    def summary(self) -> dict:
        return {"id": self.id, "family": self.family, "params": self.params,
                "description": self.description,
                "expects_refusal": self.expects_refusal,
                "n_probes": len(self.states), "tags": self.tags}


def _rng(case_id: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(case_id.encode()))


# ---------------------------------------------------------------------------
# Family builders
# ---------------------------------------------------------------------------

def _chain(N: int) -> Case:
    cid = f"chain/N={N}"
    ref = RCH.NLinkChain(N)
    rng = _rng(cid)
    states = [rng.uniform(-0.5, 0.5, 2 * N) for _ in range(8)]
    states += [rng.uniform(-math.pi, math.pi, 2 * N) * np.tile([1.0, 2.0], N)
               for _ in range(6)]
    aligned = np.zeros(2 * N)                 # fully extended, hanging straight
    aligned[1::2] = 1.0
    states.append(aligned)
    inverted = np.zeros(2 * N)                # fully extended, balanced upright
    inverted[0::2] = math.pi
    inverted[1::2] = rng.uniform(-1, 1, N)
    states.append(inverted)
    folded = np.zeros(2 * N)                  # alternating links folded back
    folded[0::2] = np.where(np.arange(N) % 2 == 0, 0.4, 0.4 + math.pi)
    states.append(folded)

    def cond(s, ref=ref):
        return scaled_condition(ref.mass_matrix(np.asarray(s)[0::2]))
    return Case(cid, "chain", {"N": N},
                f"Planar {N}-link pendulum, unit point masses on massless rods",
                ref, states, conditioning=cond,
                tags=["tree", "revolute"] + (["long-chain"] if N >= 8 else []))


def _near_singular(eps: float) -> Case:
    cid = f"near_singular/eps={eps:g}"
    ref = RCH.near_singular_system(eps)
    rng = _rng(cid)
    states = [rng.uniform(-1, 1, 4) for _ in range(8)]
    k = scaled_condition(ref.M) if eps > 0 else math.inf
    return Case(cid, "near_singular", {"eps": eps},
                ("Two masses, kinetic coupling (1-eps): mass matrix "
                 + ("exactly singular -- refusing is correct"
                    if eps == 0 else f"condition number {k:.1e}")),
                ref, states, expects_refusal=(eps == 0.0),
                conditioning=(lambda s, k=k: k), tags=["linear", "degenerate"])


def _mass_ratio(ratio: float) -> Case:
    cid = f"mass_ratio/ratio={ratio:g}"
    ref = RCH.mass_ratio_system(ratio)
    rng = _rng(cid)
    states = [rng.uniform(-1, 1, 4) for _ in range(8)]
    return Case(cid, "mass_ratio", {"ratio": ratio},
                f"Two masses joined by springs, m2/m1 = {ratio:g}",
                ref, states, conditioning=lambda s, M=ref.M: scaled_condition(M),
                tags=["linear", "mass-ratio"])


def _cartpole(mass_ratio: float) -> Case:
    cid = f"cartpole/M_over_m={mass_ratio:g}"
    ref = RCP.CartPole(mass_ratio)
    rng = _rng(cid)
    states = []
    for th in (0.0, 0.3, 1.0, 2.0, math.pi, 3.0):   # 0 and pi: det M collapses
        states.append(np.array([rng.uniform(-1, 1), rng.uniform(-2, 2),
                                th, rng.uniform(-3, 3)]))
    states += [np.concatenate([[rng.uniform(-1, 1), rng.uniform(-2, 2)],
                               [rng.uniform(-math.pi, math.pi), rng.uniform(-3, 3)]])
               for _ in range(4)]
    # Interleaved [x, xd, th, thd].
    return Case(cid, "cartpole", {"M_over_m": mass_ratio},
                f"Cart-pole, cart/pole mass ratio {mass_ratio:g}",
                ref, states,
                conditioning=lambda s, ref=ref: scaled_condition(
                    ref.mass_matrix(np.asarray(s)[0::2])),
                tags=["tree", "prismatic", "revolute"]
                + (["degenerate"] if mass_ratio < 1e-3 else []))


def _slidercrank(ratio: float, mass_ratio: float) -> Case:
    cid = f"slidercrank/l_over_r={ratio:g},ms_over_mc={mass_ratio:g}"
    ref = RSC.SliderCrank(ratio, mass_ratio)
    # Probe states must lie on the constraint manifold, so they come from the
    # reference's own consistent-initial-state routine. 0 and pi are dead centre.
    states = [ref.initial_state(th, thd)
              for th in (0.0, 0.3, 1.0, 2.0, math.pi, 4.0)
              for thd in (4.0, -1.5)]
    return Case(cid, "slidercrank", {"l_over_r": ratio, "ms_over_mc": mass_ratio},
                (f"Slider-crank (closed loop), rod/crank {ratio:g}, "
                 f"slider/crank mass {mass_ratio:g}, probed through dead centre"),
                ref, states, tags=["loop", "prismatic", "revolute", "constraint"])


def all_cases() -> List[Case]:
    cases: List[Case] = []
    cases += [_chain(N) for N in (1, 2, 3, 5, 8)]
    cases += [_near_singular(e) for e in (1e-1, 1e-4, 1e-8, 0.0)]
    cases += [_mass_ratio(r) for r in (1.0, 1e6, 1e12)]
    cases += [_cartpole(r) for r in (10.0, 1.0, 1e-2, 1e-4, 1e-6)]
    cases += [_slidercrank(a, b) for a, b in
              ((3.0, 1.0), (1.5, 1e2), (1.05, 1e4), (3.0, 1e6))]
    return cases


def by_id() -> Dict[str, Case]:
    return {c.id: c for c in all_cases()}
