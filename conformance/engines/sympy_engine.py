"""
sympy.physics.mechanics, driven through LagrangesMethod.

Two routes, both documented by SymPy:

  rhs            form_lagranges_equations() then rhs(). rhs() performs a
                 SYMBOLIC LU solve of the full mass matrix. This is the path
                 the documentation shows, so it is the path most users take.
  mass-matrix    keep mass_matrix_full and forcing_full symbolic, lambdify
                 them, and solve numerically at each evaluation. What a user
                 who knows where the symbolic solve stops scaling would write.

Every system is written natively as a Lagrangian (plus a holonomic constraint
for the slider-crank). Nothing here is translated from another engine's input.
"""

from __future__ import annotations

import numpy as np

NAME = "sympy"
DISPLAY = "SymPy (sympy.physics.mechanics)"
HOMEPAGE = "https://www.sympy.org"
PACKAGE = "sympy"

ROUTES = {
    "rhs": "LagrangesMethod.rhs(): symbolic LU solve, the documented path",
    "mass-matrix": "lambdified mass_matrix_full / forcing_full, solved numerically",
}
_ALL = {"chain", "near_singular", "mass_ratio", "cartpole", "slidercrank"}
FAMILIES = {"rhs": _ALL, "mass-matrix": _ALL}
NOT_MODELLED: dict = {}


def version() -> str:
    import sympy
    return sympy.__version__


def _system(case):
    """(Lagrangian, coordinates, holonomic constraints) for a case."""
    import sympy as sp
    from sympy.physics.mechanics import dynamicsymbols

    t = dynamicsymbols._t
    half = sp.Rational(1, 2)
    p = case.params
    fam = case.family

    if fam == "chain":
        N = int(p["N"])
        ref = case.reference
        m, l, g = ref.m, ref.l, ref.g
        q = [dynamicsymbols(f"theta{i}") for i in range(N)]
        w = [qi.diff(t) for qi in q]
        T = sp.S.Zero
        for j in range(N):
            for k in range(N):
                coeff = N - max(j, k)
                T += half * coeff * m * l ** 2 * sp.cos(q[j] - q[k]) * w[j] * w[k]
        V = sum((N - j) * m * g * l * (1 - sp.cos(q[j])) for j in range(N))
        return T - V, q, []

    if fam == "near_singular":
        x, y = dynamicsymbols("x y")
        xd, yd = x.diff(t), y.diff(t)
        c = 1.0 - float(p["eps"])
        L = (half * xd ** 2 + half * yd ** 2 + c * xd * yd
             - half * x ** 2 - half * y ** 2)
        return L, [x, y], []

    if fam == "mass_ratio":
        x, y = dynamicsymbols("x y")
        xd, yd = x.diff(t), y.diff(t)
        L = (half * xd ** 2 + half * float(p["ratio"]) * yd ** 2
             - half * (x - y) ** 2 - half * x ** 2)
        return L, [x, y], []

    if fam == "cartpole":
        cp = case.reference
        x, th = dynamicsymbols("x th")
        xd, thd = x.diff(t), th.diff(t)
        L = (half * (cp.M + cp.m) * xd ** 2
             + cp.m * cp.l * xd * thd * sp.cos(th)
             + half * cp.m * cp.l ** 2 * thd ** 2
             + cp.m * cp.g * cp.l * sp.cos(th))
        return L, [x, th], []

    if fam == "slidercrank":
        sc = case.reference
        th, x = dynamicsymbols("th x")
        thd, xd = th.diff(t), x.diff(t)
        L = (half * sc.m_c * sc.r ** 2 * thd ** 2 + half * sc.m_s * xd ** 2
             - sc.m_c * sc.g * sc.r * sp.sin(th))
        con = x ** 2 - 2 * sc.r * x * sp.cos(th) + sc.r ** 2 - sc.l ** 2
        return L, [th, x], [con]

    raise ValueError(f"family {fam!r} not modelled")


def build(route: str, case):
    import sympy as sp
    from sympy.physics.mechanics import LagrangesMethod, dynamicsymbols

    L, q, cons = _system(case)
    t = dynamicsymbols._t
    u = [qi.diff(t) for qi in q]
    n = len(q)
    lm = LagrangesMethod(L, q, hol_coneqs=cons or None)
    lm.form_lagranges_equations()

    def split(state):
        s = np.asarray(state, dtype=float)
        return list(s[0::2]), list(s[1::2])

    if route == "rhs":
        f = sp.lambdify([q, u], lm.rhs(), "numpy")

        def accel(state):
            qv, uv = split(state)
            out = np.asarray(f(qv, uv), dtype=float).reshape(-1)
            # rhs() returns [q', u', multipliers...]; accelerations are block 2.
            return out[n:2 * n]
        return accel

    if route == "mass-matrix":
        M, F = lm.mass_matrix_full, lm.forcing_full
        Mf = sp.lambdify([q, u], M, "numpy")
        Ff = sp.lambdify([q, u], F, "numpy")
        dim = M.shape[0]

        def accel(state):
            qv, uv = split(state)
            Mn = np.array(Mf(qv, uv), dtype=float).reshape(dim, dim)
            Fn = np.array(Ff(qv, uv), dtype=float).reshape(dim)
            return np.linalg.solve(Mn, Fn)[n:2 * n]
        return accel

    raise ValueError(f"unknown route {route!r}")
