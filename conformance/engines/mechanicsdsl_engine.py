"""
MechanicsDSL, driven through PhysicsCompiler.compile_dsl.

The accelerations are read from the simulator's ODE right-hand side
(`simulator.equations_of_motion`), which is what MechanicsDSL actually
integrates, rather than from the symbolic equations it reports. The two can
differ, and the right-hand side is the one that produces a user's trajectory.
"""

from __future__ import annotations

import numpy as np

NAME = "mechanicsdsl"
DISPLAY = "MechanicsDSL"
HOMEPAGE = "https://github.com/MechanicsDSL/mechanicsdsl"
PACKAGE = "mechanicsdsl-core"

ROUTES = {
    "compile_dsl": "PhysicsCompiler.compile_dsl, then simulator.equations_of_motion",
}
FAMILIES = {"compile_dsl": {"chain", "near_singular", "mass_ratio",
                            "cartpole", "slidercrank"}}
NOT_MODELLED: dict = {}


def version() -> str:
    from importlib.metadata import version as v
    return v(PACKAGE)


def _dsl(case):
    """(DSL source, coordinate names, uses constraints)."""
    p, fam = case.params, case.family
    if fam == "chain":
        N = int(p["N"])
        ref = case.reference
        lines = [r"\system{chain%d}" % N]
        lines += [r"\defvar{theta%d}{Angle}{rad}" % i for i in range(N)]
        lines += [r"\parameter{m}{%r}{kg}" % ref.m, r"\parameter{l}{%r}{m}" % ref.l,
                  r"\parameter{g}{%r}{m/s^2}" % ref.g]
        ke = [r"%r*m*l^2*\cos{theta%d - theta%d}*\dot{theta%d}*\dot{theta%d}"
              % (0.5 * (N - max(j, k)), j, k, j, k)
              for j in range(N) for k in range(N)]
        pe = [r"%r*m*g*l*(1 - \cos{theta%d})" % (float(N - j), j) for j in range(N)]
        lines.append(r"\lagrangian{(%s) - (%s)}" % (" + ".join(ke), " + ".join(pe)))
        ic = []
        for i in range(N):
            ic += [f"theta{i}={0.3 if i == 0 else 0.15}", f"theta{i}_dot=0.0"]
        lines.append(r"\initial{%s}" % ", ".join(ic))
        return "\n".join(lines), [f"theta{i}" for i in range(N)], False

    if fam == "near_singular":
        c = 1.0 - float(p["eps"])
        return "\n".join([
            r"\system{nearsing}",
            r"\defvar{x}{Position}{m}", r"\defvar{y}{Position}{m}",
            r"\parameter{m}{1.0}{kg}", r"\parameter{k}{1.0}{N/m}",
            r"\lagrangian{0.5*m*\dot{x}^2 + 0.5*m*\dot{y}^2 + %r*m*\dot{x}*\dot{y} "
            r"- 0.5*k*x^2 - 0.5*k*y^2}" % c,
            r"\initial{x=1.0, y=0.0, x_dot=0.0, y_dot=0.0}"]), ["x", "y"], False

    if fam == "mass_ratio":
        return "\n".join([
            r"\system{massratio}",
            r"\defvar{x}{Position}{m}", r"\defvar{y}{Position}{m}",
            r"\parameter{m1}{1.0}{kg}", r"\parameter{m2}{%r}{kg}" % float(p["ratio"]),
            r"\parameter{k}{1.0}{N/m}", r"\parameter{k1}{1.0}{N/m}",
            r"\lagrangian{0.5*m1*\dot{x}^2 + 0.5*m2*\dot{y}^2 "
            r"- 0.5*k*(x - y)^2 - 0.5*k1*x^2}",
            r"\initial{x=1.0, y=0.0, x_dot=0.0, y_dot=0.0}"]), ["x", "y"], False

    if fam == "cartpole":
        cp = case.reference
        return "\n".join([
            r"\system{cartpole}",
            r"\defvar{x}{Position}{m}", r"\defvar{theta}{Angle}{rad}",
            r"\parameter{Mc}{%r}{kg}" % cp.M, r"\parameter{mp}{%r}{kg}" % cp.m,
            r"\parameter{l}{%r}{m}" % cp.l, r"\parameter{g}{%r}{m/s^2}" % cp.g,
            r"\lagrangian{0.5*(Mc + mp)*\dot{x}^2 "
            r"+ mp*l*\dot{x}*\dot{theta}*\cos{theta} "
            r"+ 0.5*mp*l^2*\dot{theta}^2 + mp*g*l*\cos{theta}}",
            r"\initial{x=0.0, x_dot=0.0, theta=1.0, theta_dot=0.0}"]), ["x", "theta"], False

    if fam == "slidercrank":
        sc = case.reference
        y0 = sc.initial_state()
        return "\n".join([
            r"\system{slidercrank}",
            r"\defvar{theta}{Angle}{rad}", r"\defvar{x}{Position}{m}",
            r"\parameter{mc}{%r}{kg}" % sc.m_c, r"\parameter{ms}{%r}{kg}" % sc.m_s,
            r"\parameter{r}{%r}{m}" % sc.r, r"\parameter{l}{%r}{m}" % sc.l,
            r"\parameter{g}{%r}{m/s^2}" % sc.g,
            r"\lagrangian{0.5*mc*r^2*\dot{theta}^2 + 0.5*ms*\dot{x}^2 "
            r"- mc*g*r*\sin{theta}}",
            r"\constraint{x^2 - 2*r*x*\cos{theta} + r^2 - l^2}",
            r"\initial{theta=%r, theta_dot=%r, x=%r, x_dot=%r}"
            % tuple(float(v) for v in y0)]), ["theta", "x"], True

    raise ValueError(f"family {fam!r} not modelled")


ENGINE_WARNINGS: list = []


def build(route: str, case):
    if route != "compile_dsl":
        raise ValueError(f"unknown route {route!r}")
    from mechanics_dsl import PhysicsCompiler

    src, coords, constrained = _dsl(case)
    c = PhysicsCompiler()
    res = c.compile_dsl(src, use_hamiltonian=False, use_constraints=constrained)
    # Warnings the engine itself returns count as the engine speaking up.
    ENGINE_WARNINGS.extend(str(w) for w in (res.get("warnings") or []))
    if not res.get("success"):
        raise RuntimeError(f"compile_dsl success=False: {res.get('error', '')}")

    eom = c.simulator.equations_of_motion
    n = len(coords)

    def accel(state):
        dydt = np.asarray(eom(0.0, np.asarray(state, dtype=float)), dtype=float)
        return dydt[1:2 * n:2]
    return accel
