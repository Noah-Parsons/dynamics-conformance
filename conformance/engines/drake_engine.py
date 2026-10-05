"""
Drake, driven through a continuous-time MultibodyPlant.

Accelerations come from Drake's own dynamics terms, M(q) a = tau_g(q) - C(q,v) v,
using CalcMassMatrix, CalcBiasTerm and CalcGravityGeneralizedForces. Each
system is built from rigid bodies with point-mass inertia and ordinary joints.

Conventions handled here, and only here:
  * Drake's revolute chain reports RELATIVE joint angles; the references use
    ABSOLUTE angles from the downward vertical. theta = cumsum(q).
  * The cart-pole's revolute axis is -y so that the pole mass sits at
    x + l sin(theta), matching the reference rather than its mirror image.
"""

from __future__ import annotations

import numpy as np

NAME = "drake"
DISPLAY = "Drake"
HOMEPAGE = "https://drake.mit.edu"
PACKAGE = "drake"

ROUTES = {
    "multibody-plant": "continuous MultibodyPlant: CalcMassMatrix, CalcBiasTerm, "
                       "CalcGravityGeneralizedForces",
}
FAMILIES = {"multibody-plant": {"chain", "cartpole"}}
NOT_MODELLED = {
    "slidercrank": "Closing the loop needs a distance constraint, which Drake "
                   "enforces only inside its discrete solver; the continuous "
                   "plant cannot express it (RobotLocomotion/drake#24975).",
    "near_singular": "Needs a kinetic cross-term between two prismatic DOFs; a "
                     "rigid-body model can only induce it with a massless "
                     "intermediate body, which changes what the case tests.",
    "mass_ratio": "A spring network rather than a mechanism; any translation "
                  "would be a modelling choice rather than a transcription.",
}


def version() -> str:
    from importlib.metadata import version as v
    return v(PACKAGE)


def _point_mass(plant, name, m, com):
    from pydrake.all import RotationalInertia, SpatialInertia
    return plant.AddRigidBody(name, SpatialInertia.MakeFromCentralInertia(
        m, np.asarray(com, dtype=float), RotationalInertia(0.0, 0.0, 0.0)))


def _chain(case):
    from pydrake.all import (FixedOffsetFrame, MultibodyPlant, RevoluteJoint,
                             RigidTransform)
    ref = case.reference
    N, m, l, g = ref.N, ref.m, ref.l, ref.g
    plant = MultibodyPlant(0.0)
    com = np.array([0.0, 0.0, -l])
    bodies = [_point_mass(plant, f"m{i}", m, com) for i in range(N)]
    for i, b in enumerate(bodies):
        parent = plant.world_frame() if i == 0 else plant.AddFrame(
            FixedOffsetFrame(f"pivot{i}", bodies[i - 1].body_frame(),
                             RigidTransform(com)))
        plant.AddJoint(RevoluteJoint(f"j{i}", parent, b.body_frame(),
                                     np.array([0.0, 1.0, 0.0])))
    plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, -g])
    plant.Finalize()
    ctx = plant.CreateDefaultContext()

    def rel(x):
        x = np.asarray(x, dtype=float)
        out = x.copy()
        out[1:] -= x[:-1]
        return out

    def accel(state):
        s = np.asarray(state, dtype=float)
        plant.SetPositions(ctx, rel(s[0::2]))
        plant.SetVelocities(ctx, rel(s[1::2]))
        a_rel = np.linalg.solve(plant.CalcMassMatrix(ctx),
                                plant.CalcGravityGeneralizedForces(ctx)
                                - plant.CalcBiasTerm(ctx))
        return np.cumsum(a_rel)
    return accel


def _cartpole(case):
    from pydrake.all import MultibodyPlant, PrismaticJoint, RevoluteJoint
    cp = case.reference
    plant = MultibodyPlant(0.0)
    cart = _point_mass(plant, "cart", cp.M, np.zeros(3))
    pole = _point_mass(plant, "pole", cp.m, [0.0, 0.0, -cp.l])
    plant.AddJoint(PrismaticJoint("jx", plant.world_frame(), cart.body_frame(),
                                  np.array([1.0, 0.0, 0.0])))
    plant.AddJoint(RevoluteJoint("jth", cart.body_frame(), pole.body_frame(),
                                 np.array([0.0, -1.0, 0.0])))
    plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, -cp.g])
    plant.Finalize()
    ctx = plant.CreateDefaultContext()

    def accel(state):
        s = np.asarray(state, dtype=float)
        plant.SetPositions(ctx, s[0::2])
        plant.SetVelocities(ctx, s[1::2])
        return np.linalg.solve(plant.CalcMassMatrix(ctx),
                               plant.CalcGravityGeneralizedForces(ctx)
                               - plant.CalcBiasTerm(ctx))
    return accel


def build(route: str, case):
    if route != "multibody-plant":
        raise ValueError(f"unknown route {route!r}")
    if case.family == "chain":
        return _chain(case)
    if case.family == "cartpole":
        return _cartpole(case)
    raise ValueError(f"family {case.family!r} not modelled")
