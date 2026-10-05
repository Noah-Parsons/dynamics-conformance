"""Closed-form, library-independent references.

These modules were written for the silent-failure stress study in
MechanicsDSL's `stress_suite/` (references DOI 10.5281/zenodo.22315172) and
are copied here unchanged apart from one self-test that depended on that
suite's case generator. They use numpy only: no symbolic algebra and no code
shared with any engine under test, which is what lets them referee every engine
on equal terms.

    chain        planar N-link pendulum, and two linear 2-DOF systems
                 (near-singular coupling, extreme mass ratio)
    cartpole     prismatic cart carrying a revolute pole
    slidercrank  closed-loop slider-crank, solved as a KKT system

Each file runs its own self-tests when executed directly.
"""
