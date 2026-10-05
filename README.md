# Dynamics engine conformance

**Live results: https://noah-parsons.github.io/dynamics-conformance/**

Every day, this repository checks whether any of four multibody dynamics
engines has published a new release. Each new release is installed on a clean
GitHub Actions runner and asked for the accelerations of the same set of
mechanical systems. Every answer is compared with a closed-form reference. The
results are committed here and published as a web page.

The question each run answers is: **when an engine gets the answer wrong, does
it tell you?**

| Engine | Releases tracked from | Routes scored |
|---|---|---|
| [SymPy](https://www.sympy.org) (`sympy.physics.mechanics`) | PyPI | `rhs()` (documented symbolic path); lambdified mass matrix |
| [MechanicsDSL](https://github.com/MechanicsDSL/mechanicsdsl) | PyPI (`mechanicsdsl-core`) | `compile_dsl` → simulator right-hand side |
| [Drake](https://drake.mit.edu) | PyPI | continuous `MultibodyPlant` dynamics terms |
| [Project Chrono](https://projectchrono.org) | anaconda.org (`projectchrono::pychrono`) | `DoAssembly` + `GetPosDt2`; Chrono's system matrices |

## What gets checked

21 cases in five families (see [`conformance/cases.py`](conformance/cases.py)):

- **Pendulum chains**, 1 to 8 links, probed at random, fully extended,
  inverted and folded configurations.
- **Near-singular coupling**: a mass matrix whose determinant goes to zero,
  ending with an exactly singular one that has no answer at all. The only
  passing response to that one is a refusal.
- **Extreme mass ratio**: two masses differing by up to 10¹².
- **Cart-pole**: cart-to-pole mass ratio down to 10⁻⁶, where the mass matrix
  collapses twice per swing.
- **Slider-crank**: a closed loop, probed through dead centre, with the slider
  up to 10⁶ times heavier than the crank.

Each case is probed at 8 to 18 states. The score is the study's metric,
max |a − r| / max(|r|, 1), against a tolerance of 10⁻⁸. The tolerance is
widened only where the system is so ill-conditioned that no solver can do
better, the reference included.

## Verdicts

| Verdict | Meaning |
|---|---|
| **pass** | Agrees with the reference at every probe, or refuses a system that has no answer. |
| **silent** | A wrong answer, with no error and no warning. This is the outcome that matters. |
| **warned** | A wrong answer, but the engine emitted a warning. |
| **error** | The engine refused or crashed. That is acceptable behaviour, since you know something is wrong. |
| **timeout** | No answer within 180 s. This is a missing answer, not a wrong one. |

## The references

[`conformance/references/`](conformance/references/) holds closed-form
references written in numpy alone, with no symbolic algebra. They share no code
with any engine under test, so they can referee all four on equal terms.
Each file runs its own self-tests (`python -m conformance.references.chain`).
They were built for a study of silent failures in dynamics engines; the study's
artefacts are at [doi:10.5281/zenodo.22315172](https://doi.org/10.5281/zenodo.22315172).

Adapters (`conformance/engines/`) translate conventions such as state ordering,
relative versus absolute joint angles, and axis direction. They never correct
an engine's answer.

## Running it yourself

```bash
pip install numpy scipy packaging sympy
python -m conformance.run --engine sympy --out my-results
python -m conformance.site        # builds _site/ from results/
pytest -q
```

To score a specific release in CI, run the `conformance` workflow by hand with
`engine` and `version` set, for example `chrono` and `10.0.0-b1187`.

## Adding an engine

Write one module in `conformance/engines/` with the interface described in
[`conformance/engines/__init__.py`](conformance/engines/__init__.py), register
it there, and teach `conformance/releases.py` where its releases are published.
Pull requests from engine maintainers are especially welcome. You know the
idiomatic way to drive your engine better than anyone.

## Licence

MIT. Results are released under CC BY 4.0.
