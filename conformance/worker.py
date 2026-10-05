"""
Score one (engine, route, case) and print the result as one JSON line.

Run in its own process by `run.py`, so a hang becomes a timeout and a hard
crash (a segfault in compiled engine code, say) becomes a recorded failure
instead of taking the whole run down.

VERDICTS
--------
  pass     every probe agrees with the reference within tolerance; or, for a
           genuinely degenerate case, the engine refused it
  silent   at least one probe came back wrong (or non-finite) and the engine
           raised nothing and warned of nothing. The one that matters.
  warned   at least one probe came back wrong, but the engine emitted a
           warning while building or evaluating
  error    the engine raised (refused or crashed) and no probe was wrong
  timeout  set by run.py when the worker does not finish in time

A wrong answer outranks a refusal: a case where some probes raise and others
return wrong numbers is scored on the wrong numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
import warnings

import numpy as np

from . import cases as C
from .engines import load

_IGNORED = (DeprecationWarning, PendingDeprecationWarning, FutureWarning,
            ImportWarning, ResourceWarning)


class _LogCatcher(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        try:
            self.records.append(f"{record.name}: {record.getMessage()}"[:300])
        except Exception:
            pass


def score(engine_name: str, route: str, case_id: str) -> dict:
    case = C.by_id()[case_id]
    eng = load(engine_name)
    out = {"route": route, "case": case_id, "family": case.family}

    catcher = _LogCatcher()
    logging.getLogger().addHandler(catcher)
    logging.getLogger().setLevel(logging.WARNING)
    t0 = time.time()

    probes = []
    build_error = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fn = eng.build(route, case)
        except Exception as e:
            fn = None
            build_error = f"{type(e).__name__}: {e}"[:500]
        build_s = time.time() - t0

        if fn is not None:
            for k, state in enumerate(case.states):
                p = {"k": k}
                try:
                    res = fn(np.array(state, dtype=float))
                    if isinstance(res, tuple):
                        a, used = res
                        used = np.asarray(used, dtype=float)
                        p["shift"] = float(np.max(np.abs(used - state)))
                    else:
                        a, used = res, np.asarray(state, dtype=float)
                    a = np.asarray(a, dtype=float).reshape(-1)
                except Exception as e:
                    p["outcome"] = "raised"
                    p["error"] = f"{type(e).__name__}: {e}"[:300]
                    probes.append(p)
                    continue

                n = len(state) // 2
                if a.shape != (n,):
                    p["outcome"] = "wrong"
                    p["error"] = f"returned shape {a.shape}, expected ({n},)"
                elif not np.all(np.isfinite(a)):
                    p["outcome"] = "wrong"
                    p["error"] = "non-finite accelerations returned"
                elif case.expects_refusal:
                    p["outcome"] = "wrong"
                    p["error"] = "answered a system with no defined answer"
                else:
                    r = case.reference.accel(used)
                    err = C.relerr(a, r)
                    tol = case.tolerance(used)
                    p.update(err=err, tol=tol,
                             outcome="ok" if err <= tol else "wrong")
                    if p["outcome"] == "wrong":
                        p["engine"] = a.tolist()
                        p["reference"] = np.asarray(r).tolist()
                        p["state"] = used.tolist()
                probes.append(p)

    logging.getLogger().removeHandler(catcher)
    engine_warnings = [f"{w.category.__name__}: {w.message}"[:300]
                       for w in caught if not issubclass(w.category, _IGNORED)]
    engine_warnings += catcher.records
    engine_warnings += [f"engine: {w}"[:300]
                        for w in getattr(eng, "ENGINE_WARNINGS", [])]
    # De-duplicate, keep order, cap.
    engine_warnings = list(dict.fromkeys(engine_warnings))[:20]

    outcomes = [p["outcome"] for p in probes]
    if build_error is not None:
        verdict = "pass" if case.expects_refusal else "error"
    elif "wrong" in outcomes:
        verdict = "warned" if engine_warnings else "silent"
    elif "raised" in outcomes:
        verdict = "pass" if case.expects_refusal and all(
            o == "raised" for o in outcomes) else "error"
    else:
        verdict = "pass"

    errs = [p["err"] for p in probes if "err" in p]
    ratios = [p["err"] / p["tol"] for p in probes if "err" in p]
    worst = max(probes, key=lambda p: p.get("err", -1.0), default=None)
    out.update(
        verdict=verdict,
        refused=(build_error is not None or (bool(outcomes) and
                                             all(o == "raised" for o in outcomes))),
        worst_err=max(errs) if errs else None,
        worst_ratio=max(ratios) if ratios else None,
        n_probes=len(case.states),
        n_ok=outcomes.count("ok"), n_wrong=outcomes.count("wrong"),
        n_raised=outcomes.count("raised"),
        build_seconds=round(build_s, 3),
        seconds=round(time.time() - t0, 3),
        build_error=build_error,
        warnings=engine_warnings,
        max_shift=max((p.get("shift", 0.0) for p in probes), default=0.0),
    )
    first_err = next((p["error"] for p in probes if p.get("error")), None)
    if first_err:
        out["probe_error"] = first_err
    wrong = [p for p in probes if p["outcome"] == "wrong" and "engine" in p]
    if wrong:
        w = max(wrong, key=lambda p: p["err"])
        out["worst_probe"] = {k: w[k] for k in ("k", "state", "engine",
                                                "reference", "err", "tol")}
    elif worst is not None and "err" in worst:
        out["worst_probe"] = {"k": worst["k"], "err": worst["err"],
                              "tol": worst["tol"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--route", required=True)
    ap.add_argument("--case", required=True)
    a = ap.parse_args()
    try:
        res = score(a.engine, a.route, a.case)
    except Exception as e:                       # harness failure, not engine
        res = {"route": a.route, "case": a.case, "verdict": "harness-error",
               "build_error": f"{type(e).__name__}: {e}",
               "traceback": traceback.format_exc()[-2000:]}
    print("@@RESULT@@" + json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
