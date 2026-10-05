"""
Run the whole catalogue against the installed version of one engine.

    python -m conformance.run --engine sympy
    python -m conformance.run --engine chrono --key 10.0.0-b1354 --spec "..."

Writes results/<engine>/<key>.json. Each (route, case) runs in its own
process with a wall-clock limit; see worker.py for how a case is scored.

If the engine could not be installed, pass --install-failed: a stub result is
written so the release is shown as uninstallable rather than retried forever.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
import time

from . import cases as C
from .engines import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERDICTS = ("pass", "silent", "warned", "error", "timeout", "harness-error")


def safe_key(s: str) -> str:
    s = re.sub(r" \(build (\d+)\)$", r"-b\1", s)     # "10.0.0 (build 1354)" -> "10.0.0-b1354"
    return re.sub(r"[^A-Za-z0-9._+-]", "_", s)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("GITHUB_SHA", "unknown")


def _run_one(engine: str, route: str, case_id: str, timeout: float) -> dict:
    cmd = [sys.executable, "-m", "conformance.worker", "--engine", engine,
           "--route", route, "--case", case_id]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"route": route, "case": case_id, "verdict": "timeout",
                "seconds": round(time.time() - t0, 1), "timeout_s": timeout}
    for line in p.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    # No result line: the process died (segfault, abort, out of memory).
    # That is the engine crashing loudly, which counts as an error.
    return {"route": route, "case": case_id, "verdict": "error",
            "build_error": f"worker exited with code {p.returncode} and no result",
            "stderr": p.stderr[-1500:], "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--key", help="result file key (default: installed version)")
    ap.add_argument("--spec", default="", help="install spec, recorded verbatim")
    ap.add_argument("--timeout", type=float, default=180.0, help="seconds per case")
    ap.add_argument("--only", default="", help="regex filter on case ids")
    ap.add_argument("--install-failed", action="store_true")
    ap.add_argument("--version", help="version label when --install-failed")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()

    eng = load(a.engine)
    meta = {
        "engine": eng.NAME, "display": eng.DISPLAY, "homepage": eng.HOMEPAGE,
        "install_spec": a.spec,
        "catalogue_version": C.CATALOGUE_VERSION,
        "harness_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "run_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "run_url": (f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
                    f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
                    if os.environ.get("GITHUB_RUN_ID") else None),
        "routes": eng.ROUTES,
        "known_issues": getattr(eng, "KNOWN_ISSUES", {}),
        "not_modelled": eng.NOT_MODELLED,
    }

    if a.install_failed:
        version = a.version or a.key or "unknown"
        key = safe_key(a.key or version)
        meta.update(version=version, key=key, status="install-failed",
                    results=[], summary={})
        _write(a.out, eng.NAME, key, meta)
        return 0

    version = eng.version()
    key = safe_key(a.key or version)
    meta.update(version=version, key=key, status="ran")
    print(f"{eng.DISPLAY} {version}  ->  results/{eng.NAME}/{key}.json")

    pat = re.compile(a.only) if a.only else None
    results = []
    for route in eng.ROUTES:
        for case in C.all_cases():
            if case.family not in eng.FAMILIES[route]:
                continue
            if pat and not pat.search(case.id):
                continue
            r = _run_one(eng.NAME, route, case.id, a.timeout)
            results.append(r)
            we = r.get("worst_err")
            print(f"  {route:<16} {case.id:<44} {r['verdict']:<8}"
                  f"{'' if we is None else f'{we:10.2e}'}"
                  f"  {r.get('seconds', 0):7.1f}s", flush=True)

    meta["results"] = results
    meta["summary"] = {v: sum(1 for r in results if r["verdict"] == v)
                       for v in VERDICTS}
    _write(a.out, eng.NAME, key, meta)
    print("summary:", meta["summary"])
    return 0


def _write(out_dir, engine, key, meta):
    d = os.path.join(out_dir, engine)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, sort_keys=False)
        f.write("\n")
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
