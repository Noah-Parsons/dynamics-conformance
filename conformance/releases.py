"""
Find engine releases that have no published result yet.

    python -m conformance.releases                 # newest release of each engine
    python -m conformance.releases --backfill 3    # newest three of each
    python -m conformance.releases --engine chrono --version 10.0.0-b1187

Prints a GitHub Actions matrix: {"include": [{engine, version, key, installer,
spec}, ...]}. A release counts as done once results/<engine>/<key>.json
exists, whether it ran or failed to install, so nothing is retried forever.

Sources:
  PyPI JSON API        sympy, mechanicsdsl-core, drake (final releases only,
                       yanked ones skipped, and only those with a wheel or
                       sdist this runner can install)
  anaconda.org API     pychrono on the projectchrono channel. Chrono ships
                       rebuilds of one version number under new build
                       numbers, and those rebuilds carry real code changes, so
                       each (version, build) is its own release here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

from packaging.version import InvalidVersion, Version

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "3.12"
PY_TAG = "cp312"

PYPI = {"sympy": "sympy", "mechanicsdsl": "mechanicsdsl-core", "drake": "drake"}


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "dynamics-conformance"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _pypi_installable(files) -> bool:
    for f in files:
        if f.get("yanked"):
            continue
        name = f["filename"]
        if f["packagetype"] == "sdist":
            return True
        if name.endswith("-none-any.whl"):
            return True
        if "manylinux" in name and "x86_64" in name and (
                f"-{PY_TAG}-" in name or "-abi3-" in name):
            return True
    return False


def pypi_releases(engine: str):
    pkg = PYPI[engine]
    data = _get(f"https://pypi.org/pypi/{pkg}/json")
    out = []
    for v, files in data["releases"].items():
        try:
            pv = Version(v)
        except InvalidVersion:
            continue
        if pv.is_prerelease or pv.is_devrelease or not files:
            continue
        if not _pypi_installable(files):
            continue
        out.append((pv, {"engine": engine, "version": v, "key": v,
                         "installer": "pip", "spec": f"{pkg}=={v}"}))
    out.sort(key=lambda t: t[0])
    return [r for _, r in out]


def chrono_releases():
    data = _get("https://api.anaconda.org/package/projectchrono/pychrono")
    best = {}
    for f in data["files"]:
        a = f.get("attrs", {})
        build = a.get("build", "")
        if a.get("subdir") != "linux-64" or not build.startswith("py312"):
            continue
        try:
            pv = Version(f["version"])
        except InvalidVersion:
            continue
        bn = int(a.get("build_number", re.sub(r".*_", "", build) or 0))
        best[(pv, bn)] = (f["version"], build, bn)
    out = []
    for (pv, bn) in sorted(best):
        v, build, bn = best[(pv, bn)]
        out.append({"engine": "chrono", "version": f"{v} (build {bn})",
                    "key": f"{v}-b{bn}", "installer": "conda",
                    "spec": f"projectchrono::pychrono={v}={build}"})
    return out


def releases(engine: str):
    return chrono_releases() if engine == "chrono" else pypi_releases(engine)


def done(engine: str, key: str) -> bool:
    return os.path.exists(os.path.join(ROOT, "results", engine, f"{key}.json"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="all")
    ap.add_argument("--version", default="", help="a specific key to run")
    ap.add_argument("--backfill", type=int, default=1,
                    help="consider the newest N releases of each engine")
    ap.add_argument("--force", action="store_true", help="rerun even if done")
    a = ap.parse_args()

    engines = ["sympy", "mechanicsdsl", "drake", "chrono"] \
        if a.engine in ("", "all") else [a.engine]
    todo = []
    for e in engines:
        try:
            rel = releases(e)
        except Exception as ex:
            print(f"warning: could not list {e} releases: {ex}", file=sys.stderr)
            continue
        if a.version:
            rel = [r for r in rel if r["key"] == a.version or r["version"] == a.version]
        else:
            rel = rel[-max(a.backfill, 1):]
        for r in rel:
            if a.force or not done(e, r["key"]):
                todo.append(r)

    print(json.dumps({"include": todo}))
    print(f"{len(todo)} release(s) to run: "
          + ", ".join(f"{r['engine']} {r['version']}" for r in todo),
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
