"""
Build the public conformance page from results/.

    python -m conformance.site            # writes _site/

Output:
  _site/index.html                the page
  _site/results/<engine>/*.json   every raw result, as committed
  _site/badges/<engine>.json      shields.io endpoint badge for the newest release
  _site/latest.json               newest result per engine, for scripts
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
from collections import OrderedDict

from packaging.version import InvalidVersion, Version

from . import cases as C
from .engines import ENGINES, load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_URL = "https://github.com/Noah-Parsons/dynamics-conformance"
DATASET_DOI = "10.5281/zenodo.22315172"

ORDER = ["pass", "warned", "error", "timeout", "silent", "harness-error"]
LABEL = {"pass": "pass", "silent": "silent", "warned": "warned", "error": "error",
         "timeout": "timeout", "harness-error": "harness", "n/m": "not modelled"}


def _vkey(res):
    v = res.get("version", "0")
    m = re.match(r"([^ ]+)(?: \(build (\d+)\))?", v)
    try:
        pv = Version(m.group(1))
    except (InvalidVersion, AttributeError):
        pv = Version("0")
    return (pv, int(m.group(2) or 0) if m else 0, res.get("run_at", ""))


def load_results():
    out = OrderedDict()
    base = os.path.join(ROOT, "results")
    for eng in ENGINES:
        d = os.path.join(base, eng)
        rows = []
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith(".json"):
                    with open(os.path.join(d, fn), encoding="utf-8") as f:
                        r = json.load(f)
                    r["_file"] = f"results/{eng}/{fn}"
                    rows.append(r)
        rows.sort(key=_vkey)
        out[eng] = rows
    return out


def cell_map(res):
    return {(r["route"], r["case"]): r for r in res.get("results", [])}


def changes(rows):
    """Verdict changes between consecutive releases that both ran."""
    ran = [r for r in rows if r.get("status") == "ran"]
    out = []
    for prev, cur in zip(ran, ran[1:]):
        if prev.get("catalogue_version") != cur.get("catalogue_version"):
            continue
        a, b = cell_map(prev), cell_map(cur)
        for k in b:
            if k in a and a[k]["verdict"] != b[k]["verdict"]:
                out.append((prev, cur, k, a[k]["verdict"], b[k]["verdict"]))
    return out


def _fmt(x):
    return "" if x is None else f"{x:.1e}"


def _cell_title(r):
    bits = [f"{LABEL.get(r['verdict'], r['verdict'])}"]
    if r.get("worst_err") is not None:
        bits.append(f"worst relative error {r['worst_err']:.2e}")
    if r.get("n_wrong"):
        bits.append(f"{r['n_wrong']} of {r.get('n_probes')} probes wrong")
    if r.get("refused"):
        bits.append("refused")
    if r.get("build_error"):
        bits.append(r["build_error"][:200])
    elif r.get("probe_error"):
        bits.append(r["probe_error"][:200])
    if r.get("warnings"):
        bits.append("warned: " + r["warnings"][0][:160])
    if r.get("verdict") == "timeout":
        bits.append(f"no answer within {r.get('timeout_s', '?')} s")
    return " | ".join(bits)


def esc(s):
    return html.escape(str(s), quote=True)


CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e4e1d8;--card:#fff;
--pass:#2f7d4f;--pass-bg:#e3f1e8;--silent:#b3261e;--silent-bg:#fbe3e1;
--warned:#9a6700;--warned-bg:#fbf0d4;--error:#3f5f9e;--error-bg:#e3eaf7;
--timeout:#6b6a65;--timeout-bg:#ecebe6;--nm:#a9a7a0;--accent:#1f4e79}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe6;--muted:#a3a19a;
--line:#2c2b29;--card:#1c1c1a;--pass:#7cc79a;--pass-bg:#1d3326;--silent:#ff8a80;
--silent-bg:#3d1c1a;--warned:#e8c063;--warned-bg:#3a2f12;--error:#9db8ec;
--error-bg:#1c2639;--timeout:#a3a19a;--timeout-bg:#2a2a27;--nm:#5f5e59;--accent:#8cb8e6}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:19px;margin:40px 0 10px}
p{max-width:760px}
a{color:var(--accent)}
.lede{color:var(--muted);font-size:16px;margin-top:0}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card h3{margin:0;font-size:16px}.card .v{color:var(--muted);font-size:13px;margin:2px 0 10px}
.counts{display:flex;flex-wrap:wrap;gap:6px}
.pill{display:inline-block;border-radius:999px;padding:1px 9px;font-size:12.5px;font-weight:600;white-space:nowrap}
.pass{color:var(--pass);background:var(--pass-bg)}.silent{color:var(--silent);background:var(--silent-bg)}
.warned{color:var(--warned);background:var(--warned-bg)}.error{color:var(--error);background:var(--error-bg)}
.timeout,.harness-error{color:var(--timeout);background:var(--timeout-bg)}
.nm{color:var(--nm)}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;background:var(--card);position:sticky;top:0}
td.c{text-align:center;white-space:nowrap}
tr.fam td{background:var(--bg);font-weight:600;color:var(--muted);font-size:12.5px;
text-transform:uppercase;letter-spacing:.04em}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px}
.small{font-size:13px;color:var(--muted)}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:8px 0 0;font-size:13.5px}
.legend span.pill{margin-right:5px}
footer{margin-top:48px;color:var(--muted);font-size:13px}
"""


def render(results):
    cases = C.all_cases()
    latest = {e: next((r for r in reversed(rows) if r.get("status") == "ran"), None)
              for e, rows in results.items()}
    newest = max((r.get("run_at", "") for rows in results.values() for r in rows),
                 default="")

    H = []
    w = H.append
    w("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    w("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    w("<title>Dynamics Engine Conformance</title>")
    w("<meta name='description' content='Every release of SymPy, MechanicsDSL, "
      "Drake and Project Chrono, checked automatically against closed-form "
      "multibody references.'>")
    w(f"<style>{CSS}</style></head><body><main>")
    w("<h1>Dynamics engine conformance</h1>")
    w("<p class='lede'>Each new release of each engine is installed automatically "
      "and asked for the accelerations of the same mechanical systems. Every "
      "answer is checked against a closed-form reference that shares no code "
      "with any engine. The question is the one that matters to a user: "
      "<b>when an engine is wrong, does it tell you?</b></p>")
    w(f"<p class='small'>Last run {esc(newest[:10]) if newest else 'never'} · "
      f"<a href='{REPO_URL}'>source and raw results</a> · "
      f"<a href='latest.json'>latest.json</a></p>")

    w("<div class='legend'>"
      "<div><span class='pill pass'>pass</span>agrees with the reference, or "
      "correctly refuses a system with no answer</div>"
      "<div><span class='pill silent'>silent</span>wrong answer, no error, no warning</div>"
      "<div><span class='pill warned'>warned</span>wrong answer, with a warning</div>"
      "<div><span class='pill error'>error</span>refused or crashed</div>"
      "<div><span class='pill timeout'>timeout</span>no answer in time</div>"
      "</div>")

    # -- cards ---------------------------------------------------------------
    w("<h2>Newest release of each engine</h2><div class='cards'>")
    for e, rows in results.items():
        mod = load(e)
        r = latest[e]
        w(f"<div class='card'><h3><a href='{esc(mod.HOMEPAGE)}'>{esc(mod.DISPLAY)}</a></h3>")
        if r is None:
            w("<div class='v'>no results yet</div></div>")
            continue
        run = (f" · <a href='{esc(r['run_url'])}'>run</a>" if r.get("run_url") else "")
        w(f"<div class='v'>{esc(r['version'])} · {esc(r['run_at'][:10])}{run}</div>"
          "<div class='counts'>")
        for v in ORDER:
            n = r["summary"].get(v, 0)
            if n:
                w(f"<span class='pill {v}'>{n} {LABEL[v]}</span>")
        w("</div></div>")
    w("</div>")

    # -- changes -------------------------------------------------------------
    ch = [c for rows in results.values() for c in changes(rows)]
    w("<h2>What changed between releases</h2>")
    if not ch:
        w("<p class='small'>No verdict has changed between consecutive releases "
          "yet. When one does, it is listed here.</p>")
    else:
        w("<div class='scroll'><table><tr><th>Engine</th><th>From → to</th>"
          "<th>Route</th><th>Case</th><th>Verdict</th></tr>")
        for prev, cur, (route, cid), a, b in reversed(ch):
            w(f"<tr><td>{esc(cur['display'])}</td>"
              f"<td class='mono'>{esc(prev['version'])} → {esc(cur['version'])}</td>"
              f"<td class='mono'>{esc(route)}</td><td class='mono'>{esc(cid)}</td>"
              f"<td><span class='pill {a}'>{LABEL.get(a, a)}</span> → "
              f"<span class='pill {b}'>{LABEL.get(b, b)}</span></td></tr>")
        w("</table></div>")

    # -- matrix --------------------------------------------------------------
    cols = []
    for e in results:
        mod = load(e)
        for route in mod.ROUTES:
            cols.append((e, mod, route))
    w("<h2>Every case, newest release</h2>")
    w("<p class='small'>Hover a cell for the worst relative error and any message "
      "the engine gave. Each case is probed at 8 to 18 states, including the "
      "configurations that make it degenerate.</p>")
    w("<div class='scroll'><table><tr><th>Case</th>")
    for e, mod, route in cols:
        r = latest[e]
        ver = esc(r["version"]) if r else "–"
        issue = mod.__dict__.get("KNOWN_ISSUES", {}).get(route)
        iss = f" <a href='{esc(issue)}' title='known issue'>⚑</a>" if issue else ""
        w(f"<th title='{esc(mod.ROUTES[route])}'>{esc(mod.DISPLAY.split(' (')[0])}"
          f"<br><span class='small mono'>{esc(route)} · {ver}</span>{iss}</th>")
    w("</tr>")
    fam = None
    for case in cases:
        if case.family != fam:
            fam = case.family
            w(f"<tr class='fam'><td colspan='{len(cols) + 1}'>{esc(fam.replace('_', ' '))}</td></tr>")
        w(f"<tr><td><span class='mono'>{esc(case.id.split('/', 1)[1])}</span>"
          f"<br><span class='small'>{esc(case.description)}</span></td>")
        for e, mod, route in cols:
            r = latest[e]
            if case.family not in mod.FAMILIES[route]:
                why = mod.NOT_MODELLED.get(case.family, "not modelled")
                w(f"<td class='c nm' title='{esc(why)}'>–</td>")
                continue
            cell = cell_map(r).get((route, case.id)) if r else None
            if cell is None:
                w("<td class='c nm'>·</td>")
                continue
            v = cell["verdict"]
            w(f"<td class='c' title='{esc(_cell_title(cell))}'>"
              f"<span class='pill {v}'>{LABEL.get(v, v)}</span>"
              f"<br><span class='small mono'>{_fmt(cell.get('worst_err'))}</span></td>")
        w("</tr>")
    w("</table></div>")

    # -- history -------------------------------------------------------------
    w("<h2>History</h2><div class='scroll'><table><tr><th>Engine</th><th>Release</th>"
      "<th>Run</th><th>Verdicts</th><th>Data</th></tr>")
    for e, rows in results.items():
        for r in reversed(rows):
            if r.get("status") == "install-failed":
                counts = "<span class='small'>could not be installed on the runner</span>"
            else:
                counts = " ".join(f"<span class='pill {v}'>{r['summary'].get(v, 0)} {LABEL[v]}</span>"
                                  for v in ORDER if r["summary"].get(v))
            run = (f"<a href='{esc(r['run_url'])}'>{esc(r['run_at'][:10])}</a>"
                   if r.get("run_url") else esc(r.get("run_at", "")[:10]))
            w(f"<tr><td>{esc(r['display'])}</td><td class='mono'>{esc(r['version'])}</td>"
              f"<td>{run}</td><td>{counts}</td>"
              f"<td><a class='mono' href='{esc(r['_file'])}'>json</a></td></tr>")
    w("</table></div>")

    # -- method --------------------------------------------------------------
    w("<h2>How it works</h2>")
    w("<p>Every day a scheduled job asks PyPI and anaconda.org for each engine's "
      "releases. Any release without a result is installed on a fresh GitHub "
      "Actions runner (Ubuntu, Python 3.12) and scored on the whole catalogue. "
      "Results are committed to the repository, so the history is permanent and "
      "anyone can audit it.</p>")
    w("<p>At every probe state the engine's generalised accelerations are compared "
      "with the reference using max<sub>i</sub> |a<sub>i</sub> − r<sub>i</sub>| / "
      "max(|r<sub>i</sub>|, 1). The tolerance is 10<sup>−8</sup>, widened only "
      "where the system is so ill-conditioned that no solver, the reference "
      "included, can do better: to 10<sup>−14</sup> times the condition number of "
      "the diagonally scaled mass matrix. Systems that have no defined answer, "
      "such as an exactly singular mass matrix, pass only if the engine refuses "
      "them.</p>")
    w("<p>The references are written in numpy alone, with no symbolic algebra, so "
      "they share no code with any engine under test. They come from a study of "
      "silent failures in dynamics engines (artefacts: "
      f"<a href='https://doi.org/{DATASET_DOI}'>doi:{DATASET_DOI}</a>). "
      "Adapters translate conventions such as relative versus absolute joint "
      "angles and nothing else; an adapter never corrects an engine's answer.</p>")
    w("<p>Where an engine offers two documented ways to get accelerations, both "
      "are scored as separate columns. A dash means that engine cannot model the "
      "system without changing what the case tests; hover it for the reason.</p>")
    w(f"<footer>Built by Noah Parsons. Badges: "
      f"<span class='mono'>https://img.shields.io/endpoint?url=&lt;this site&gt;/badges/&lt;engine&gt;.json</span>"
      f"</footer></main></body></html>")
    return "\n".join(H)


def badge(r):
    if r is None:
        return {"schemaVersion": 1, "label": "conformance", "message": "no data",
                "color": "lightgrey"}
    s = r["summary"]
    total = sum(s.values())
    if s.get("silent"):
        color, msg = "red", f"{s['silent']} silent of {total}"
    elif s.get("warned") or s.get("error") or s.get("timeout"):
        color, msg = "yellow", f"{s.get('pass', 0)}/{total} pass"
    else:
        color, msg = "brightgreen", f"{s.get('pass', 0)}/{total} pass"
    return {"schemaVersion": 1, "label": f"conformance {r['version']}",
            "message": msg, "color": color}


def main() -> int:
    results = load_results()
    out = os.path.join(ROOT, "_site")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "badges"))
    if os.path.isdir(os.path.join(ROOT, "results")):
        shutil.copytree(os.path.join(ROOT, "results"), os.path.join(out, "results"))
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(render(results))
    latest = {}
    for e, rows in results.items():
        r = next((x for x in reversed(rows) if x.get("status") == "ran"), None)
        with open(os.path.join(out, "badges", f"{e}.json"), "w") as f:
            json.dump(badge(r), f)
        if r:
            latest[e] = {k: v for k, v in r.items() if k != "_file"}
    with open(os.path.join(out, "latest.json"), "w") as f:
        json.dump(latest, f, indent=1)
    open(os.path.join(out, ".nojekyll"), "w").close()
    print("wrote _site/ for", {e: len(r) for e, r in results.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
