"""Build the benchmark report as a PDF (benchmark/REPORT.pdf) from results/summary.json and the training logs.
HTML + inline SVG charts, printed with headless Chrome; no plotting library needed.
    python3 benchmark/make_report.py            (host Python is enough: json + csv only, Chrome on PATH)"""
import csv
import html
import json
import pathlib
import subprocess
import sys

B = pathlib.Path(__file__).resolve().parent
ROOT = B.parent
S = json.loads((B / "results" / "summary.json").read_text())
OUT_HTML = ROOT / "runs" / "benchmark" / "report.html"
OUT_PDF = B / "REPORT.pdf"

# Colours: categorical slots in fixed order (validated: tools/dataviz validator, light mode, 6 slots, all checks pass;
# sub-3:1 slots always carry direct labels). Status colours only for outcomes.
COL = {"ppo": "#2a78d6", "dqn": "#eb6834", "rule": "#1baf7a", "tabq": "#eda100", "dwa": "#e87ba4", "goto": "#008300"}
OK, COLL, TIME = "#0ca30c", "#d03b3b", "#fab219"
INK, INK2, INK3, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"

NAMES = {"ppo": "PPO (deployed)", "ppo_final": "PPO, 8M checkpoint", "dqn_s1": "DQN seed 1", "dqn_s2": "DQN seed 2 ★",
         "dqn_s3": "DQN seed 3", "rule": "Rule controller", "tabq_s1": "Tabular Q seed 1", "tabq_s2": "Tabular Q seed 2",
         "tabq_s3": "Tabular Q seed 3 ★", "tabq_fine_explore": "Tabular Q fine (expl.)", "dwa": "DWA (privileged)", "goto": "Go-to-goal"}


def fam(n):
    return "ppo" if n.startswith("ppo") else "dqn" if n.startswith("dqn") else "tabq" if n.startswith("tabq") else n


def T(n, s="test", sh=1):
    return S["twod"][f"{n}|{s}|{sh}"]


def p(x, d=1):
    return f"{100 * x:.{d}f}"


def esc(s):
    return html.escape(str(s))


# ---------------------------------------------------------------- SVG charts
def dot_ci(rows, title_x="Success rate, %", ref=None, lo=40, hi=100, w=640, row_h=24, label_w=170, val_w=120):
    """Horizontal dot + 95% CI per row. rows: (label, value, (ci_lo, ci_hi), colour, bold)."""
    h = 30 + row_h * len(rows) + 26
    x0, x1 = label_w, w - val_w
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" class="chart" role="img">']
    for t in range(lo, hi + 1, 10):
        o.append(f'<line x1="{X(t):.1f}" y1="18" x2="{X(t):.1f}" y2="{h - 26}" stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{X(t):.1f}" y="{h - 10}" class="tick" text-anchor="middle">{t}</text>')
    o.append(f'<text x="{(x0 + x1) / 2:.1f}" y="11" class="axis" text-anchor="middle">{esc(title_x)}</text>')
    if ref is not None:
        o.append(f'<line x1="{X(ref):.1f}" y1="18" x2="{X(ref):.1f}" y2="{h - 26}" stroke="{COL["ppo"]}" stroke-width="1.5" stroke-dasharray="4 3"/>')
    for i, (lab, v, ci, c, bold) in enumerate(rows):
        y = 30 + row_h * i + row_h / 2
        wt = ' font-weight="650"' if bold else ""
        o.append(f'<text x="{label_w - 12}" y="{y + 4:.1f}" class="lab" text-anchor="end"{wt}>{esc(lab)}</text>')
        o.append(f'<line x1="{X(ci[0]):.1f}" y1="{y:.1f}" x2="{X(ci[1]):.1f}" y2="{y:.1f}" stroke="{c}" stroke-width="2.5" stroke-linecap="round" opacity="0.55"/>')
        o.append(f'<circle cx="{X(v):.1f}" cy="{y:.1f}" r="5.5" fill="{c}" stroke="{SURF}" stroke-width="2"/>')
        o.append(f'<text x="{x1 + 12}" y="{y + 4:.1f}" class="val"{wt}>{v:.1f}%  <tspan class="ci">[{ci[0]:.1f}–{ci[1]:.1f}]</tspan></text>')
    o.append("</svg>")
    return "".join(o)


def dumbbell(rows, w=640, row_h=26, label_w=170, lo=40, hi=100):
    """Safety layer on (filled) vs off (hollow). rows: (label, on, off, colour)."""
    h = 34 + row_h * len(rows) + 28
    x0, x1 = label_w, w - 110
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" class="chart" role="img">']
    for t in range(lo, hi + 1, 10):
        o.append(f'<line x1="{X(t):.1f}" y1="22" x2="{X(t):.1f}" y2="{h - 28}" stroke="{GRID}"/>')
        o.append(f'<text x="{X(t):.1f}" y="{h - 12}" class="tick" text-anchor="middle">{t}</text>')
    o.append(f'<circle cx="{x0 + 6}" cy="9" r="5" fill="{INK2}"/><text x="{x0 + 16}" y="13" class="axis">safety layer on</text>')
    o.append(f'<circle cx="{x0 + 130}" cy="9" r="4.5" fill="{SURF}" stroke="{INK2}" stroke-width="2"/><text x="{x0 + 140}" y="13" class="axis">safety layer off</text>')
    o.append(f'<text x="{x1 + 14}" y="13" class="axis">change</text>')
    for i, (lab, on, off, c) in enumerate(rows):
        y = 34 + row_h * i + row_h / 2
        o.append(f'<text x="{label_w - 12}" y="{y + 4:.1f}" class="lab" text-anchor="end">{esc(lab)}</text>')
        o.append(f'<line x1="{X(min(on, off)):.1f}" y1="{y:.1f}" x2="{X(max(on, off)):.1f}" y2="{y:.1f}" stroke="{c}" stroke-width="3" opacity="0.45"/>')
        o.append(f'<circle cx="{X(off):.1f}" cy="{y:.1f}" r="5" fill="{SURF}" stroke="{c}" stroke-width="2.5"/>')
        o.append(f'<circle cx="{X(on):.1f}" cy="{y:.1f}" r="5.5" fill="{c}" stroke="{SURF}" stroke-width="2"/>')
        d = off - on
        o.append(f'<text x="{x1 + 14}" y="{y + 4:.1f}" class="val">{d:+.1f} pts</text>')
    o.append("</svg>")
    return "".join(o)


def outcome_bars(rows, w=640, row_h=30, label_w=170):
    """Stacked outcome shares per method. rows: (label, ok, coll, timeout) as counts."""
    h = 30 + row_h * len(rows) + 8
    x0, x1 = label_w, w - 20
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" class="chart" role="img">']
    lx = x0
    for name, c in (("success", OK), ("collision", COLL), ("timeout", TIME)):
        o.append(f'<rect x="{lx}" y="3" width="11" height="11" rx="2" fill="{c}"/><text x="{lx + 16}" y="13" class="axis">{name}</text>')
        lx += 95
    for i, (lab, a, b, c) in enumerate(rows):
        n = a + b + c
        y = 28 + row_h * i
        o.append(f'<text x="{label_w - 12}" y="{y + row_h / 2 + 2:.1f}" class="lab" text-anchor="end">{esc(lab)}</text>')
        x = x0
        for k, (cnt, col) in enumerate(((a, OK), (b, COLL), (c, TIME))):
            wd = (x1 - x0) * cnt / n
            if wd <= 0:
                continue
            o.append(f'<rect x="{x:.1f}" y="{y + 3}" width="{max(wd - 2, 1):.1f}" height="{row_h - 10}" rx="3" fill="{col}"/>')
            if wd > 26:
                fill = "#ffffff" if col in (OK, COLL) else INK
                o.append(f'<text x="{x + wd / 2 - 1:.1f}" y="{y + row_h / 2 + 2:.1f}" class="bar" text-anchor="middle" fill="{fill}">{cnt}</text>')
            x += wd
    o.append("</svg>")
    return "".join(o)


def curves(series, w=320, h=230, ymin=0, ymax=100):
    """Validation success vs training steps. series: (label, [(Msteps, pct)], colour, dash, end_label)."""
    x0, x1, y0, y1 = 40, w - 78, 16, h - 34
    X = lambda v: x0 + v / 8 * (x1 - x0)
    Y = lambda v: y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" class="chart" role="img">']
    for t in range(ymin, ymax + 1, 20):
        o.append(f'<line x1="{x0}" y1="{Y(t):.1f}" x2="{x1}" y2="{Y(t):.1f}" stroke="{GRID}"/>')
        o.append(f'<text x="{x0 - 8}" y="{Y(t) + 4:.1f}" class="tick" text-anchor="end">{t}</text>')
    for t in range(0, 9, 2):
        o.append(f'<text x="{X(t):.1f}" y="{h - 16}" class="tick" text-anchor="middle">{t}M</text>')
    o.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{h - 2}" class="axis" text-anchor="middle">training steps</text>')
    o.append(f'<text x="{x0 - 30}" y="10" class="axis">success, %</text>')
    ends = []
    for lab, pts, c, dash, endlab in series:
        d = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        o.append(f'<polyline points="{d}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round"{da}/>')
        for a, b in pts:
            o.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="2.6" fill="{c}"/>')
        if endlab:
            ends.append([Y(pts[-1][1]), X(pts[-1][0]), lab, c, Y(pts[-1][1])])
    ends.sort()
    for i in range(1, len(ends)):                      # de-collide end labels
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 12)
    for yy, xx, lab, c, y_true in ends:
        o.append(f'<line x1="{xx + 4:.1f}" y1="{y_true:.1f}" x2="{x1 + 10}" y2="{yy:.1f}" stroke="{c}" stroke-width="1" opacity="0.6"/>')
        o.append(f'<text x="{x1 + 14}" y="{yy + 4:.1f}" class="lab">{esc(lab)}</text>')
    o.append("</svg>")
    return "".join(o)


def read_curve(path, ycol="val_success"):
    with open(path) as f:
        return [(int(r["steps"]) / 1e6, 100 * float(r[ycol])) for r in csv.DictReader(f)]


# ---------------------------------------------------------------- content
def table(head, rows, cls=""):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr" + (' class="hl"' if r[0].startswith("<b>") else "") + ">" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def build():
    G = S["gazebo"]
    order = ["ppo", "ppo_final", "dqn_s2", "dqn_s1", "dqn_s3", "rule", "dwa", "tabq_s3", "tabq_s1", "tabq_s2", "tabq_fine_explore", "goto"]

    # Chart 1: 2D test, shield on
    rows_on = [(NAMES[n], 100 * T(n)["success"], [100 * v for v in T(n)["success_ci"]], COL[fam(n)], n in ("ppo", "dqn_s2", "rule", "tabq_s3")) for n in order]
    c_on = dot_ci(rows_on, ref=100 * T("ppo")["success"], lo=40)
    rows_off = [(NAMES[n], 100 * T(n, sh=0)["success"], [100 * v for v in T(n, sh=0)["success_ci"]], COL[fam(n)], n in ("ppo", "dqn_s2", "rule", "tabq_s3")) for n in order]
    c_off = dot_ci(rows_off, ref=100 * T("ppo", sh=0)["success"], lo=30)
    main = ["ppo", "dqn_s2", "rule", "dwa", "tabq_s3", "goto"]
    c_db = dumbbell([(NAMES[n], 100 * T(n)["success"], 100 * T(n, sh=0)["success"], COL[fam(n)]) for n in main], lo=30)
    g_rows = [(NAMES[n], 100 * G[n]["success"], [100 * v for v in G[n]["success_ci"]], COL[fam(n)], True) for n in ("ppo", "dqn_s2", "rule")]
    c_gz = dot_ci(g_rows, ref=100 * G["ppo"]["success"], lo=60, row_h=30)
    c_gz_out = outcome_bars([(NAMES[n], G[n]["counts"]["success"], G[n]["counts"]["collision"], G[n]["counts"]["timeout"]) for n in ("ppo", "dqn_s2", "rule")])
    c_2d_out = outcome_bars([(NAMES[n], round(400 * T(n)["success"]), round(400 * T(n)["collision"]), round(400 * T(n)["timeout"])) for n in main], row_h=28)

    R = ROOT / "runs" / "benchmark"
    ppo_c = ("PPO*", read_curve_ppo(), COL["ppo"], "5 3", True)
    c_dqn = curves([ppo_c] + [(f"DQN s{s}", read_curve(R / f"dqn/final_s{s}/log.csv"), COL["dqn"], "", True) for s in (1, 2, 3)], ymin=20)
    c_tq = curves([ppo_c] + [(f"Tab. Q s{s}", read_curve(R / f"tabq/final_s{s}/log.csv"), COL["tabq"], "", True) for s in (1, 2, 3)]
                  + [("fine", read_curve(R / "tabq/explore_fine_fg_s1/log.csv"), "#b07800", "2 3", True)], ymin=20)
    c_curves = f'<div class="two"><div><h3 class="ph">DQN, 3 seeds</h3>{c_dqn}</div><div><h3 class="ph">Tabular Q, 3 seeds + fine table</h3>{c_tq}</div></div>'

    def drow(n, sh=1):
        m = T(n, sh=sh)
        d = "—" if n == "ppo" else f'{100 * m["d_success_vs_ref"]:+.1f} <span class="ci">[{100 * m["d_success_vs_ref_ci"][0]:+.1f}, {100 * m["d_success_vs_ref_ci"][1]:+.1f}]</span>'
        name = f"<b>{NAMES[n]}</b>" if n in ("ppo", "dqn_s2", "rule", "tabq_s3") else NAMES[n]
        sw = f'{p(m["collision"])} ({p(m["coll_robot"])})' if sh == 0 else p(m["collision"])
        return [name, f'{p(m["success"])} <span class="ci">[{p(m["success_ci"][0])}, {p(m["success_ci"][1])}]</span>', sw, p(m["timeout"]),
                p(m["all_four"], 0), f'{m["time_s"]:.1f}', d]
    t_on = table(["Method", "Success % [95% CI]", "Collision %", "Timeout %", "All four %", "Time to goal s", "Δ success vs PPO [95% CI]"], [drow(n) for n in order], "num")
    t_off = table(["Method", "Success % [95% CI]", "Collision % (robot)", "Timeout %", "All four %", "Time to goal s", "Δ vs PPO, both off"], [drow(n, 0) for n in order], "num")

    gz_t = table(["Method", "Success % [95% CI]", "ok / collision / timeout", "All four %", "Time to goal s", "Δ vs PPO [95% CI]"],
                 [[f"<b>{NAMES[n]}</b>", f'{p(G[n]["success"])} <span class="ci">[{p(G[n]["success_ci"][0])}, {p(G[n]["success_ci"][1])}]</span>',
                   f'{G[n]["counts"]["success"]} / {G[n]["counts"]["collision"]} / {G[n]["counts"]["timeout"]}', p(G[n]["all_four"], 0), f'{G[n]["mean_time_s"]:.1f}',
                   "—" if n == "ppo" else f'{100 * G[n]["d_success_vs_ppo"]:+.1f} <span class="ci">[{100 * G[n]["d_success_vs_ppo_ci"][0]:+.1f}, {100 * G[n]["d_success_vs_ppo_ci"][1]:+.1f}]</span>']
                  for n in ("ppo", "dqn_s2", "rule")], "num")

    Bh = S["behaviour"]
    beh_t = table(["Method", "Path efficiency", "Mean speed m/s", "Time stopped<br>(successful robots)", "Steering flips / min", "Closest obstacle m<br>(median)", "Closest robot m<br>(median)", "Safety-layer<br>overrides"],
                  [[NAMES[n], f'{Bh[n + "|1"]["path_efficiency"]:.2f}', f'{Bh[n + "|1"]["mean_speed_mps"]:.3f}', f'{p(Bh[n + "|1"]["stopped_frac_success"])}%',
                    f'{Bh[n + "|1"]["turn_flips_per_min"]:.0f}', f'{Bh[n + "|1"]["min_gap_obstacle_m_median"]:.3f}', f'{Bh[n + "|1"]["min_gap_robot_m_median"]:.3f}',
                    f'{p(T(n)["override_rate"])}%'] for n in main], "num")

    D = S["deploy"]
    dep_t = table(["Method", "Parameters", "Model size", "One robot's decision<br>(median / p99, µs)"],
                  [["PPO / DQN (same network)", "92,717", "372 KB", f'{D["ppo"]["single_us_median"]:.0f} / {D["dqn_s2"]["single_us_median"]:.0f}'],
                   ["Tabular Q (coarse / fine)", "291,600 / 7.1 M", "1.2 MB / 28 MB", f'{D["tabq_s3"]["single_us_median"]:.0f} / {D["tabq_fine_explore"]["single_us_median"]:.0f}'],
                   ["Rule controller", "7 numbers", "—", f'{D["rule"]["single_us_median"]:.0f}'],
                   ["Go-to-goal", "—", "—", f'{D["goto"]["single_us_median"]:.0f}'],
                   ["DWA (privileged)", "—", "—", "needs simulator state"]], "num")
    Cm = S["compute"]
    cost_t = table(["Work", "Where", "Wall time", "Compute", "Energy"],
                   [["Tabular Q: 13 runs (tuning, finals, fine)", "laptop, 1 thread each", f'{Cm["tabq/train"]["wall_s"] / 3600:.1f} h summed', f'{Cm["tabq/train"]["cpu_s"] / 3600:.1f} CPU h', f'~{Cm["tabq/train"]["cpu_J_est"] / 3600:.0f} Wh (est.)'],
                    ["Rule tuning: 50 parameter sets", "laptop, 2–7 processes", f'{Cm["rule/tune"]["wall_s"] / 3600:.1f} h', f'{Cm["rule/tune"]["cpu_s"] / 3600:.1f} CPU h', f'~{Cm["rule/tune"]["cpu_J_est"] / 3600:.0f} Wh (est.)'],
                    ["DQN tuning: 6 × 1M steps", "Colab T4 + 2 vCPU", f'{Cm["dqn/train"]["wall_s"] / 3600:.1f} h summed', "GPU 4–7% busy", f'{Cm["dqn/train"]["gpu_J"] / 3600:.0f} Wh GPU (measured)'],
                    ["DQN finals: 3 × 6M steps, concurrent", "Colab T4 + 2 vCPU", "1.75 h each", "~955 steps/s per run; GPU 8–15% busy", "~50 Wh GPU (est.)"],
                    ["Gazebo, 20 layouts", "laptop", f'{S["compute"]["ppo/gazebo"]["wall_s"] / 60:.0f}–{S["compute"]["rule/gazebo"]["wall_s"] / 60:.0f} min per method', "", ""],
                    ["2D evaluation, 100 layouts", "laptop", "1–5 min per condition", "", ""]])

    ppo, dqn, rul = T("ppo"), T("dqn_s2"), T("rule")
    css = CSS
    body = f"""
<section class="cover">
  <div class="kicker">qaro_Simulation · multi-robot navigation · benchmark report</div>
  <h1>Is PPO the best policy?</h1>
  <p class="sub">The deployed PPO policy against tabular Q-learning, deep Q-learning (DQN), a hand-written
  controller and two planners, on held-out layouts in a 2D simulator and in Gazebo.</p>
  <div class="tiles">
    <div class="tile"><div class="big" style="color:{COL['ppo']}">{p(ppo['success'])}%</div><div class="tl">PPO success, held-out test<br>with safety layer</div></div>
    <div class="tile"><div class="big" style="color:{COL['dqn']}">{p(dqn['success'])}%</div><div class="tl">DQN (chosen seed): statistically<br>level with PPO, 3× fewer collisions</div></div>
    <div class="tile"><div class="big" style="color:#12865c">{p(T('rule', sh=0)['success'])}%</div><div class="tl">Rule controller without safety layer<br>vs PPO {p(T('ppo', sh=0)['success'])}% without it</div></div>
  </div>
  <div class="verdict">
    <h3>Verdict</h3>
    <p><b>With the safety layer (as deployed), PPO is matched, not beaten:</b> DQN ties it in 2D (90.0% vs 90.2%) and in Gazebo
    (73 of 80 robots each), with fewer collisions but slower arrivals. <b>Without the safety layer, PPO is weak:</b> it learned
    to lean on that layer, and a classic dynamic-window planner using the same sensors beats it by 16.5 points in 2D. That planner does
    not hold up in Gazebo (80%), where both learned policies stay at 91%. Tabular Q-learning trails far behind (69.5%).</p>
  </div>
  <h3 style="margin-top:8mm">At a glance: what happened to the 400 test robots (2D, safety layer on)</h3>
  {c_2d_out}
  <div class="meta">Branch <code>benchmark-policies</code>, tag <code>benchmark-v1</code> · 10 November 2026 ·
  laptop (i5-12450H) + Google Colab (Tesla T4) · every number regenerated from raw per-robot outcomes</div>
</section>

<section>
  <h2><span class="n">1</span>Key findings</h2>
  <ol class="findings">
    <li><b>PPO and DQN are tied in the deployed configuration.</b> Test success 90.2% vs 90.0% (difference −0.2 points,
    95% interval −4.0 to +3.5). Collisions 4.5% vs 1.5%. DQN is slower to the goal (16.1 s vs 10.9 s) and times out more
    (8.5% vs 5.2%). DQN trained for only 75% of PPO's budget and with far less tuning.</li>
    <li><b>DQN's seeds disagree</b>: 88.0%, 90.0%, 83.0%. PPO exists as a single trained policy, so its own seed spread is unknown;
    its 90.2% may be a favourable draw just as DQN's best seed is.</li>
    <li><b>PPO depends on the safety layer.</b> The layer replaces 26% of PPO's actions; without it PPO falls to 68.0% and a quarter of
    its robots hit another robot. DQN falls to 76.0%. The rule controller does not change (84.0% → 84.5%).</li>
    <li><b>The hand-written controller wins in 2D without the safety layer</b> (+16.5 points over PPO, interval +9.5 to +23.2),
    with 0.5% collisions, but <b>drops to 80.0% in Gazebo</b> (−11.2 points vs PPO), mainly from timeouts.
    Learned policies transferred better to realistic physics and sensing.</li>
    <li><b>Tabular Q-learning is limited by its state compression</b>: 61–70% across seeds, ~20% collisions. A 24× finer
    table learned more slowly and ended worse (54.2%).</li>
    <li><b>Planners that see the true map are not enough either</b>: DWA reaches 79.8% with almost no collisions, but 20% of its
    robots time out.</li>
  </ol>
  <h3>Success on the held-out test layouts, safety layer on and off</h3>
  <p class="cap">2D simulator, layouts 100–199 (400 robot runs per method). Filled dot: safety layer on (deployed); hollow: off.</p>
  {c_db}
</section>

<section class="pb">
  <h2><span class="n">2</span>What was compared, and how</h2>
  <div class="two">
  <div>
  <h3>The task</h3>
  <p>Four differential-drive robots each drive to their own goal in a random arena of obstacles, avoiding obstacles, walls and
  each other. Every method controls each robot from the <b>same 59 numbers</b> (36 lidar sectors, goal distance and direction,
  own speed, camera estimates of the three nearest robots) and picks one of the <b>same 45 commands</b> (speed × turn rate),
  ten times a second. A rule-based <b>safety layer</b> can veto a command that leads into another robot; it is part of the
  deployed system.</p>
  <h3>Data split</h3>
  <p><b>Test: layouts 100–199</b>, never used to train, tune or pick anything. (PPO's checkpoint was picked on layouts 0–99,
  so those are shown only for reference.) New methods trained on layouts 1000–2899 and were tuned and checkpoint-selected on
  <b>validation layouts 2900–2999</b> only.</p>
  <h3>Statistics</h3>
  <p>Success, collision and timeout per robot. 95% intervals by bootstrap over layouts (10,000 resamples), because the four
  robots of one layout are correlated; differences to PPO paired on the same layouts.</p>
  </div>
  <div>
  <h3>Contenders</h3>
  {table(["Method", "What it is", "Training here"], [
      [f'<span class="sw" style="background:{COL["ppo"]}"></span>PPO', "Deployed policy (existing), MLP 59-256-256-45", "none; 10 earlier runs"],
      [f'<span class="sw" style="background:{COL["dqn"]}"></span>DQN', "Deep Q-learning, same network shape", "6 configs × 1M; 3 seeds × 6M"],
      [f'<span class="sw" style="background:{COL["rule"]}"></span>Rule', "Dynamic-window search: predicts each command's arc, vetoes collisions, scores progress", "50 parameter sets"],
      [f'<span class="sw" style="background:{COL["tabq"]}"></span>Tabular Q', "Q-table over 6,480 discretised states", "9 configs; 3 seeds × 8M"],
      [f'<span class="sw" style="background:{COL["dwa"]}"></span>DWA', "Planner that sees the <b>true map</b> (privileged)", "none"],
      [f'<span class="sw" style="background:{COL["goto"]}"></span>Go-to-goal', "Drives straight at the goal", "none"]], "small")}
  <div class="note"><b>Written before any result.</b> The protocol (contenders, split, budgets, selection rule
  “success − 0.5 × collision” on validation) was committed first. Later amendments, each recorded with its reason, are in section 8.</div>
  </div></div>
</section>

<section class="pb">
  <h2><span class="n">3</span>2D results with the safety layer (deployed configuration)</h2>
  <p class="cap">Success rate with 95% interval; dashed line: PPO. ★ = seed chosen on validation (never on test).</p>
  {c_on}
  {t_on}
  <p class="foot">Override = share of robot-steps where the safety layer replaced the method's command: PPO {p(ppo['override_rate'])}%, DQN {p(dqn['override_rate'])}%,
  rule {p(rul['override_rate'])}%. Selecting PPO's checkpoint on layouts 0–99 made little difference: the never-selected 8M checkpoint scores 89.5%.</p>
</section>

<section class="pb">
  <h2><span class="n">4</span>Without the safety layer</h2>
  <p class="cap">Same layouts and methods, safety layer removed. Dashed line: PPO without the layer.</p>
  {c_off}
  {t_off}
  <div class="note"><b>Why the order flips.</b> PPO and DQN were trained with the safety layer always on, so they learned to let it
  resolve robot-robot conflicts: without it, most of their collisions are with other robots (PPO 25.2 of 29.2 points,
  DQN 15.8 of 16.5). The rule controller does its own
  collision check every step, so removing the external layer changes nothing for it.</div>
</section>

<section class="pb">
  <h2><span class="n">5</span>Gazebo confirmation</h2>
  <p>Full pipeline: overhead camera with AprilTags, 360-beam lidar, odometry and physics; layouts 100–119 (80 robots), safety
  layer on. PPO, the validation-chosen DQN seed and the better of rule / tabular Q from the 2D test.</p>
  <p class="cap">Success rate with 95% interval (bootstrap over the 20 layouts)</p>
  {c_gz}
  <p class="cap">Outcome of each of the 80 robots</p>
  {c_gz_out}
  {gz_t}
  <p class="foot">PPO's 91.2% repeats its earlier Gazebo record (89.8% on layouts 1–50), a check that the benchmark harness drives
  Gazebo correctly. With 20 layouts the intervals are wide: only the rule controller's drop is close to resolved
  (upper bound 0.0). Gazebo ends an episode after 70 s (2D: 90 s), which turns slow robots into timeouts.</p>
</section>

<section class="pb">
  <h2><span class="n">6</span>How they learn and how they drive</h2>
  <p class="cap">Validation success (layouts 2900–2999) during training, checkpoints every 1M steps. PPO's curve is from its
  original training log (*measured on layouts 0–99), so it is indicative only. DQN stopped at 6M (Colab reset).
  DQN reaches PPO's level by 4–5M steps but swings between checkpoints; tabular Q flattens at about 60–70%.</p>
  {c_curves}
  <h3>Driving style (2D test, safety layer on)</h3>
  {beh_t}
  <p class="foot">Path efficiency = straight-line distance / driven path for successful robots; it can exceed 1 because a robot stops
  within 0.12 m of its goal point.</p>
  <ul class="tight">
    <li>PPO drives fastest and passes closest to other robots, leaving conflicts to the safety layer. DQN keeps more distance, drives slower and stops more.</li>
    <li>Both learned policies switch steering direction 200–250 times a minute (jittery commands); the planners 4–12 times.</li>
    <li>The rule controller skims obstacles (tuned margin 2 cm; 117 near-miss steps per robot vs PPO's 1.2): harmless with exact 2D geometry, a likely reason it lost ground in Gazebo.</li>
  </ul>
</section>

<section class="pb">
  <h2><span class="n">7</span>Cost: deployment, compute and energy</h2>
  {dep_t}
  <p class="foot">All are far below the 100 ms control period. In Gazebo (four robots per call) the median decision took 98 µs (PPO), 84 µs (DQN), 1.2 ms (rule).</p>
  <h3>Training and evaluation compute</h3>
  {cost_t}
  <p class="foot">CPU energy is an <b>estimate</b> (CPU seconds × 3.75 W per busy thread): Intel's RAPL energy counters are root-only
  since CVE-2020-8694 and were deliberately left closed. GPU energy was measured (NVML). The T4 was mostly idle: a 93k-parameter
  network is tiny for it, and Colab's two CPU cores (simulator, replay buffer) were the bottleneck. PPO's original 8M-step
  training took about 2 h on the laptop; its energy was not measured.</p>
</section>

<section class="pb">
  <h2><span class="n">8</span>Integrity: deviations, bugs, limitations</h2>
  <div class="two"><div>
  <h3>Protocol amendments</h3>
  <ol class="tight">
    <li><b>Finer tabular Q</b> tried after the coarse table plateaued at ~60% with 99% of states visited; it lost on validation, so the
    coarse table stands. One exploratory 8M run is reported separately.</li>
    <li>Rule grid <b>extended past its edge</b> within the pre-set 50-set budget, decided before any rule test result. The final choice
    still sits on the edge of two parameters.</li>
    <li><b>DQN trained 6M of 8M steps</b>: the Colab runtime was reset and the replay buffers were lost; the 6M checkpoints were kept by
    the owner's decision. This can only disadvantage DQN; two of three seeds were still improving.</li>
  </ol>
  <h3>Bugs found and fixed before any reported number</h3>
  <ul class="tight">
    <li>Tabular Q reused the terminal state as the next state after a reset (0.6% of updates): first tuning sweep discarded.</li>
    <li>The energy meter double-counted guest CPU time.</li>
    <li>The unattended pipeline had a path and a shell-strictness bug: 70 minutes lost, no result affected.</li>
  </ul>
  </div><div>
  <h3>Findings about the original project</h3>
  <ul class="tight">
    <li>PPO's checkpoint was selected on layouts 0–99, so its 91.5% there is slightly optimistic (90.2% on clean layouts).</li>
    <li>The training layouts predate the goal-visibility rule: 385 of 2000 differ from today's generator. All learners here trained on
    exactly PPO's layouts, so the comparison is fair.</li>
  </ul>
  <h3>Limitations</h3>
  <ul class="tight">
    <li><b>Tuning effort favours PPO</b> (10 runs plus reward and environment fixes).</li>
    <li>PPO is one trained policy; no seed spread is available.</li>
    <li>Gazebo used 20 layouts: intervals of ±6–9 points.</li>
    <li>The 2D simulator's motion model is exact, which flatters planners that predict motion.</li>
    <li>CPU energy is estimated, not measured.</li>
  </ul>
  </div></div>
  <div class="note">Everything is reproducible from the repository: protocol <code>benchmark/PROTOCOL.md</code>, lab notebook
  <code>benchmark/LOGBOOK.md</code>, tables <code>benchmark/RESULTS.md</code>; <code>benchmark/make_results.py</code> recomputes every
  statistic from the raw per-robot files and <code>benchmark/make_report.py</code> rebuilds this PDF.</div>
</section>
"""
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Benchmark report</title><style>{css}</style></head><body>{body}</body></html>'


def read_curve_ppo():
    with open(ROOT / "docs" / "run10_eval.csv") as f:
        return [(int(r[0]) / 1e6, 100 * float(r[1])) for r in csv.reader(f) if r and r[0][0].isdigit()]


CSS = f"""
@page {{ size: A4; margin: 16mm 17mm 18mm 17mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "Noto Sans", "DejaVu Sans", sans-serif; color: {INK}; background: #ffffff; font-size: 9.6pt; line-height: 1.45; margin: 0; }}
h1 {{ font-size: 34pt; line-height: 1.05; margin: 6mm 0 4mm; letter-spacing: -0.02em; }}
h2 {{ font-size: 16pt; margin: 0 0 3mm; letter-spacing: -0.01em; display: flex; align-items: center; gap: 3mm; }}
h2 .n {{ display: inline-flex; width: 8mm; height: 8mm; border-radius: 50%; background: {INK}; color: #fff; font-size: 10.5pt;
         align-items: center; justify-content: center; }}
h3 {{ font-size: 10.8pt; margin: 4mm 0 1.5mm; }}
p {{ margin: 0 0 2.2mm; }}
code {{ font-family: "DejaVu Sans Mono", monospace; font-size: 8.4pt; background: #f0efec; padding: 0 1mm; border-radius: 1mm; }}
section {{ margin-bottom: 6mm; }}
.pb {{ break-before: page; }}
.cover {{ min-height: 255mm; display: flex; flex-direction: column; }}
.kicker {{ text-transform: uppercase; letter-spacing: 0.12em; font-size: 8pt; color: {INK2}; margin-top: 18mm; border-top: 3px solid {INK}; padding-top: 3mm; }}
.sub {{ font-size: 12.5pt; color: {INK2}; max-width: 150mm; line-height: 1.4; }}
.tiles {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 5mm; margin: 12mm 0 9mm; }}
.tile {{ border: 1px solid {GRID}; border-radius: 3mm; padding: 5mm 5mm 4mm; background: {SURF}; }}
.big {{ font-size: 30pt; font-weight: 700; letter-spacing: -0.02em; line-height: 1; }}
.tl {{ font-size: 8.6pt; color: {INK2}; margin-top: 2.5mm; }}
.verdict {{ background: #f4f3f0; border-left: 4px solid {INK}; padding: 4mm 6mm; border-radius: 0 3mm 3mm 0; }}
.verdict h3 {{ margin-top: 0; }}
.verdict p {{ font-size: 10.6pt; line-height: 1.5; }}
.meta {{ margin-top: auto; font-size: 8pt; color: {INK3}; border-top: 1px solid {GRID}; padding-top: 3mm; }}
.findings li {{ margin-bottom: 2mm; }}
.findings {{ padding-left: 5mm; }}
.cap {{ font-size: 8.6pt; color: {INK2}; margin: 2mm 0 1mm; }}
.foot {{ font-size: 8.2pt; color: {INK2}; margin-top: 2mm; }}
.note {{ background: #f4f3f0; border-radius: 2mm; padding: 3mm 4mm; font-size: 8.8pt; margin-top: 3mm; }}
.two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 7mm; }}
.chart {{ display: block; margin: 1mm 0 3mm; }}
.chart text {{ font-family: "Noto Sans", sans-serif; fill: {INK}; }}
.chart .tick {{ font-size: 9px; fill: {INK3}; }}
.chart .axis {{ font-size: 9.5px; fill: {INK2}; }}
.chart .lab {{ font-size: 10.5px; }}
.chart .val {{ font-size: 10.5px; font-variant-numeric: tabular-nums; }}
.chart .ci, .ci {{ fill: {INK3}; color: {INK3}; font-size: 0.86em; }}
.chart .bar {{ font-size: 10px; font-weight: 600; }}
table {{ width: 100%; border-collapse: collapse; margin: 2mm 0; font-size: 8.4pt; }}
th {{ text-align: left; font-weight: 600; color: {INK2}; border-bottom: 1.5px solid {INK}; padding: 1.4mm 1.6mm; vertical-align: bottom; font-size: 7.9pt; }}
td {{ border-bottom: 1px solid {GRID}; padding: 1.2mm 1.6mm; vertical-align: top; }}
table.num td:not(:first-child), table.num th:not(:first-child) {{ text-align: right; font-variant-numeric: tabular-nums; }}
tr.hl td {{ background: #f7f6f3; }}
table.small {{ font-size: 8pt; }}
table.small td:first-child {{ white-space: nowrap; }}
h3.ph {{ font-size: 9.5pt; margin: 2mm 0 0; color: {INK2}; }}
.sw {{ display: inline-block; width: 2.6mm; height: 2.6mm; border-radius: 0.6mm; margin-right: 1.6mm; vertical-align: -0.2mm; }}
ul.tight, ol.tight {{ padding-left: 5mm; margin: 1mm 0 2mm; }}
ul.tight li, ol.tight li {{ margin-bottom: 1.2mm; }}
"""

if __name__ == "__main__":
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build())
    chrome = "google-chrome"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={OUT_PDF}",
                    "--virtual-time-budget=5000", OUT_HTML.as_uri()], check=True, capture_output=True)
    print("wrote", OUT_PDF, OUT_PDF.stat().st_size, "bytes", file=sys.stderr)
