from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

APPROVAL_MIN = 55
FINAL_GRADE_COL = "Nota Final"
DATA_FILE = Path("Data") / "Database v2026.01.xlsx"
OUTPUT_FILE = Path("static_dashboard.html")

NON_EVAL_COLS = {
    "Semester",
    "Semestre",
    "Nombres",
    "Nombre",
    "Rut",
    "RUT",
    "Activo",
    "Paralelo",
    "Nota Final",
    "Promedio",
    "Ap.Paterno",
    "Ap.Materno",
    "Correo",
}

EXCLUDED_EVAL_COLS = {
    # "Asistencia",
}

EVAL_ORDER = [
    "C1",
    "C2",
    "C3",
    "C4",
    "EJ1",
    "EJ2",
    "EJ3",
    "CL1",
    "CL2",
    "Proyecto",
]


def normalize_col_name(col_name: str) -> str:
    return str(col_name).strip().casefold()


def sort_evals(eval_list) -> list[str]:
    order_map = {col: i for i, col in enumerate(EVAL_ORDER)}
    return sorted(eval_list, key=lambda c: (order_map.get(c, len(EVAL_ORDER)), c))


def histogram_percent_bins(values: np.ndarray, bin_start=0, bin_end=100, bin_size=10):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[(values >= bin_start) & (values <= bin_end)]
    if values.size == 0:
        return np.array([]), np.array([]), np.array([])

    edges = np.arange(bin_start, bin_end + bin_size, bin_size, dtype=float)
    counts, _ = np.histogram(values, bins=edges)
    percents = (counts / values.size) * 100.0
    centers = edges[:-1] + (bin_size / 2.0)
    ranges = np.column_stack([edges[:-1], edges[1:]])
    return centers, percents, ranges


def kde_percent_curve(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros_like(x_grid)

    sample_std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    bandwidth = 1.06 * sample_std * (values.size ** (-1.0 / 5.0)) if sample_std > 0 else 0.0
    bandwidth = max(bandwidth, 2.0)

    normalized_diff = (x_grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * normalized_diff ** 2).sum(axis=1)
    density /= (values.size * bandwidth * np.sqrt(2.0 * np.pi))
    return density * 100.0


def load_semester_data() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    data_file = DATA_FILE
    if not data_file.exists():
        available = sorted(Path("Data").glob("*.xlsx"))
        if not available:
            raise FileNotFoundError("No xlsx files found in Data folder")
        data_file = available[0]

    workbook = pd.read_excel(data_file, sheet_name=None, header=0)

    non_eval_norm = {normalize_col_name(col) for col in NON_EVAL_COLS}
    excluded_norm = {normalize_col_name(col) for col in EXCLUDED_EVAL_COLS}

    frames = []
    semester_eval_map: dict[str, list[str]] = {}
    for sheet_name, temp_df in workbook.items():
        temp_df = temp_df.copy()
        temp_df["Semester"] = str(sheet_name)

        eval_cols_sheet = [
            col
            for col in temp_df.columns
            if normalize_col_name(col) not in non_eval_norm
            and normalize_col_name(col) not in excluded_norm
            and pd.api.types.is_numeric_dtype(temp_df[col])
        ]

        if not eval_cols_sheet:
            continue

        semester_eval_map[str(sheet_name)] = sort_evals(eval_cols_sheet)
        frames.append(temp_df)

    if not frames:
        raise ValueError("No sheets with valid evaluation columns")

    final_df = pd.concat(frames, ignore_index=True)
    final_df["Semester"] = final_df["Semester"].astype(str)
    return final_df, semester_eval_map


def round_or_none(value, digits=2):
    if value is None:
        return None
    if pd.isna(value):
        return None
    return round(float(value), digits)


def build_payload(df: pd.DataFrame, semester_eval_map: dict[str, list[str]]) -> dict:
    semesters = sorted(df["Semester"].dropna().unique().tolist())

    payload = {
        "meta": {
            "title": "MIN102 y MIN20125 - Dashboard de Rendimiento Academico",
            "approvalMin": APPROVAL_MIN,
            "finalGradeCol": FINAL_GRADE_COL,
        },
        "semesters": semesters,
        "single": {"bySemester": {}},
      "comparison": {"approvalAll": [], "kdeGrid": [], "kdeByEval": {}},
    }

    for semester in semesters:
        sem_df = df[df["Semester"] == semester].copy()
        evals = semester_eval_map.get(semester, [])

        avg_by_eval = {}
        hist_by_eval = {}
        stats_by_eval = {}

        for eval_col in evals:
            if eval_col not in sem_df.columns:
                continue

            s = sem_df[eval_col].dropna()
            if s.empty:
                continue

            avg_by_eval[eval_col] = round_or_none(s.mean())

            centers, percents, ranges = histogram_percent_bins(s.values)
            hist_by_eval[eval_col] = {
                "centers": [round(float(x), 2) for x in centers.tolist()],
                "percents": [round(float(y), 4) for y in percents.tolist()],
                "ranges": [[round(float(a), 2), round(float(b), 2)] for a, b in ranges.tolist()],
            }

            stats_by_eval[eval_col] = {
                "mean": round_or_none(s.mean()),
                "median": round_or_none(s.median()),
                "min": round_or_none(s.min()),
                "max": round_or_none(s.max()),
                "std": round_or_none(s.std()),
                "n": int(s.shape[0]),
            }

        final_avg = None
        approval = None
        if FINAL_GRADE_COL in sem_df.columns:
            final_s = sem_df[FINAL_GRADE_COL].dropna()
            if not final_s.empty:
                final_avg = round_or_none(final_s.mean())
                approved = int((final_s >= APPROVAL_MIN).sum())
                failed = int((final_s < APPROVAL_MIN).sum())
                rate = round((approved / len(final_s)) * 100.0, 2)
                approval = {
                    "approved": approved,
                    "failed": failed,
                    "rate": rate,
                }
                payload["comparison"]["approvalAll"].append({
                    "semester": semester,
                    "rate": rate,
                })

        payload["single"]["bySemester"][semester] = {
            "evals": evals,
            "avgByEval": avg_by_eval,
            "finalAvg": final_avg,
            "approval": approval,
            "histByEval": hist_by_eval,
            "statsByEval": stats_by_eval,
        }

    # Precompute KDE curves server-side so static chart matches Python dashboard behavior.
    x_grid = np.linspace(0, 100, 401)
    payload["comparison"]["kdeGrid"] = [round(float(x), 4) for x in x_grid.tolist()]
    all_evals = sort_evals({col for cols in semester_eval_map.values() for col in cols})

    for eval_col in all_evals:
        curves_by_semester = {}
        for semester in semesters:
            if eval_col not in semester_eval_map.get(semester, []):
                continue
            sem_df = df[df["Semester"] == semester]
            if eval_col not in sem_df.columns:
                continue

            sem_values = sem_df[eval_col].dropna().astype(float).values
            sem_values = sem_values[(sem_values >= 0) & (sem_values <= 100)]
            if sem_values.size == 0:
                continue

            y_curve = kde_percent_curve(sem_values, x_grid)
            curves_by_semester[semester] = [round(float(y), 6) for y in y_curve.tolist()]

        if curves_by_semester:
            payload["comparison"]["kdeByEval"][eval_col] = curves_by_semester

    return payload


def html_template(payload_json: str) -> str:
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>Static Dashboard</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    :root {{
      --primary: #0f766e;
      --secondary: #14b8a6;
      --accent: #f59e0b;
      --bg: #f2f7f8;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --border: #d7e3e9;
    }}
    body {{
      margin: 0;
      font-family: \"Segoe UI\", \"Calibri\", \"Trebuchet MS\", Verdana, sans-serif;
      background: radial-gradient(circle at top right, rgba(20,184,166,0.10), transparent 35%), var(--bg);
      color: var(--text);
    }}
    .header {{
      background: linear-gradient(135deg, #ffffff 0%, #eaf6f5 100%);
      border-bottom: 3px solid var(--primary);
      padding: 24px 20px 20px;
      text-align: center;
    }}
    .title {{
      margin: 0;
      font-family: \"Trebuchet MS\", \"Segoe UI\", \"Calibri\", sans-serif;
      font-size: 40px;
      font-weight: 800;
      letter-spacing: -0.7px;
      line-height: 1.15;
    }}
    .wrap {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 18px;
    }}
    .tabs {{
      display: flex;
      gap: 6px;
      margin: 0 0 14px 0;
      border-bottom: 1px solid var(--border);
    }}
    .tab-btn {{
      border: 1px solid var(--border);
      border-bottom: none;
      background: #e9f2f3;
      color: var(--muted);
      border-radius: 10px 10px 0 0;
      font-size: 15px;
      font-weight: 700;
      padding: 10px 18px;
      cursor: pointer;
    }}
    .tab-btn.active {{
      background: var(--card);
      color: var(--text);
      border-top: 3px solid var(--primary);
      padding-top: 8px;
    }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      box-shadow: 0 10px 24px rgba(15,23,42,0.08);
      padding: 18px;
      margin-bottom: 16px;
    }}
    .row {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      align-items: stretch;
      margin-bottom: 16px;
    }}
    .col {{
      flex: 1 1 480px;
      min-width: 320px;
    }}
    .hist-row {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      align-items: flex-start;
    }}
    .hist-plot {{ flex: 2 1 620px; min-width: 360px; }}
    .hist-table {{ flex: 1 1 320px; min-width: 280px; overflow-x: auto; padding-bottom: 4px; }}
    .semester-checklist {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 8px 12px;
      margin-top: 6px;
    }}
    .semester-checklist label {{ margin: 0; font-weight: 600; }}
    .mini-actions {{
      display: flex;
      gap: 8px;
      margin: 6px 0 4px;
      flex-wrap: wrap;
    }}
    .mini-btn {{
      border: 1px solid var(--border);
      background: #ffffff;
      color: var(--text);
      border-radius: 7px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
    }}
    .mini-btn:hover {{ background: #f3f7f8; }}
    @media (max-width: 1120px) {{
      .col {{ min-width: 100%; }}
      .hist-plot, .hist-table {{ min-width: 100%; }}
    }}
    h3 {{
      margin: 0 0 14px 0;
      font-size: 20px;
      font-weight: 700;
      border-bottom: 2px solid var(--primary);
      padding-bottom: 8px;
    }}
    label {{
      display: block;
      margin-bottom: 8px;
      font-size: 15px;
      font-weight: 600;
    }}
    select {{
      width: 100%;
      padding: 9px 10px;
      border: 1px solid var(--border);
      border-radius: 8px;
      font-size: 14px;
      background: #fff;
      color: var(--text);
    }}
    .checklist {{ display: flex; flex-wrap: wrap; gap: 10px 14px; font-size: 14px; }}
    .checklist label {{ margin: 0; font-weight: 500; }}
    .stats-title {{ margin: 0 0 10px 0; font-size: 18px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; table-layout: fixed; }}
    th {{
      background: var(--primary);
      color: #fff;
      padding: 9px 10px;
      font-size: 13px;
      font-weight: 700;
      text-align: center;
    }}
    td {{
      border: 1px solid var(--border);
      padding: 8px 10px;
      font-size: 13px;
      text-align: center;
      white-space: normal;
      word-break: break-word;
    }}
    tbody tr:nth-child(even) td {{ background: #edf4f7; }}
  </style>
</head>
<body>
  <div class=\"header\">
    <h1 class=\"title\" id=\"page-title\"></h1>
  </div>
  <div class=\"wrap\">
    <div class=\"tabs\">
      <button class=\"tab-btn active\" data-tab=\"single-tab\">Semestre Individual</button>
      <button class=\"tab-btn\" data-tab=\"comp-tab\">Comparacion de Semestres</button>
    </div>

    <section id=\"single-tab\" class=\"tab-panel active\">
      <div class=\"card\">
        <label for=\"single-semester\">Seleccionar Semestre:</label>
        <select id=\"single-semester\"></select>
      </div>

      <div class=\"row\">
        <div class=\"card col\">
          <h3>Tasa de Aprobacion</h3>
          <div id=\"single-approval-chart\"></div>
        </div>
        <div class=\"card col\">
          <h3>Promedio por Evaluacion</h3>
          <label>Mostrar evaluaciones:</label>
          <div id=\"single-avg-toggle\" class=\"checklist\"></div>
          <div id=\"single-avg-chart\"></div>
        </div>
      </div>

      <div class=\"card\">
        <h3>Distribucion de Notas por Evaluacion</h3>
        <label for=\"single-eval\">Seleccionar evaluacion:</label>
        <select id=\"single-eval\"></select>
        <div class=\"hist-row\">
          <div class=\"hist-plot\"><div id=\"single-hist-chart\"></div></div>
          <div class=\"hist-table\" id=\"single-stats\"></div>
        </div>
      </div>
    </section>

    <section id=\"comp-tab\" class=\"tab-panel\">
      <div class=\"card\">
        <label>Seleccionar semestres para comparar:</label>
        <div class=\"mini-actions\">
          <button type=\"button\" id=\"comp-select-all\" class=\"mini-btn\">Seleccionar todos</button>
          <button type=\"button\" id=\"comp-clear-all\" class=\"mini-btn\">Limpiar</button>
        </div>
        <div id=\"comp-semesters-toggle\" class=\"semester-checklist\"></div>
      </div>

      <div class=\"row\">
        <div class=\"card col\">
          <h3>Tasa de Aprobacion</h3>
          <div id=\"comp-approval-chart\"></div>
        </div>
        <div class=\"card col\">
          <h3>Promedio por Evaluacion</h3>
          <label>Mostrar evaluaciones:</label>
          <div id=\"comp-avg-toggle\" class=\"checklist\"></div>
          <div id=\"comp-avg-chart\"></div>
        </div>
      </div>

      <div class=\"card\">
        <h3>Comparación de Distribuciones</h3>
        <label for=\"comp-eval\">Seleccionar evaluacion:</label>
        <select id=\"comp-eval\"></select>
        <div class=\"hist-row\">
          <div class=\"hist-plot\"><div id=\"comp-hist-chart\"></div></div>
          <div class=\"hist-table\" id=\"comp-stats\"></div>
        </div>
      </div>
    </section>
  </div>

<script>
const DATA = {payload_json};
const COLORS = ["#0f766e", "#14b8a6", "#f59e0b", "#d62728", "#17becf", "#8c564b", "#e377c2", "#7f7f7f"];
const LAYOUT_BASE = {{
  font: {{family: 'Segoe UI, Calibri, Trebuchet MS, Verdana, sans-serif', size: 15, color: '#0f172a'}},
  title: {{font: {{family: 'Trebuchet MS, Segoe UI, Calibri, sans-serif', size: 24, color: '#0f172a'}}}},
  legend: {{title: {{font: {{size: 15}}}}, font: {{size: 14}}}},
  margin: {{l: 52, r: 20, t: 72, b: 52}},
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(255,255,255,0.65)',
  xaxis: {{titlefont: {{size: 17}}, tickfont: {{size: 14}}, gridcolor: 'rgba(148,163,184,0.22)', zerolinecolor: 'rgba(148,163,184,0.28)'}},
  yaxis: {{titlefont: {{size: 17}}, tickfont: {{size: 14}}, gridcolor: 'rgba(148,163,184,0.22)', zerolinecolor: 'rgba(148,163,184,0.28)'}}
}};

const state = {{
  singleSemester: null,
  singleSelectedEvals: [],
  singleEval: null,
  compSemesters: [],
  compSelectedEvals: [],
  compEval: null,
}};

function clone(obj) {{ return JSON.parse(JSON.stringify(obj)); }}
function bySemester(sem) {{ return DATA.single.bySemester[sem] || null; }}
function getSelectedValues(selectEl) {{ return Array.from(selectEl.selectedOptions).map(o => o.value); }}
function format2(x) {{ return Number.isFinite(x) ? x.toFixed(2) : '-'; }}

function setTabs() {{
  const buttons = document.querySelectorAll('.tab-btn');
  buttons.forEach(btn => {{
    btn.addEventListener('click', () => {{
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      document.getElementById(btn.dataset.tab).classList.add('active');
      renderActiveTab(btn.dataset.tab);
      setTimeout(() => window.dispatchEvent(new Event('resize')), 60);
    }});
  }});
}}

function renderActiveTab(tabId) {{
  if (tabId === 'single-tab') {{
    renderAllSingle();
  }} else if (tabId === 'comp-tab') {{
    renderAllComparison();
  }}
}}

function fillSelect(selectEl, values, selected, multiple=false) {{
  selectEl.innerHTML = '';
  values.forEach(v => {{
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = v;
    if (multiple) {{
      opt.selected = selected.includes(v);
    }} else {{
      opt.selected = (v === selected);
    }}
    selectEl.appendChild(opt);
  }});
}}

function renderChecklist(container, options, selectedValues, onChange) {{
  container.innerHTML = '';
  options.forEach(opt => {{
    const id = `${{container.id}}-${{opt}}`;
    const lbl = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.id = id;
    cb.value = opt;
    cb.checked = selectedValues.includes(opt);
    cb.addEventListener('change', onChange);
    lbl.appendChild(cb);
    lbl.append(' ' + opt);
    container.appendChild(lbl);
  }});
}}

function renderSingleControls() {{
  const single = bySemester(state.singleSemester);
  const evals = (single && single.evals) ? single.evals : [];

  if (!state.singleSelectedEvals.length) {{
    state.singleSelectedEvals = evals.slice();
  }} else {{
    state.singleSelectedEvals = state.singleSelectedEvals.filter(e => evals.includes(e));
    if (!state.singleSelectedEvals.length) state.singleSelectedEvals = evals.slice();
  }}

  if (!evals.includes(state.singleEval)) {{
    state.singleEval = evals.length ? evals[0] : null;
  }}

  const singleEvalSelect = document.getElementById('single-eval');
  fillSelect(singleEvalSelect, evals, state.singleEval, false);

  renderChecklist(document.getElementById('single-avg-toggle'), evals, state.singleSelectedEvals, () => {{
    state.singleSelectedEvals = Array.from(document.querySelectorAll('#single-avg-toggle input:checked')).map(x => x.value);
    renderSingleAvg();
  }});
}}

function renderSingleApproval() {{
  const single = bySemester(state.singleSemester);
  const div = 'single-approval-chart';
  if (!single || !single.approval) {{
    Plotly.newPlot(div, [], clone(LAYOUT_BASE));
    return;
  }}
  const ap = single.approval;
  const data = [{{
    type: 'pie',
    labels: ['Aprobados', 'Reprobados'],
    values: [ap.approved, ap.failed],
    hole: 0.45,
    textinfo: 'label+percent+value',
    textposition: 'outside',
    marker: {{colors: ['#2ca02c', '#d62728']}},
  }}];
  const layout = clone(LAYOUT_BASE);
  layout.title = {{text: `Tasa de Aprobacion - ${{state.singleSemester}} (${{ap.rate.toFixed(1)}}%)`, font: LAYOUT_BASE.title.font}};
  layout.xaxis = {{visible: false}};
  layout.yaxis = {{visible: false}};
  Plotly.newPlot(div, data, layout, {{displayModeBar: false, responsive: true}});
}}

function renderSingleAvg() {{
  const single = bySemester(state.singleSemester);
  const div = 'single-avg-chart';
  if (!single) return;

  const selected = state.singleSelectedEvals.filter(e => single.avgByEval[e] !== undefined);
  const x = [];
  const y = [];
  const colors = [];
  selected.forEach(ev => {{
    x.push(ev);
    y.push(single.avgByEval[ev]);
    colors.push('#0f766e');
  }});

  if (single.finalAvg !== null && single.finalAvg !== undefined) {{
    x.push(DATA.meta.finalGradeCol);
    y.push(single.finalAvg);
    colors.push('#f59e0b');
  }}

  const data = [{{
    type: 'bar',
    x, y,
    marker: {{color: colors}},
    text: y.map(v => Number(v).toFixed(2)),
    textposition: 'outside',
  }}];

  const layout = clone(LAYOUT_BASE);
  layout.title = {{text: `Nota Promedio por Evaluacion - ${{state.singleSemester}}`, font: LAYOUT_BASE.title.font}};
  layout.xaxis = {{...layout.xaxis, title: 'Evaluacion'}};
  layout.yaxis = {{...layout.yaxis, title: 'Nota Promedio', range: [0, 100]}};
  Plotly.newPlot(div, data, layout, {{displayModeBar: false, responsive: true}});
}}

function renderSingleHist() {{
  const single = bySemester(state.singleSemester);
  const div = 'single-hist-chart';
  if (!single || !state.singleEval || !single.histByEval[state.singleEval]) return;

  const h = single.histByEval[state.singleEval];
  const data = [{{
    type: 'bar',
    x: h.centers,
    y: h.percents,
    width: Array(h.centers.length).fill(9.2),
    marker: {{color: '#0f766e', line: {{color: '#ffffff', width: 1}}}},
    customdata: h.ranges,
    hovertemplate: 'Rango=%{{customdata[0]:.0f}}-%{{customdata[1]:.0f}}<br>Porcentaje=%{{y:.2f}}%<extra></extra>',
  }}];

  const layout = clone(LAYOUT_BASE);
  layout.title = {{text: `Histograma de Notas - ${{state.singleEval}} (${{state.singleSemester}})`, font: LAYOUT_BASE.title.font}};
  layout.xaxis = {{...layout.xaxis, title: 'Nota', range: [0, 100], tickmode: 'linear', tick0: 0, dtick: 10}};
  layout.yaxis = {{...layout.yaxis, title: 'Porcentaje de Estudiantes'}};
  Plotly.newPlot(div, data, layout, {{displayModeBar: false, responsive: true}});
}}

function renderSingleStats() {{
  const single = bySemester(state.singleSemester);
  const target = document.getElementById('single-stats');
  if (!single || !state.singleEval || !single.statsByEval[state.singleEval]) {{
    target.innerHTML = '';
    return;
  }}
  const s = single.statsByEval[state.singleEval];
  target.innerHTML = `
    <h4 class=\"stats-title\">Estadisticas - ${{state.singleEval}}</h4>
    <table>
      <thead>
        <tr><th>Sem.</th><th>Prom.</th><th>Med.</th><th>Min.</th><th>Max.</th><th>Desv.</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>${{state.singleSemester}}</td>
          <td>${{format2(s.mean)}}</td>
          <td>${{format2(s.median)}}</td>
          <td>${{format2(s.min)}}</td>
          <td>${{format2(s.max)}}</td>
          <td>${{format2(s.std)}}</td>
        </tr>
      </tbody>
    </table>
  `;
}}

function getComparisonUnionEvals(sems) {{
  const set = new Set();
  sems.forEach(sem => (bySemester(sem)?.evals || []).forEach(e => set.add(e)));
  const evals = Array.from(set);
  const order = new Map(['C1','C2','C3','C4','EJ1','EJ2','EJ3','CL1','CL2','Proyecto'].map((x,i)=>[x,i]));
  evals.sort((a,b) => (order.has(a) ? order.get(a) : 999) - (order.has(b) ? order.get(b) : 999) || a.localeCompare(b));
  return evals;
}}

function renderComparisonControls() {{
  renderChecklist(document.getElementById('comp-semesters-toggle'), DATA.semesters, state.compSemesters, () => {{
    const selected = Array.from(document.querySelectorAll('#comp-semesters-toggle input:checked')).map(x => x.value);
    state.compSemesters = selected.length ? selected : DATA.semesters.slice();
    state.compSelectedEvals = [];
    state.compEval = null;
    renderAllComparison();
  }});

  const unionEvals = getComparisonUnionEvals(state.compSemesters);

  state.compSelectedEvals = state.compSelectedEvals.filter(e => unionEvals.includes(e));
  if (!state.compSelectedEvals.length) state.compSelectedEvals = unionEvals.slice();

  if (!unionEvals.includes(state.compEval)) state.compEval = unionEvals.length ? unionEvals[0] : null;

  fillSelect(document.getElementById('comp-eval'), unionEvals, state.compEval, false);
  renderChecklist(document.getElementById('comp-avg-toggle'), unionEvals, state.compSelectedEvals, () => {{
    state.compSelectedEvals = Array.from(document.querySelectorAll('#comp-avg-toggle input:checked')).map(x => x.value);
    renderComparisonAvg();
  }});
}}

function setComparisonSemesterSelection(selectedSemesters) {{
  state.compSemesters = selectedSemesters.length ? selectedSemesters.slice() : DATA.semesters.slice();
  state.compSelectedEvals = [];
  state.compEval = null;
  renderAllComparison();
}}

function renderComparisonApproval() {{
  const rowsRaw = DATA.comparison.approvalAll || [];
  const rateBySemester = new Map(rowsRaw.map(r => [r.semester, Number(r.rate)]));
  const rows = DATA.semesters
    .filter(sem => rateBySemester.has(sem))
    .map(sem => ({{semester: sem, rate: rateBySemester.get(sem)}}));

  if (!rows.length) {{
    Plotly.newPlot('comp-approval-chart', [], clone(LAYOUT_BASE), {{displayModeBar: false, responsive: true}});
    return;
  }}

  const data = [{{
    type: 'bar',
    x: rows.map(r => r.semester),
    y: rows.map(r => r.rate),
    text: rows.map(r => Number(r.rate).toFixed(1) + '%'),
    textposition: 'outside',
    cliponaxis: false,
    marker: {{color: '#0f766e'}},
  }}];
  const layout = clone(LAYOUT_BASE);
  layout.title = {{text: 'Tasa de Aprobacion por Semestre', font: LAYOUT_BASE.title.font}};
  layout.xaxis = {{
    ...layout.xaxis,
    title: 'Semestre',
    type: 'category',
    categoryorder: 'array',
    categoryarray: rows.map(r => r.semester),
    tickmode: 'array',
    tickvals: rows.map(r => r.semester),
    ticktext: rows.map(r => r.semester),
  }};
  layout.yaxis = {{...layout.yaxis, title: 'Porcentaje de Aprobacion', range: [0, 100]}};
  Plotly.newPlot('comp-approval-chart', data, layout, {{displayModeBar: false, responsive: true}});
}}

function renderComparisonAvg() {{
  const traces = [];
  const evals = state.compSelectedEvals;
  if (!evals.length) {{
    Plotly.newPlot('comp-avg-chart', [], clone(LAYOUT_BASE));
    return;
  }}

  state.compSemesters.forEach((sem, idx) => {{
    const s = bySemester(sem);
    if (!s) return;

    const x = [];
    const y = [];
    evals.forEach(ev => {{
      if (s.avgByEval[ev] !== undefined) {{
        x.push(ev);
        y.push(s.avgByEval[ev]);
      }}
    }});
    if (s.finalAvg !== null && s.finalAvg !== undefined) {{
      x.push(DATA.meta.finalGradeCol);
      y.push(s.finalAvg);
    }}

    if (x.length) {{
      traces.push({{
        type: 'bar',
        name: sem,
        x, y,
        marker: {{color: COLORS[idx % COLORS.length]}},
      }});
    }}
  }});

  const layout = clone(LAYOUT_BASE);
  layout.title = {{text: 'Comparacion de Nota Promedio por Evaluacion', font: LAYOUT_BASE.title.font}};
  layout.barmode = 'group';
  layout.xaxis = {{...layout.xaxis, title: ''}};
  layout.yaxis = {{...layout.yaxis, title: 'Nota Promedio', range: [0, 100]}};
  Plotly.newPlot('comp-avg-chart', traces, layout, {{displayModeBar: false, responsive: true}});
}}

function renderComparisonDist() {{
  const xGrid = DATA.comparison.kdeGrid || [];
  const curvesForEval = (DATA.comparison.kdeByEval || {{}})[state.compEval] || {{}};
  const traces = [];
  state.compSemesters.forEach((sem, idx) => {{
    const y = curvesForEval[sem];
    if (!xGrid.length || !y || !y.length) return;
    traces.push({{
      type: 'scatter',
      mode: 'lines',
      name: sem,
      x: xGrid,
      y,
      line: {{width: 3, color: COLORS[idx % COLORS.length]}},
      hovertemplate: 'Semestre=%{{fullData.name}}<br>Nota=%{{x:.1f}}<br>Densidad=%{{y:.2f}}%<extra></extra>',
    }});
  }});

  const layout = clone(LAYOUT_BASE);
  layout.title = {{text: `Comparación de Distribuciones - ${{state.compEval || ''}}`, font: LAYOUT_BASE.title.font}};
  layout.xaxis = {{...layout.xaxis, title: 'Nota', range: [0, 100]}};
  layout.yaxis = {{...layout.yaxis, title: 'Densidad estimada (%)'}};
  Plotly.newPlot('comp-hist-chart', traces, layout, {{displayModeBar: false, responsive: true}});
}}

function renderComparisonStats() {{
  const target = document.getElementById('comp-stats');
  if (!state.compEval) {{ target.innerHTML = ''; return; }}

  const rows = [];
  state.compSemesters.forEach(sem => {{
    const s = bySemester(sem);
    const st = s?.statsByEval?.[state.compEval];
    if (!st) return;
    rows.push(`
      <tr>
        <td>${{sem}}</td><td>${{format2(st.mean)}}</td><td>${{format2(st.median)}}</td>
        <td>${{format2(st.min)}}</td><td>${{format2(st.max)}}</td><td>${{format2(st.std)}}</td>
      </tr>
    `);
  }});

  target.innerHTML = `
    <h4 class=\"stats-title\">Estadisticas - ${{state.compEval}}</h4>
    <table>
      <thead>
        <tr><th>Sem.</th><th>Prom.</th><th>Med.</th><th>Min.</th><th>Max.</th><th>Desv.</th></tr>
      </thead>
      <tbody>${{rows.join('')}}</tbody>
    </table>
  `;
}}

function renderAllSingle() {{
  renderSingleControls();
  renderSingleApproval();
  renderSingleAvg();
  renderSingleHist();
  renderSingleStats();
}}

function renderAllComparison() {{
  renderComparisonControls();
  renderComparisonApproval();
  renderComparisonAvg();
  renderComparisonDist();
  renderComparisonStats();
}}

function init() {{
  document.getElementById('page-title').textContent = DATA.meta.title;
  setTabs();

  fillSelect(document.getElementById('single-semester'), DATA.semesters, DATA.semesters[0], false);
  state.singleSemester = DATA.semesters[0];

  state.compSemesters = DATA.semesters.slice();

  document.getElementById('single-semester').addEventListener('change', (e) => {{
    state.singleSemester = e.target.value;
    const sem = bySemester(state.singleSemester);
    state.singleSelectedEvals = (sem?.evals || []).slice();
    state.singleEval = (sem?.evals || [null])[0];
    renderAllSingle();
  }});

  document.getElementById('single-eval').addEventListener('change', (e) => {{
    state.singleEval = e.target.value;
    renderSingleHist();
    renderSingleStats();
  }});

  document.getElementById('comp-eval').addEventListener('change', (e) => {{
    state.compEval = e.target.value;
    renderComparisonDist();
    renderComparisonStats();
  }});

  document.getElementById('comp-select-all').addEventListener('click', () => {{
    setComparisonSemesterSelection(DATA.semesters);
  }});

  document.getElementById('comp-clear-all').addEventListener('click', () => {{
    setComparisonSemesterSelection([]);
  }});

  renderAllSingle();
  renderAllComparison();
}}

init();
</script>
</body>
</html>
"""


def main() -> None:
    df, semester_eval_map = load_semester_data()
    payload = build_payload(df, semester_eval_map)
    html = html_template(json.dumps(payload, ensure_ascii=True))
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    print(f"Generated {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
