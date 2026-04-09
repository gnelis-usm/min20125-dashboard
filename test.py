
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
from pathlib import Path
from plotly.subplots import make_subplots

import plotly.express as px
import plotly.graph_objects as go

APPROVAL_MIN = 55
FINAL_GRADE_COL = 'Nota Final'   # column used for approval rate calculations
DATA_FILE = Path('Data') / 'Database v2026.01.xlsx'

NON_EVAL_COLS = {
    'Semester',
    'Semestre',
    'Nombres',
    'Nombre',
    'Rut',
    'RUT',
    'Activo',
    'Paralelo',
    'Nota Final',
    'Promedio',
}



def style_chart_figure(fig):
    fig.update_layout(
        font={'family': APP_FONT, 'size': 15, 'color': TEXT_PRIMARY},
        title={'font': {'family': TITLE_FONT, 'size': 24, 'color': TEXT_PRIMARY}},
        legend={'title': {'font': {'size': 15}}, 'font': {'size': 14}},
        margin={'l': 52, 'r': 20, 't': 72, 'b': 52},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.65)',
    )
    fig.update_xaxes(
        title_font={'size': 17},
        tickfont={'size': 14},
        gridcolor='rgba(148,163,184,0.22)',
        zerolinecolor='rgba(148,163,184,0.28)'
    )
    fig.update_yaxes(
        title_font={'size': 17},
        tickfont={'size': 14},
        gridcolor='rgba(148,163,184,0.22)',
        zerolinecolor='rgba(148,163,184,0.28)'
    )
    return fig
# Editable list: add any columns here that should never be treated as evaluations.
EXCLUDED_EVAL_COLS = {
    # 'Asistencia',
    # 'Seccion',
    # 'Observaciones',
}

# Editable list: preferred display order for evaluations in the average grade charts.
# Evaluations not listed here will appear after the listed ones, sorted alphabetically.
EVAL_ORDER = [
    'C1',
    'C2',
    'C3',
    'C4',
    'EJ1',
    'EJ2',
    'EJ3',
    'CL1',
    'CL2',
    'Proyecto',
]


def sort_evals(eval_list):
    """Sort evaluations by EVAL_ORDER; unlisted ones go last, alphabetically."""
    order_map = {col: i for i, col in enumerate(EVAL_ORDER)}
    return sorted(eval_list, key=lambda c: (order_map.get(c, len(EVAL_ORDER)), c))


def normalize_col_name(col_name):
    return str(col_name).strip().casefold()


def kde_percent_curve(values, x_grid):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros_like(x_grid)

    sample_std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    bandwidth = 1.06 * sample_std * (values.size ** (-1.0 / 5.0)) if sample_std > 0 else 0.0
    bandwidth = max(bandwidth, 2.0)

    # Gaussian-kernel density estimate scaled to percentage units.
    normalized_diff = (x_grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * normalized_diff ** 2).sum(axis=1)
    density /= (values.size * bandwidth * np.sqrt(2.0 * np.pi))
    return density * 100.0


def histogram_percent_bins(values, bin_start=0, bin_end=100, bin_size=10):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[(values >= bin_start) & (values <= bin_end)]
    if values.size == 0:
        return np.array([]), np.array([]), np.array([])

    # Fixed bin edges to keep comparability and avoid leaking raw values.
    edges = np.arange(bin_start, bin_end + bin_size, bin_size, dtype=float)
    counts, _ = np.histogram(values, bins=edges)
    percents = (counts / values.size) * 100.0
    centers = edges[:-1] + (bin_size / 2.0)
    return centers, percents, edges


NON_EVAL_COLS_NORM = {normalize_col_name(col) for col in NON_EVAL_COLS}
EXCLUDED_EVAL_COLS_NORM = {normalize_col_name(col) for col in EXCLUDED_EVAL_COLS}


def load_data_with_semesters():
    data_file = DATA_FILE
    if not data_file.exists():
        available = sorted(Path('Data').glob('*.xlsx'))
        if not available:
            raise FileNotFoundError('No se encontraron archivos Excel en la carpeta Data.')
        data_file = available[0]

    workbook = pd.read_excel(data_file, sheet_name=None, header=0)

    frames = []
    semester_eval_map = {}
    for sheet_name, temp_df in workbook.items():
        temp_df = temp_df.copy()
        temp_df['Semester'] = str(sheet_name)

        eval_cols_sheet = [
            col for col in temp_df.columns
            if normalize_col_name(col) not in NON_EVAL_COLS_NORM
            and normalize_col_name(col) not in EXCLUDED_EVAL_COLS_NORM
            and pd.api.types.is_numeric_dtype(temp_df[col])
        ]
        if not eval_cols_sheet:
            continue

        semester_eval_map[str(sheet_name)] = sort_evals(eval_cols_sheet)
        frames.append(temp_df)

    if not frames:
        raise ValueError(
            'Ninguna hoja del archivo contiene columnas validas de evaluaciones.'
        )

    final_df = pd.concat(frames, ignore_index=True)
    final_df['Semester'] = final_df['Semester'].astype(str)
    return final_df, semester_eval_map


def empty_figure(message):
    fig = go.Figure()
    fig.update_layout(
        title=message,
        xaxis={'visible': False},
        yaxis={'visible': False},
        annotations=[
            {
                'text': message,
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 16, 'family': APP_FONT, 'color': TEXT_SECONDARY}
            }
        ]
    )
    return style_chart_figure(fig)


def build_approval_rate_chart(selected_semesters=None):
    if FINAL_GRADE_COL not in df.columns:
        return empty_figure(f'"{FINAL_GRADE_COL}" no disponible en los datos')

    semester_order = selected_semesters or SEMESTERS
    if not semester_order:
        return empty_figure('No hay semestres disponibles')

    approval_records = []
    for semester in semester_order:
        sem_df = df[df['Semester'] == semester]
        if sem_df.empty:
            continue
        final_grades = sem_df[FINAL_GRADE_COL].dropna()
        if final_grades.empty:
            continue
        approval_records.append(
            {
                'Semester': semester,
                'Tasa de Aprobacion': (final_grades >= APPROVAL_MIN).mean() * 100,
            }
        )

    if not approval_records:
        return empty_figure(f'"{FINAL_GRADE_COL}" no disponible para los semestres seleccionados')

    approval_df = pd.DataFrame(approval_records)
    fig = px.bar(
        approval_df,
        x='Semester',
        y='Tasa de Aprobacion',
        title='Tasa de Aprobacion por Semestre' if selected_semesters is None else 'Comparacion de Tasa de Aprobacion por Semestre',
        text=approval_df['Tasa de Aprobacion'].round(1),
        category_orders={'Semester': semester_order}
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_yaxes(range=[0, 100], title='Porcentaje de Aprobacion')
    fig.update_xaxes(title='Semestre')
    return style_chart_figure(fig)


df, SEMESTER_EVAL_MAP = load_data_with_semesters()
SEMESTERS = sorted(df['Semester'].dropna().unique().tolist())
DEFAULT_SEMESTER = SEMESTERS[0] if SEMESTERS else None
ALL_EVAL_COLS = sort_evals({col for cols in SEMESTER_EVAL_MAP.values() for col in cols})
DEFAULT_SINGLE_EVAL = SEMESTER_EVAL_MAP.get(DEFAULT_SEMESTER, [None])[0]
DEFAULT_COMPARISON_EVAL = ALL_EVAL_COLS[0] if ALL_EVAL_COLS else None

app = dash.Dash(__name__)

# Modern color scheme
PRIMARY_COLOR = '#0f766e'
SECONDARY_COLOR = '#14b8a6'
ACCENT_COLOR = '#f59e0b'
BG_COLOR = '#f2f7f8'
CARD_BG = '#ffffff'
TEXT_PRIMARY = '#0f172a'
TEXT_SECONDARY = '#475569'
BORDER_COLOR = '#d7e3e9'
APP_FONT = '"Segoe UI", "Calibri", "Trebuchet MS", Verdana, sans-serif'
TITLE_FONT = '"Trebuchet MS", "Segoe UI", "Calibri", sans-serif'
TAB_FONT = '"Segoe UI", "Calibri", "Trebuchet MS", sans-serif'

# Common styles
CARD_STYLE = {
    'backgroundColor': CARD_BG,
    'borderRadius': '12px',
    'boxShadow': '0 10px 24px rgba(15, 23, 42, 0.08)',
    'padding': '24px',
    'marginBottom': '20px',
    'border': f'1px solid {BORDER_COLOR}'
}

SECTION_TITLE_STYLE = {
    'color': TEXT_PRIMARY,
    'fontSize': '20px',
    'fontWeight': '700',
    'marginBottom': '16px',
    'paddingBottom': '10px',
    'borderBottom': f'2px solid {PRIMARY_COLOR}'
}

LABEL_STYLE = {
    'fontWeight': '600',
    'color': TEXT_PRIMARY,
    'marginBottom': '10px',
    'fontSize': '15px',
    'display': 'block'
}

TAB_STYLE = {
    'padding': '9px 18px',
    'fontSize': '15px',
    'fontWeight': '700',
    'height': '42px',
    'lineHeight': '24px',
    'display': 'flex',
    'alignItems': 'center',
    'justifyContent': 'center',
    'fontFamily': TAB_FONT,
    'letterSpacing': '0.2px',
    'border': f'1px solid {BORDER_COLOR}',
    'borderBottom': 'none',
    'borderTopLeftRadius': '10px',
    'borderTopRightRadius': '10px',
    'backgroundColor': '#e9f2f3',
    'color': TEXT_SECONDARY,
}

TAB_SELECTED_STYLE = {
    **TAB_STYLE,
    'fontWeight': '800',
    'borderTop': f'3px solid {PRIMARY_COLOR}',
    'backgroundColor': CARD_BG,
    'color': TEXT_PRIMARY,
}

TOP_ROW_STYLE = {
    'display': 'flex',
    'gap': '20px',
    'alignItems': 'stretch',
    'flexWrap': 'wrap',
    'marginBottom': '20px',
}

TOP_COL_STYLE = {
    'flex': '1 1 480px',
    'minWidth': '320px',
}

HIST_ROW_STYLE = {
    'display': 'flex',
    'gap': '20px',
    'alignItems': 'flex-start',
    'flexWrap': 'wrap',
}

HIST_GRAPH_COL_STYLE = {
    'flex': '2 1 620px',
    'minWidth': '360px',
}

HIST_TABLE_COL_STYLE = {
    'flex': '1 1 320px',
    'minWidth': '280px',
}

app.layout = html.Div([
    html.Div([
        html.H1(
            'MIN102 y MIN20125 - Dashboard de Rendimiento Académico',
            style={
                'textAlign': 'center',
                'marginBottom': '8px',
                'color': TEXT_PRIMARY,
                'fontFamily': TITLE_FONT,
                'fontSize': '40px',
                'fontWeight': '800',
                'letterSpacing': '-0.7px',
                'lineHeight': '1.15'
            }
        )
    ], style={'backgroundImage': f'linear-gradient(135deg, {CARD_BG} 0%, #eaf6f5 100%)', 'paddingTop': '30px', 'paddingBottom': '20px', 'borderBottom': f'3px solid {PRIMARY_COLOR}'}),

    dcc.Tabs([
        dcc.Tab(label='Semestre Individual',
                style=TAB_STYLE,
                selected_style=TAB_SELECTED_STYLE,
                children=[
            html.Div([
                html.Div([
                    html.Label('Seleccionar Semestre:', style=LABEL_STYLE),
                    dcc.Dropdown(
                        id='semester-selector',
                        options=[{'label': sem, 'value': sem} for sem in SEMESTERS],
                        value=DEFAULT_SEMESTER,
                        clearable=False,
                        style={'width': '100%'}
                    ),
                ], style={**CARD_STYLE, 'marginBottom': '25px'}),

                html.Div([
                    html.Div([
                        html.H3('Tasa de Aprobación', style=SECTION_TITLE_STYLE),
                        dcc.Graph(id='single-approval-rate-chart'),
                    ], style={**CARD_STYLE, **TOP_COL_STYLE}),

                    html.Div([
                        html.H3('Promedio por Evaluación', style=SECTION_TITLE_STYLE),
                        html.Div([
                            html.Label('Mostrar evaluaciones:', style=LABEL_STYLE),
                            dcc.Checklist(
                                id='single-avg-eval-toggle',
                                options=[{'label': col, 'value': col} for col in SEMESTER_EVAL_MAP.get(DEFAULT_SEMESTER, [])],
                                value=SEMESTER_EVAL_MAP.get(DEFAULT_SEMESTER, []),
                                inline=True,
                                inputStyle={'marginRight': '6px', 'cursor': 'pointer'},
                                labelStyle={'marginRight': '14px', 'cursor': 'pointer', 'fontSize': '14px'}
                            )
                        ]),
                        dcc.Graph(id='single-subject-avg-chart'),
                    ], style={**CARD_STYLE, **TOP_COL_STYLE}),
                ], style=TOP_ROW_STYLE),

                html.Div([
                    html.H3('Distribución de Notas por Evaluación', style=SECTION_TITLE_STYLE),
                    html.Div([
                        html.Label('Seleccionar evaluación:', style=LABEL_STYLE),
                        dcc.Dropdown(
                            id='single-evaluation-selector',
                            options=[{'label': col, 'value': col} for col in SEMESTER_EVAL_MAP.get(DEFAULT_SEMESTER, [])],
                            value=DEFAULT_SINGLE_EVAL,
                            clearable=False,
                            style={'width': '100%'}
                        )
                    ], style={'marginBottom': '16px'}),
                    html.Div([
                        html.Div([
                            dcc.Graph(id='single-evaluation-histogram')
                        ], style=HIST_GRAPH_COL_STYLE),
                        html.Div(
                            id='single-stats-table',
                            style={**HIST_TABLE_COL_STYLE, 'marginTop': '10px'}
                        )
                    ], style=HIST_ROW_STYLE)
                ], style=CARD_STYLE)
            ], style={'margin': '20px'})
        ]),

        dcc.Tab(label='Comparación de Semestres',
            style=TAB_STYLE,
            selected_style=TAB_SELECTED_STYLE,
                children=[
            html.Div([
                html.Div([
                    html.Label('Seleccionar semestres para comparar:', style=LABEL_STYLE),
                    dcc.Dropdown(
                        id='comparison-semester-selector',
                        options=[{'label': sem, 'value': sem} for sem in SEMESTERS],
                        value=SEMESTERS,
                        multi=True,
                        clearable=False,
                        style={'width': '100%'}
                    ),
                ], style={**CARD_STYLE, 'marginBottom': '25px'}),

                html.Div([
                    html.Div([
                        html.H3('Tasa de Aprobación', style=SECTION_TITLE_STYLE),
                        dcc.Graph(id='comparison-approval-rate-chart', figure=build_approval_rate_chart()),
                    ], style={**CARD_STYLE, **TOP_COL_STYLE}),

                    html.Div([
                        html.H3('Promedio por Evaluación', style=SECTION_TITLE_STYLE),
                        html.Div([
                            html.Label('Mostrar evaluaciones:', style=LABEL_STYLE),
                            dcc.Checklist(
                                id='comparison-avg-eval-toggle',
                                options=[{'label': col, 'value': col} for col in ALL_EVAL_COLS],
                                value=ALL_EVAL_COLS,
                                inline=True,
                                inputStyle={'marginRight': '6px', 'cursor': 'pointer'},
                                labelStyle={'marginRight': '14px', 'cursor': 'pointer', 'fontSize': '14px'}
                            )
                        ]),
                        dcc.Graph(id='comparison-subject-avg-chart'),
                    ], style={**CARD_STYLE, **TOP_COL_STYLE}),
                ], style=TOP_ROW_STYLE),

                html.Div([
                    html.H3('Histograma Comparativo por Evaluación', style=SECTION_TITLE_STYLE),
                    html.Div([
                        html.Label('Seleccionar evaluación:', style=LABEL_STYLE),
                        dcc.Dropdown(
                            id='comparison-evaluation-selector',
                            options=[{'label': col, 'value': col} for col in ALL_EVAL_COLS],
                            value=DEFAULT_COMPARISON_EVAL,
                            clearable=False,
                            style={'width': '100%'}
                        )
                    ], style={'marginBottom': '16px'}),
                    html.Div([
                        html.Div([
                            dcc.Graph(id='comparison-evaluation-histogram')
                        ], style=HIST_GRAPH_COL_STYLE),
                        html.Div(
                            id='comparison-stats-table',
                            style={**HIST_TABLE_COL_STYLE, 'marginTop': '10px'}
                        )
                    ], style=HIST_ROW_STYLE)
                ], style=CARD_STYLE)
            ], style={'margin': '20px'})
        ])
    ],
    parent_style={'backgroundColor': BG_COLOR, 'padding': '0 20px'},
    style={'height': '44px'},
    content_style={'backgroundColor': BG_COLOR, 'minHeight': 'calc(100vh - 44px)', 'fontFamily': APP_FONT, 'color': TEXT_PRIMARY, 'backgroundImage': 'radial-gradient(circle at top right, rgba(20,184,166,0.10), transparent 35%)'})
])

@app.callback(
    Output('single-approval-rate-chart', 'figure'),
    Input('semester-selector', 'value')
)
def update_single_approval_rate_chart(selected_semester):
    dff = df[df['Semester'] == selected_semester]
    if dff.empty:
        return empty_figure('No hay datos para el semestre seleccionado')

    if FINAL_GRADE_COL not in dff.columns or dff[FINAL_GRADE_COL].dropna().empty:
        return empty_figure(f'"{FINAL_GRADE_COL}" no disponible para este semestre')

    approved = (dff[FINAL_GRADE_COL] >= APPROVAL_MIN).sum()
    failed = (dff[FINAL_GRADE_COL] < APPROVAL_MIN).sum()
    approval_rate = (approved / len(dff[dff[FINAL_GRADE_COL].notna()])) * 100

    fig = px.pie(
        names=['Aprobados', 'Reprobados'],
        values=[approved, failed],
        title=f'Tasa de Aprobacion - {selected_semester} ({approval_rate:.1f}%)',
        hole=0.45,
        color=['Aprobados', 'Reprobados'],
        color_discrete_map={'Aprobados': '#2ca02c', 'Reprobados': '#d62728'}
    )
    fig.update_traces(textinfo='label+percent+value', textposition='outside', textfont={'size': 14})
    return style_chart_figure(fig)

@app.callback(
    Output('single-avg-eval-toggle', 'options'),
    Output('single-avg-eval-toggle', 'value'),
    Input('semester-selector', 'value'),
    State('single-avg-eval-toggle', 'value')
)
def update_single_avg_toggle_options(selected_semester, current_selection):
    eval_cols = SEMESTER_EVAL_MAP.get(selected_semester, [])
    options = [{'label': col, 'value': col} for col in eval_cols]
    # Keep previously selected evals that still exist; otherwise select all
    new_value = [col for col in (current_selection or []) if col in eval_cols] or eval_cols
    return options, new_value


@app.callback(
    Output('single-subject-avg-chart', 'figure'),
    Input('single-avg-eval-toggle', 'value'),
    State('semester-selector', 'value')
)
def update_single_avg_chart(selected_evals, selected_semester):
    dff = df[df['Semester'] == selected_semester]
    if dff.empty:
        return empty_figure('No hay datos para el semestre seleccionado')

    eval_cols = sort_evals([col for col in (selected_evals or []) if col in SEMESTER_EVAL_MAP.get(selected_semester, [])])
    if not eval_cols:
        return empty_figure('Selecciona al menos una evaluación para mostrar')

    avg_by_eval = dff[eval_cols].mean().dropna().reindex(eval_cols).dropna()
    if avg_by_eval.empty:
        return empty_figure('No hay notas para calcular promedios en este semestre')

    chart_df = avg_by_eval.rename_axis('Evaluacion').reset_index(name='Nota')
    chart_df['Tipo'] = 'Evaluación'
    display_order = list(eval_cols)

    if FINAL_GRADE_COL in dff.columns:
        final_avg = dff[FINAL_GRADE_COL].mean()
        if pd.notna(final_avg):
            extra = pd.DataFrame({'Evaluacion': [FINAL_GRADE_COL], 'Nota': [round(final_avg, 2)], 'Tipo': ['Nota Final']})
            chart_df = pd.concat([chart_df, extra], ignore_index=True)
            display_order.append(FINAL_GRADE_COL)

    fig = px.bar(
        chart_df,
        x='Evaluacion',
        y='Nota',
        color='Tipo',
        color_discrete_map={'Evaluación': PRIMARY_COLOR, 'Nota Final': ACCENT_COLOR},
        text='Nota',
        category_orders={'Evaluacion': display_order},
        labels={'Evaluacion': 'Evaluación', 'Nota': 'Nota Promedio', 'Tipo': ''},
        title=f'Nota Promedio por Evaluación - {selected_semester}',
    )
    fig.update_traces(textposition='outside', texttemplate='%{text:.2f}')
    fig.update_yaxes(range=[0, 100])
    return style_chart_figure(fig)

@app.callback(
    Output('single-evaluation-histogram', 'figure'),
    Input('single-evaluation-selector', 'value'),
    State('semester-selector', 'value')
)
def update_single_evaluation_histogram(selected_eval, selected_semester):
    dff = df[df['Semester'] == selected_semester]
    if dff.empty:
        return empty_figure('No hay datos para el semestre seleccionado')
    if not selected_eval:
        return empty_figure('Selecciona una evaluación')
    if selected_eval not in SEMESTER_EVAL_MAP.get(selected_semester, []):
        return empty_figure('La evaluación seleccionada no existe en este semestre')

    values = dff[selected_eval].dropna().values
    centers, percents, edges = histogram_percent_bins(values, bin_start=0, bin_end=100, bin_size=10)
    if centers.size == 0:
        return empty_figure('No hay notas disponibles para la evaluación seleccionada')

    custom_ranges = np.column_stack([edges[:-1], edges[1:]])
    fig = go.Figure(
        data=[
            go.Bar(
                x=centers,
                y=percents,
                width=9.2,
                marker={'color': PRIMARY_COLOR, 'line': {'color': CARD_BG, 'width': 1}},
                customdata=custom_ranges,
                hovertemplate='Rango=%{customdata[0]:.0f}-%{customdata[1]:.0f}<br>Porcentaje=%{y:.2f}%<extra></extra>',
            )
        ]
    )
    fig.update_layout(
        title=f'Histograma de Notas - {selected_eval} ({selected_semester})',
        xaxis_title='Nota',
        yaxis_title='Porcentaje de Estudiantes',
    )
    fig.update_xaxes(range=[0, 100], tickmode='linear', tick0=0, dtick=10)
    return style_chart_figure(fig)


@app.callback(
    Output('single-evaluation-selector', 'options'),
    Output('single-evaluation-selector', 'value'),
    Input('semester-selector', 'value'),
    State('single-evaluation-selector', 'value')
)
def update_single_evaluation_options(selected_semester, current_eval):
    eval_cols = SEMESTER_EVAL_MAP.get(selected_semester, [])
    options = [{'label': col, 'value': col} for col in eval_cols]

    if not eval_cols:
        return options, None

    selected_value = current_eval if current_eval in eval_cols else eval_cols[0]
    return options, selected_value


@app.callback(
    Output('comparison-avg-eval-toggle', 'options'),
    Output('comparison-avg-eval-toggle', 'value'),
    Input('comparison-semester-selector', 'value'),
    State('comparison-avg-eval-toggle', 'value')
)
def update_comparison_avg_toggle_options(selected_semesters, current_selection):
    if not selected_semesters:
        return [], []
    union_evals = sort_evals({
        col for semester in selected_semesters
        for col in SEMESTER_EVAL_MAP.get(semester, [])
    })
    options = [{'label': col, 'value': col} for col in union_evals]
    new_value = sort_evals([col for col in (current_selection or []) if col in union_evals]) or union_evals
    return options, new_value


@app.callback(
    Output('comparison-subject-avg-chart', 'figure'),
    Input('comparison-avg-eval-toggle', 'value'),
    State('comparison-semester-selector', 'value')
)
def update_comparison_avg_chart(selected_evals, selected_semesters):
    if not selected_semesters:
        return empty_figure('Selecciona al menos un semestre')
    if not selected_evals:
        return empty_figure('Selecciona al menos una evaluación para mostrar')

    ordered_selected_semesters = [semester for semester in SEMESTERS if semester in selected_semesters]
    if not ordered_selected_semesters:
        return empty_figure('No hay datos para los semestres seleccionados')

    dff = df[df['Semester'].isin(ordered_selected_semesters)]
    if dff.empty:
        return empty_figure('No hay datos para los semestres seleccionados')

    avg_records = []
    ordered_evals = sort_evals(selected_evals)
    has_final_grade = False
    for semester in ordered_selected_semesters:
        sem_df = dff[dff['Semester'] == semester]
        for eval_col in ordered_evals:
            if eval_col not in SEMESTER_EVAL_MAP.get(semester, []):
                continue
            if eval_col in sem_df.columns:
                avg_value = sem_df[eval_col].mean()
                if pd.notna(avg_value):
                    avg_records.append(
                        {'Semester': semester, 'Evaluacion': eval_col, 'Nota': avg_value}
                    )
        if FINAL_GRADE_COL in sem_df.columns:
            final_avg = sem_df[FINAL_GRADE_COL].mean()
            if pd.notna(final_avg):
                avg_records.append({'Semester': semester, 'Evaluacion': FINAL_GRADE_COL, 'Nota': final_avg})
                has_final_grade = True

    if not avg_records:
        return empty_figure('No hay evaluaciones disponibles para comparar')

    display_order = ordered_evals + ([FINAL_GRADE_COL] if has_final_grade else [])
    avg_df = pd.DataFrame(avg_records)

    fig = px.bar(
        avg_df,
        x='Evaluacion',
        y='Nota',
        color='Semester',
        barmode='group',
        title='Comparacion de Nota Promedio por Evaluacion',
        labels={'Nota': 'Nota Promedio', 'Evaluacion': 'Evaluación', 'Semester': 'Semestre'},
        category_orders={'Evaluacion': display_order, 'Semester': ordered_selected_semesters}
    )
    fig.update_yaxes(range=[0, 100])
    return style_chart_figure(fig)


@app.callback(
    Output('comparison-evaluation-selector', 'options'),
    Output('comparison-evaluation-selector', 'value'),
    Input('comparison-semester-selector', 'value'),
    State('comparison-evaluation-selector', 'value')
)
def update_comparison_evaluation_options(selected_semesters, current_eval):
    if not selected_semesters:
        return [], None

    union_evals = sort_evals({
        eval_col
        for semester in selected_semesters
        for eval_col in SEMESTER_EVAL_MAP.get(semester, [])
    })
    options = [{'label': col, 'value': col} for col in union_evals]

    if not union_evals:
        return options, None

    selected_value = current_eval if current_eval in union_evals else union_evals[0]
    return options, selected_value


@app.callback(
    Output('comparison-evaluation-histogram', 'figure'),
    Input('comparison-evaluation-selector', 'value'),
    State('comparison-semester-selector', 'value')
)
def update_comparison_histogram(selected_eval, selected_semesters):
    if not selected_semesters:
        return empty_figure('Selecciona al menos un semestre')
    if not selected_eval:
        return empty_figure('Selecciona una evaluación')

    x_min = 0
    x_max = 100
    x_grid = np.linspace(x_min, x_max, 401)

    ordered_selected_semesters = [semester for semester in SEMESTERS if semester in selected_semesters]
    if not ordered_selected_semesters:
        return empty_figure('No hay datos para los semestres seleccionados')

    valid_semesters = [
        semester for semester in ordered_selected_semesters
        if selected_eval in SEMESTER_EVAL_MAP.get(semester, [])
    ]
    if not valid_semesters:
        return empty_figure('La evaluación seleccionada no existe en los semestres elegidos')

    dff = df[df['Semester'].isin(valid_semesters)]
    if dff.empty:
        return empty_figure('No hay datos para los semestres seleccionados')

    dff = dff.dropna(subset=[selected_eval])
    if dff.empty:
        return empty_figure('No hay notas disponibles para la evaluación seleccionada')

    semester_colors = [
        PRIMARY_COLOR,
        SECONDARY_COLOR,
        ACCENT_COLOR,
        '#d62728',
        '#17becf',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
    ]

    fig = go.Figure()
    for index, semester in enumerate(valid_semesters):
        sem_values = dff.loc[dff['Semester'] == semester, selected_eval].dropna()
        sem_values = sem_values[(sem_values >= x_min) & (sem_values <= x_max)]
        if sem_values.empty:
            continue
        y_curve = kde_percent_curve(sem_values.values, x_grid)
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_curve,
                name=semester,
                mode='lines',
                line={'width': 3, 'color': semester_colors[index % len(semester_colors)]},
                hovertemplate='Semestre=%{fullData.name}<br>Nota=%{x:.1f}<br>Densidad=%{y:.2f}%<extra></extra>',
            )
        )

    if not fig.data:
        return empty_figure('No hay notas disponibles para la evaluación seleccionada')

    fig.update_layout(
        title=f'Densidad Comparativa (KDE) - {selected_eval}',
        xaxis_title='Nota',
        yaxis_title='Densidad estimada (%)',
        legend_title='Semestre',
    )
    fig.update_xaxes(range=[x_min, x_max])
    return style_chart_figure(fig)


TABLE_HEADER_STYLE = {
    'backgroundColor': PRIMARY_COLOR,
    'color': 'white',
    'padding': '11px 14px',
    'textAlign': 'center',
    'fontWeight': '700',
    'border': 'none',
    'fontSize': '16px'
}
TABLE_CELL_STYLE = {
    'padding': '10px 14px',
    'textAlign': 'center',
    'border': f'1px solid {BORDER_COLOR}',
    'fontSize': '15px',
    'color': TEXT_PRIMARY
}
TABLE_ROW_ALT_STYLE = {**{'backgroundColor': '#edf4f7'}, 'padding': '10px 14px', 'textAlign': 'center', 'border': f'1px solid {BORDER_COLOR}', 'fontSize': '15px', 'color': TEXT_PRIMARY}


def build_stats_table(stats_rows, eval_label):
    """Build an html.Table from a list of (semester, mean, median, min, max, std) tuples."""
    headers = ['Semestre', 'Promedio', 'Mediana', 'Mínimo', 'Máximo', 'Desv. Est.']
    header_row = html.Tr([html.Th(h, style=TABLE_HEADER_STYLE) for h in headers])
    data_rows = [
        html.Tr(
            [html.Td(cell, style=TABLE_CELL_STYLE if i % 2 == 0 else TABLE_ROW_ALT_STYLE) for cell in row],
        )
        for i, row in enumerate(stats_rows)
    ]
    return html.Div([
        html.H4(f'Estadísticas - {eval_label}', style={'marginBottom': '10px', 'fontSize': '20px', 'color': TEXT_PRIMARY}),
        html.Table(
            [html.Thead(header_row), html.Tbody(data_rows)],
            style={'borderCollapse': 'collapse', 'width': '100%', 'fontSize': '15px'}
        )
    ])


@app.callback(
    Output('single-stats-table', 'children'),
    Input('single-evaluation-selector', 'value'),
    State('semester-selector', 'value')
)
def update_single_stats_table(selected_eval, selected_semester):
    if not selected_eval:
        return None
    dff = df[df['Semester'] == selected_semester]
    if dff.empty or selected_eval not in dff.columns:
        return None
    s = dff[selected_eval].dropna()
    if s.empty:
        return None
    row = [
        selected_semester,
        f'{s.mean():.2f}',
        f'{s.median():.2f}',
        f'{s.min():.2f}',
        f'{s.max():.2f}',
        f'{s.std():.2f}'
    import plotly.graph_objects as go

    # Static dashboard generation
    def generate_static_dashboard():
        """Generate a static HTML dashboard with all visualizations."""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=("Approval Rate", "Average by Evaluation", 
                           "Histogram", "Statistics", "Comparison", "Distribution"),
            specs=[[{}, {}], [{}, {}], [{}, {}]]
        )
        
        # Add your charts to subplots
        # ... add traces using fig.add_trace()
        
        fig.write_html("dashboard.html")
        print("Static dashboard saved as dashboard.html")

    generate_static_dashboard()
    ]
    return build_stats_table([row], selected_eval)


@app.callback(
    Output('comparison-stats-table', 'children'),
    Input('comparison-evaluation-selector', 'value'),
    State('comparison-semester-selector', 'value')
)
def update_comparison_stats_table(selected_eval, selected_semesters):
    if not selected_eval or not selected_semesters:
        return None
    valid_semesters = [
        sem for sem in selected_semesters
        if selected_eval in SEMESTER_EVAL_MAP.get(sem, [])
    ]
    if not valid_semesters:
        return None
    rows = []
    for sem in valid_semesters:
        s = df[(df['Semester'] == sem)][selected_eval].dropna()
        if s.empty:
            continue
        rows.append([
            sem,
            f'{s.mean():.2f}',
            f'{s.median():.2f}',
            f'{s.min():.2f}',
            f'{s.max():.2f}',
            f'{s.std():.2f}'
        ])
    if not rows:
        return None
    return build_stats_table(rows, selected_eval)


if __name__ == '__main__':
    app.run()