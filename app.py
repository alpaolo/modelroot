"""
ModelRoot Streamlit app — catalog-first UX for license-aware model discovery.

Modes: Model Catalog (default), Model Detail (1-hop graph), License Intelligence, Graph Explorer.
Catalog: filter + table row selection only. Model name search uses HF token boundaries (/ - _ .), not raw substring (bert ≠ roberta).
Detail: Find model searches all Model nodes in Neo4j; radio lists all matching versions; fixed model bar under header.
Graph Explorer: find field + in-memory match list (detail_model).
Sidebar query_limit controls LIMIT on list/graph queries across all modes.
Sidebar data_view_height sets native st.dataframe / graph iframe height in pixels.
Fixed main chrome: 10vh header + 10vh footer in center column only (sidebar unchanged).
Minimal header: ModelRoot label in sidebar; mode title in fixed main header (no emoji).
UI constants (colors, columns, license URLs) live in constants/app_constants.py.
Main chrome layout constants live in app.py (center-column header/footer).
Neo4j queries live in constants/query.py.
"""
import html
import importlib
import json
import os
import re
import sys
import tempfile
from urllib.parse import quote

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from neo4j import GraphDatabase
from pyvis.network import Network

_APP_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _APP_ROOT)
sys.path.insert(0, os.path.join(_APP_ROOT, ".env"))
import config as env
from constants.brand_urls import BRAND_HF_URLS
from constants.dataset_urls import DATASET_DOCUMENTATION_URLS
from constants.tech_domains import resolve_tech_domain_names_for_task


def _load_query_constants():
    """Reload query.py each Streamlit rerun — avoids stale sys.modules after deploy."""
    query_py_path = os.path.join(_APP_ROOT, "constants", "query.py")
    required_symbols = ("MODEL_PICKER_BROWSE_CYPHER", "MODEL_PICKER_SEARCH_CYPHER")
    with open(query_py_path, encoding="utf-8") as query_file:
        query_source = query_file.read()
    missing_symbols = [symbol for symbol in required_symbols if symbol not in query_source]
    if missing_symbols:
        raise ImportError(
            f"Outdated {query_py_path} — missing {', '.join(missing_symbols)}. "
            "Deploy the latest constants/query.py with app.py, then "
            "rm -rf constants/__pycache__ && restart Streamlit."
        )
    import constants.query as query_constants
    return importlib.reload(query_constants)


query_constants = _load_query_constants()
ALL_MODEL_NAMES_BY_DOWNLOADS_CYPHER = query_constants.ALL_MODEL_NAMES_BY_DOWNLOADS_CYPHER
DATABASE_SNAPSHOT_CYPHER = query_constants.DATABASE_SNAPSHOT_CYPHER
DERIVED_MODELS_CYPHER = query_constants.DERIVED_MODELS_CYPHER
DISTINCT_BRANDS_CYPHER = query_constants.DISTINCT_BRANDS_CYPHER
DISTINCT_TASKS_CYPHER = query_constants.DISTINCT_TASKS_CYPHER
GRAPH_EXPLORER_SUBGRAPH_CYPHER = query_constants.GRAPH_EXPLORER_SUBGRAPH_CYPHER
LICENSE_GROUP_METADATA_CYPHER = query_constants.LICENSE_GROUP_METADATA_CYPHER
MODEL_CATALOG_CYPHER = query_constants.MODEL_CATALOG_CYPHER
MODEL_DETAIL_CYPHER = query_constants.MODEL_DETAIL_CYPHER
MODEL_NEIGHBORHOOD_CYPHER = query_constants.MODEL_NEIGHBORHOOD_CYPHER
MODEL_PICKER_BROWSE_CYPHER = query_constants.MODEL_PICKER_BROWSE_CYPHER
MODEL_PICKER_SEARCH_CYPHER = query_constants.MODEL_PICKER_SEARCH_CYPHER
MODELS_BY_LICENSE_GROUP_CYPHER = query_constants.MODELS_BY_LICENSE_GROUP_CYPHER
MODELS_USING_DATASET_CYPHER = query_constants.MODELS_USING_DATASET_CYPHER
RELATIONSHIP_TYPES_CYPHER = query_constants.RELATIONSHIP_TYPES_CYPHER
TOP_DATASETS_BY_USAGE_CYPHER = query_constants.TOP_DATASETS_BY_USAGE_CYPHER
TOP_LICENSES_CYPHER = query_constants.TOP_LICENSES_CYPHER
TOP_MODELS_BY_RISK_CYPHER = query_constants.TOP_MODELS_BY_RISK_CYPHER
from constants.app_constants import (
    BENCHMARK_PANEL_BACKGROUND_COLOR,
    BENCHMARK_PANEL_BORDER_COLOR,
    CATALOG_DISPLAY_COLUMNS,
    CATALOG_LINK_COLUMNS,
    CATALOG_STYLED_COLUMNS,
    CENTER_MODEL_NODE_COLOR,
    DATA_VIEW_HEIGHT_DEFAULT,
    DATA_VIEW_HEIGHT_MAX,
    DATA_VIEW_HEIGHT_MIN,
    DATAFRAME_PLACEHOLDER,
    DEFAULT_RELATION_COLOR,
    GRAPH_BACKGROUND_COLOR,
    GRAPH_EXPLORER_DEFAULT_REL_LIMIT,
    GRAPH_EXPLORER_PHYSICS_OPTIONS,
    LICENSE_GROUP_CELL_STYLE,
    LICENSE_GROUP_COLORS,
    LICENSE_OFFICIAL_DOCUMENT_URLS,
    LICENSE_WIKIPEDIA_PAGES,
    LINK_COLUMN_DISPLAY_TEXT,
    MINI_GRAPH_OPTIONS,
    MODE_OPTIONS,
    MODEL_PICKER_BROWSE_LIMIT,
    MODEL_PICKER_MATCH_LIMIT,
    NEIGHBORHOOD_KIND_LABELS,
    NEIGHBORHOOD_LINK_DISPLAY_REGEX,
    NEIGHBORHOOD_PLACEHOLDER_LINK_BASE,
    QUERY_LIMIT_DEFAULT,
    QUERY_LIMIT_MAX,
    QUERY_LIMIT_MIN,
    RELATION_COLORS,
    TOP_MODELS_BY_RISK_DISPLAY_COLUMNS,
    TOP_MODELS_BY_RISK_STYLED_COLUMNS,
    UNCLASSIFIED_LICENSE_GROUP_LEGEND,
    UNKNOWN_LICENSE_GROUP_COLOR,
)
NEO4J_URI = env.NEO4J_URI
NEO4J_AUTH = env.NEO4J_AUTH

_GRAPH_CANVAS_HEIGHT_TRIM_PX = 18

# Main chrome layout (center column only; kept in app.py to avoid deploy/import drift)
MAIN_CHROME_HEADER_HEIGHT_VH = 10
MAIN_CHROME_FOOTER_HEIGHT_VH = 10
MAIN_CHROME_STREAMLIT_HEADER_OFFSET_REM = 3.75
MAIN_CHROME_SIDEBAR_WIDTH_REM = 21
MAIN_CHROME_SIDEBAR_COLLAPSED_WIDTH_REM = 2.875
MAIN_CHROME_BACKGROUND_COLOR = "#f8fafc"
MAIN_CHROME_BORDER_COLOR = "#e2e8f0"
MAIN_CHROME_TAGLINE = "AI Lineage & Compliance Knowledge Graph"
MAIN_CHROME_FOOTER_DISCLAIMER = (
    "Compliance guidance is indicative — verify license terms before deployment."
)
MAIN_CHROME_FOOTER_DATA_SOURCES = "Data: Neo4j graph · Hugging Face · Open LLM Leaderboard"
MAIN_CONTENT_BASE_FONT_REM = 0.875
DETAIL_MODEL_BAR_HEIGHT_REM = 2.25

st.set_page_config(layout="wide", page_title="ModelRoot")


def inject_app_styles(show_detail_model_bar=False):
    detail_model_bar_offset = (
        f" + {DETAIL_MODEL_BAR_HEIGHT_REM}rem" if show_detail_model_bar else ""
    )
    main_chrome_top_offset = (
        f"calc({MAIN_CHROME_HEADER_HEIGHT_VH}vh + "
        f"{MAIN_CHROME_STREAMLIT_HEADER_OFFSET_REM}rem{detail_model_bar_offset})"
    )
    st.markdown(
        f"""
        <style>
        section[data-testid="stMain"] .block-container {{
            padding-top: {main_chrome_top_offset} !important;
            padding-bottom: calc({MAIN_CHROME_FOOTER_HEIGHT_VH}vh + 1rem) !important;
            max-width: 100%;
            font-size: {MAIN_CONTENT_BASE_FONT_REM}rem;
        }}
        .modelroot-main-header,
        .modelroot-main-footer {{
            position: fixed;
            z-index: 998;
            box-sizing: border-box;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.5rem 1.25rem;
            background: {MAIN_CHROME_BACKGROUND_COLOR};
            color: #334155;
            font-size: 0.8rem;
            line-height: 1.35;
            overflow: hidden;
            left: {MAIN_CHROME_SIDEBAR_WIDTH_REM}rem;
            right: 0;
        }}
        .modelroot-main-header {{
            top: {MAIN_CHROME_STREAMLIT_HEADER_OFFSET_REM}rem;
            height: {MAIN_CHROME_HEADER_HEIGHT_VH}vh;
            min-height: 3rem;
            border-bottom: 1px solid {MAIN_CHROME_BORDER_COLOR};
        }}
        .modelroot-main-footer {{
            bottom: 0;
            height: {MAIN_CHROME_FOOTER_HEIGHT_VH}vh;
            min-height: 3rem;
            border-top: 1px solid {MAIN_CHROME_BORDER_COLOR};
        }}
        .modelroot-detail-model-bar {{
            position: fixed;
            z-index: 997;
            box-sizing: border-box;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.35rem 1.25rem;
            background: #eef2ff;
            border-bottom: 1px solid {MAIN_CHROME_BORDER_COLOR};
            color: #1e3a8a;
            font-size: 0.82rem;
            left: {MAIN_CHROME_SIDEBAR_WIDTH_REM}rem;
            right: 0;
            top: calc({MAIN_CHROME_STREAMLIT_HEADER_OFFSET_REM}rem + {MAIN_CHROME_HEADER_HEIGHT_VH}vh);
            height: {DETAIL_MODEL_BAR_HEIGHT_REM}rem;
            overflow: hidden;
        }}
        .modelroot-detail-model-bar-label {{
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 0.72rem;
            color: #475569;
            white-space: nowrap;
        }}
        .modelroot-detail-model-bar-name {{
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .modelroot-main-header-title {{
            font-weight: 700;
            font-size: 0.875rem;
            color: #0f172a;
            white-space: nowrap;
        }}
        .modelroot-main-header-mode {{
            font-weight: 600;
            color: #1e40af;
            white-space: nowrap;
        }}
        .modelroot-main-header-meta {{
            color: #64748b;
            text-align: right;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .modelroot-main-footer-text {{
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        [data-testid="stAppViewContainer"]:has(
            section[data-testid="stSidebar"][aria-expanded="false"]
        ) .modelroot-main-header,
        [data-testid="stAppViewContainer"]:has(
            section[data-testid="stSidebar"][aria-expanded="false"]
        ) .modelroot-main-footer,
        [data-testid="stAppViewContainer"]:has(
            section[data-testid="stSidebar"][aria-expanded="false"]
        ) .modelroot-detail-model-bar {{
            left: {MAIN_CHROME_SIDEBAR_COLLAPSED_WIDTH_REM}rem;
        }}
        @media (max-width: 768px) {{
            .modelroot-main-header,
            .modelroot-main-footer,
            .modelroot-detail-model-bar {{
                left: 0;
            }}
        }}
        section[data-testid="stSidebar"] h5 {{
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #334155;
            line-height: 1.5 !important;
            padding-top: 0.25rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e2e8f0;
            margin-bottom: 0.75rem;
        }}
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] h1 {{
            font-size: 1.35rem;
        }}
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] h2,
        section[data-testid="stMain"] h2 {{
            font-size: 1.1rem;
        }}
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] h3,
        section[data-testid="stMain"] h3 {{
            margin-top: 0.25rem;
            line-height: 1.35;
            font-size: 1rem;
        }}
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] h4 {{
            font-size: 0.95rem;
        }}
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] h5 {{
            font-size: 0.9rem;
        }}
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] h6 {{
            font-size: 0.85rem;
            font-weight: 600;
        }}
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stMain"] [data-testid="stMarkdownContainer"] li {{
            font-size: {MAIN_CONTENT_BASE_FONT_REM}rem;
        }}
        section[data-testid="stMain"] [data-testid="stMetricLabel"] {{
            font-size: 0.78rem;
        }}
        section[data-testid="stMain"] [data-testid="stMetricValue"] {{
            font-size: 1.05rem;
        }}
        section[data-testid="stMain"] [data-testid="stCaptionContainer"] {{
            font-size: 0.75rem;
        }}
        section[data-testid="stMain"] label,
        section[data-testid="stMain"] [data-testid="stWidgetLabel"] p {{
            font-size: 0.8rem !important;
        }}
        section[data-testid="stMain"] input,
        section[data-testid="stMain"] textarea,
        section[data-testid="stMain"] [data-baseweb="select"] {{
            font-size: 0.82rem !important;
        }}
        section[data-testid="stMain"] button {{
            font-size: 0.82rem;
        }}
        section[data-testid="stMain"] div[data-testid="stDataFrame"] {{
            font-size: 0.75rem;
        }}
        .modelroot-benchmark-suite {{
            border: 2px solid {BENCHMARK_PANEL_BORDER_COLOR};
            border-radius: 0.5rem;
            background: {BENCHMARK_PANEL_BACKGROUND_COLOR};
            padding: 0.5rem 0.75rem;
            width: fit-content;
            max-width: 100%;
            overflow: visible;
        }}
        .modelroot-benchmark-suite table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
        }}
        .modelroot-benchmark-suite th,
        .modelroot-benchmark-suite td {{
            padding: 0.35rem 0.9rem;
            text-align: left;
            white-space: nowrap;
        }}
        .modelroot-benchmark-suite thead th {{
            border-bottom: 1px solid #cbd5e1;
            color: #334155;
        }}
        div[data-testid="stButton"] button[kind="primary"] {{
            background-color: {BENCHMARK_PANEL_BORDER_COLOR};
            border-color: {BENCHMARK_PANEL_BORDER_COLOR};
            color: #ffffff;
        }}
        div[data-testid="stButton"] button[kind="primary"]:hover {{
            background-color: #1d4ed8;
            border-color: #1d4ed8;
            color: #ffffff;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_database_snapshot_summary():
    snapshot_rows = run_query(DATABASE_SNAPSHOT_CYPHER)
    if not snapshot_rows:
        return None
    return snapshot_rows[0]


def render_main_header(mode, snapshot_summary):
    snapshot_meta_text = ""
    if snapshot_summary:
        snapshot_meta_text = (
            f"{snapshot_summary['models']:,} models · "
            f"{snapshot_summary['licenses']:,} licenses · "
            f"{snapshot_summary['groups']} risk groups"
        )
    st.markdown(
        f"""
        <div class="modelroot-main-header">
          <div>
            <span class="modelroot-main-header-title">ModelRoot</span>
            <span> — {MAIN_CHROME_TAGLINE}</span>
          </div>
          <span class="modelroot-main-header-mode">{mode}</span>
          <span class="modelroot-main-header-meta">{snapshot_meta_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_detail_model_bar(model_name):
    escaped_model_name = html.escape(model_name)
    st.markdown(
        f"""
        <div class="modelroot-detail-model-bar">
          <span class="modelroot-detail-model-bar-label">Selected model</span>
          <span class="modelroot-detail-model-bar-name">{escaped_model_name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clear_detail_search_session_state():
    for session_key in (
        "detail_model_find",
        "detail_model_active_search",
        "detail_model_disambiguation",
    ):
        st.session_state.pop(session_key, None)


def render_main_footer():
    st.markdown(
        f"""
        <div class="modelroot-main-footer">
          <span class="modelroot-main-footer-text">{MAIN_CHROME_FOOTER_DISCLAIMER}</span>
          <span class="modelroot-main-footer-text">{MAIN_CHROME_FOOTER_DATA_SOURCES}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


def run_query(query, params=None):
    with get_driver().session() as session:
        return [dict(record) for record in session.run(query, params or {})]


@st.cache_data
def get_license_group_metadata():
    return run_query(LICENSE_GROUP_METADATA_CYPHER)


def get_license_groups():
    return [group["id"] for group in get_license_group_metadata()]


@st.cache_data
def get_tasks():
    return [r["task"] for r in run_query(DISTINCT_TASKS_CYPHER)]


@st.cache_data
def get_brands():
    return [r["brand"] for r in run_query(DISTINCT_BRANDS_CYPHER)]


def get_license_group_color(license_group_id):
    return LICENSE_GROUP_COLORS.get(license_group_id, UNKNOWN_LICENSE_GROUP_COLOR)


def build_license_group_cell_style(license_group_id):
    return f"background-color: {get_license_group_color(license_group_id)}; {LICENSE_GROUP_CELL_STYLE}"


def render_license_with_risk(license_name, license_group_id, risk_label):
    if not license_group_id or license_group_id == "UNKNOWN":
        risk_display = risk_label or "Unclassified"
        badge_color = UNKNOWN_LICENSE_GROUP_COLOR
    else:
        risk_display = risk_label or license_group_id
        badge_color = get_license_group_color(license_group_id)
    license_risk_text = f"{license_name} ({risk_display})"
    return (
        f'<span style="background:{badge_color};'
        f'{LICENSE_GROUP_CELL_STYLE};padding:2px 8px;border-radius:4px;">{license_risk_text}</span>'
    )


def model_name_matches_hf_token_search(model_name, search_text):
    query_text = search_text.strip().lower()
    if not query_text:
        return True
    model_name_lower = model_name.lower()
    for search_term in query_text.split():
        token_pattern = rf"(?:^|[/\-_.]){re.escape(search_term)}(?:[/\-_.]|\d|$)"
        if not re.search(token_pattern, model_name_lower):
            return False
    return True


def build_hf_token_search_regex(search_text):
    search_terms = search_text.strip().split()
    if not search_terms:
        return None
    term_patterns = []
    for search_term in search_terms:
        escaped_term = re.escape(search_term)
        term_patterns.append(
            rf"(?:^|[/\-_.]){escaped_term}(?:[/\-_.]|\d|$)"
        )
    combined_pattern = ".*".join(term_patterns)
    return f"(?i).*{combined_pattern}.*"


def load_model_catalog(search, license_groups, tasks, brands, in_benchmark, limit):
    return run_query(MODEL_CATALOG_CYPHER, {
        "search_pattern": build_hf_token_search_regex(search),
        "license_groups": license_groups,
        "tasks": tasks,
        "brands": brands,
        "in_benchmark": in_benchmark,
        "limit": limit,
    })


def load_model_detail(model_name):
    rows = run_query(MODEL_DETAIL_CYPHER, {"model_name": model_name})
    if not rows:
        return None
    detail = rows[0]
    detail["tech_domain_names"] = resolve_tech_domain_names_for_task(
        detail.get("task"),
        detail.get("tech_domain_names"),
    )
    return detail


def model_has_benchmark(detail):
    return detail.get("oll_rank") is not None


def render_benchmark_section(detail, model_name):
    if not model_has_benchmark(detail):
        return

    benchmark_panel_session_key = f"benchmark_panel_open_{model_name}"
    if st.button("Benchmark", type="primary", key=f"benchmark_toggle_{model_name}"):
        st.session_state[benchmark_panel_session_key] = not st.session_state.get(
            benchmark_panel_session_key,
            False,
        )

    if not st.session_state.get(benchmark_panel_session_key, False):
        return

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Rank", f"#{int(detail['oll_rank'])}")
    with metric_col2:
        st.metric("Average", f"{detail['oll_average']:.2f}")
    with metric_col3:
        params_b = detail.get("oll_params_b")
        st.metric("Params (B)", f"{params_b:.2f}" if params_b is not None else "—")

    benchmark_suite_rows = []
    for field_key, suite_name in (
        ("oll_ifeval", "IFEval"),
        ("oll_bbh", "BBH"),
        ("oll_math_lvl5", "MATH Lvl 5"),
        ("oll_gpqa", "GPQA"),
        ("oll_musr", "MUSR"),
        ("oll_mmlu_pro", "MMLU-PRO"),
    ):
        score = detail.get(field_key)
        if score is not None:
            benchmark_suite_rows.append({
                "Suite": suite_name,
                "Score": round(float(score), 2),
            })

    if benchmark_suite_rows:
        benchmark_table_rows_html = "".join(
            (
                f"<tr><td>{benchmark_row['Suite']}</td>"
                f"<td>{benchmark_row['Score']}</td></tr>"
            )
            for benchmark_row in benchmark_suite_rows
        )
        st.markdown(
            f"""
            <div class="modelroot-benchmark-suite">
              <table>
                <thead>
                  <tr><th>Suite</th><th>Score</th></tr>
                </thead>
                <tbody>{benchmark_table_rows_html}</tbody>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    submission_date = detail.get("oll_submission_date")
    if submission_date:
        st.caption(f"Evaluated on: {submission_date}")


def format_task_with_tech_domain(task_name, tech_domain_names):
    tech_domain_label = ", ".join(tech_domain_names) if tech_domain_names else None
    if tech_domain_label and task_name and task_name != "—":
        return f"{tech_domain_label} · {task_name}"
    if tech_domain_label:
        return tech_domain_label
    return task_name or "—"


@st.cache_data
def load_all_model_names_by_downloads():
    return [
        row["model"]
        for row in run_query(ALL_MODEL_NAMES_BY_DOWNLOADS_CYPHER)
    ]


@st.cache_data
def load_top_model_names_by_downloads(limit):
    return [
        row["model"]
        for row in run_query(MODEL_PICKER_BROWSE_CYPHER, {"limit": limit})
    ]


@st.cache_data
def load_model_names_by_hf_token_search(search_text, limit):
    search_pattern = build_hf_token_search_regex(search_text)
    if search_pattern is None:
        return []
    return [
        row["model"]
        for row in run_query(
            MODEL_PICKER_SEARCH_CYPHER,
            {"search_pattern": search_pattern, "limit": limit},
        )
    ]


def build_model_picker_options(session_key):
    model_options = list(load_all_model_names_by_downloads())
    current_model = st.session_state.get(session_key)
    if current_model and current_model not in model_options:
        model_options.insert(0, current_model)
    return model_options


def filter_models_by_hf_token_search(model_options, search_text):
    if not search_text.strip():
        return model_options[:MODEL_PICKER_BROWSE_LIMIT]
    return [
        model_name
        for model_name in model_options
        if model_name_matches_hf_token_search(model_name, search_text)
    ][:MODEL_PICKER_MATCH_LIMIT]


def ensure_current_model_in_picker_options(picker_options, session_key):
    current_model = st.session_state.get(session_key)
    if current_model and current_model not in picker_options:
        return [current_model, *picker_options]
    return picker_options


def sort_detail_search_matches_with_exact_first(find_query, search_matches):
    query_lower = find_query.strip().lower()
    exact_matches = [
        model_name for model_name in search_matches if model_name.lower() == query_lower
    ]
    other_matches = [
        model_name for model_name in search_matches if model_name.lower() != query_lower
    ]
    return exact_matches + other_matches


def pick_best_detail_search_match(find_query, search_matches):
    sorted_matches = sort_detail_search_matches_with_exact_first(find_query, search_matches)
    return sorted_matches[0]


def detail_model_disambiguation_session_key(session_key):
    return f"{session_key}_disambiguation"


def render_detail_model_search_picker(session_key):
    find_query = st.text_input(
        "Find model",
        placeholder="e.g. gpt2, openai-community/gpt2, qwen",
        key=f"{session_key}_find",
        help=(
            "HF token match on / - _ . boundaries. Shows all matching versions in the database; "
            "a full org/model name is pre-selected when it exists."
        ),
    )
    find_query_normalized = find_query.strip()
    active_search_session_key = f"{session_key}_active_search"
    disambiguation_session_key = detail_model_disambiguation_session_key(session_key)

    if find_query_normalized:
        raw_search_matches = load_model_names_by_hf_token_search(
            find_query_normalized,
            MODEL_PICKER_MATCH_LIMIT,
        )
        search_matches = sort_detail_search_matches_with_exact_first(
            find_query_normalized,
            raw_search_matches,
        )
        if not search_matches:
            st.warning(f"No model token matches '{find_query_normalized}'.")
            return st.session_state.get(session_key)

        if st.session_state.get(active_search_session_key) != find_query_normalized:
            st.session_state[active_search_session_key] = find_query_normalized
            best_match_model = pick_best_detail_search_match(
                find_query_normalized,
                search_matches,
            )
            st.session_state[disambiguation_session_key] = best_match_model

        if st.session_state.get(disambiguation_session_key) not in search_matches:
            st.session_state[disambiguation_session_key] = search_matches[0]

        if len(search_matches) > 1:
            st.caption(f"{len(search_matches)} matches — pick one to update graph and tables.")
            st.radio(
                "Matching models",
                search_matches,
                format_func=lambda model_name: model_name,
                key=disambiguation_session_key,
            )

        selected_model = st.session_state[disambiguation_session_key]
        st.session_state[session_key] = selected_model
        return selected_model

    st.session_state.pop(active_search_session_key, None)
    st.session_state.pop(disambiguation_session_key, None)
    selected_model = st.session_state.get(session_key)
    if not selected_model:
        browse_default_models = load_top_model_names_by_downloads(1)
        if not browse_default_models:
            st.error("No models found.")
            return None
        selected_model = browse_default_models[0]
        st.session_state[session_key] = selected_model

    st.caption("Type above to search any model in the database.")
    return selected_model


def render_model_picker(session_key, search_full_database=False):
    if search_full_database:
        return render_detail_model_search_picker(session_key)

    find_query = st.text_input(
        "Find model",
        placeholder="e.g. bert, google bert, org/model",
        key=f"{session_key}_find",
        help="HF token match on / - _ . boundaries (google-bert matches bert; roberta does not).",
    )

    model_options = build_model_picker_options(session_key)
    if not model_options:
        st.error("No models found.")
        return None
    picker_options = filter_models_by_hf_token_search(model_options, find_query)
    browse_fallback_options = model_options[:MODEL_PICKER_BROWSE_LIMIT]

    if find_query.strip() and not picker_options:
        st.warning(f"No model token matches '{find_query.strip()}'.")
        picker_options = browse_fallback_options

    picker_options = ensure_current_model_in_picker_options(picker_options, session_key)
    if not picker_options:
        st.error("No models found.")
        return None

    if st.session_state.get(session_key) not in picker_options:
        st.session_state[session_key] = picker_options[0]

    picker_label = (
        f"Matching models ({len(picker_options)})"
        if find_query.strip()
        else "Top models by downloads"
    )
    st.selectbox(picker_label, picker_options, key=session_key)
    return st.session_state[session_key]


def load_derived_models(model_name, limit):
    return run_query(
        DERIVED_MODELS_CYPHER,
        {"model_name": model_name, "limit": limit},
    )


def load_model_neighborhood(model_name, limit):
    return run_query(
        MODEL_NEIGHBORHOOD_CYPHER,
        {"model_name": model_name, "limit": limit},
    )


def normalize_optional_url(url_value):
    if url_value is None or pd.isna(url_value):
        return pd.NA
    url_text = str(url_value).strip()
    if not url_text or url_text.lower() in {"none", "nan"}:
        return pd.NA
    return url_text


def prepare_dataframe_link_columns(dataframe, link_column_names):
    prepared_dataframe = dataframe.copy()
    for link_column_name in link_column_names:
        if link_column_name in prepared_dataframe.columns:
            prepared_dataframe[link_column_name] = prepared_dataframe[link_column_name].map(
                normalize_optional_url
            )
    return prepared_dataframe


def get_hf_link_column():
    return st.column_config.LinkColumn("HF link", display_text=LINK_COLUMN_DISPLAY_TEXT, width="small")


def get_license_link_column():
    return st.column_config.LinkColumn(
        "Lic. link",
        display_text=LINK_COLUMN_DISPLAY_TEXT,
        width="small",
    )


def resolve_license_documentation_url(license_name):
    normalized_license_name = license_name.strip().lower()
    if normalized_license_name in {"", "unknown", "other", "unknown / unspecified"}:
        return None
    if normalized_license_name in LICENSE_OFFICIAL_DOCUMENT_URLS:
        return LICENSE_OFFICIAL_DOCUMENT_URLS[normalized_license_name]
    if normalized_license_name in LICENSE_WIKIPEDIA_PAGES:
        return f"https://en.wikipedia.org/wiki/{LICENSE_WIKIPEDIA_PAGES[normalized_license_name]}"
    return (
        "https://en.wikipedia.org/wiki/Special:Search?search="
        f"{quote(normalized_license_name + ' software license')}"
    )


def build_neighborhood_link_cell(entity_label, entity_kind, entity_url, placeholder_link_base):
    if entity_url is not pd.NA:
        return f"{entity_url}#{entity_label}"
    if entity_kind == "Publisher":
        brand_hf_url = BRAND_HF_URLS.get(entity_label)
        if brand_hf_url:
            return f"{brand_hf_url}#{entity_label}"
        return f"{placeholder_link_base}{entity_label}"
    if entity_kind == "Dataset":
        dataset_documentation_url = DATASET_DOCUMENTATION_URLS.get(entity_label)
        if dataset_documentation_url:
            return f"{dataset_documentation_url}#{entity_label}"
        return f"{placeholder_link_base}{entity_label}"
    if entity_kind == "License":
        license_document_url = resolve_license_documentation_url(entity_label)
        if license_document_url:
            return f"{license_document_url}#{entity_label}"
    return pd.NA


def render_model_neighborhood_table(neighborhood_edges, data_view_height):
    neighborhood_rows = []
    for neighborhood_edge in neighborhood_edges:
        relation = neighborhood_edge["relation"]
        entity_type = neighborhood_edge["entity_type"]
        entity_label = neighborhood_edge["entity"]
        entity_kind = (
            NEIGHBORHOOD_KIND_LABELS.get(relation)
            or NEIGHBORHOOD_KIND_LABELS.get(entity_type)
            or relation.replace("_", " ").title()
        )
        entity_url = normalize_optional_url(neighborhood_edge.get("entity_url"))
        entity_link = build_neighborhood_link_cell(
            entity_label,
            entity_kind,
            entity_url,
            NEIGHBORHOOD_PLACEHOLDER_LINK_BASE,
        )
        neighborhood_rows.append({
            "Kind": entity_kind,
            "Link": entity_link,
        })
    render_scrollable_dataframe(
        pd.DataFrame(neighborhood_rows),
        data_view_height,
        column_config={
            "Kind": st.column_config.TextColumn("Kind", width="small"),
            "Link": st.column_config.LinkColumn(
                "Link",
                display_text=NEIGHBORHOOD_LINK_DISPLAY_REGEX,
                width="large",
            ),
        },
        use_container_width=True,
    )


def format_license_group_filter_label(group_id, license_group_metadata_by_id):
    if group_id == "UNKNOWN":
        return "UNKNOWN — No risk group assigned"
    group_metadata = license_group_metadata_by_id.get(group_id)
    if not group_metadata:
        return group_id
    return f"{group_id} — {group_metadata['name']}"


def render_scrollable_dataframe(dataframe, data_view_height, **dataframe_kwargs):
    dataframe_kwargs.setdefault("placeholder", DATAFRAME_PLACEHOLDER)
    st.dataframe(dataframe, height=data_view_height, **dataframe_kwargs)


def render_license_group_legend(data_view_height):
    license_group_rows = get_license_group_metadata()
    with st.expander("License risk levels reference", expanded=False):
        render_scrollable_dataframe(
            pd.DataFrame([
                {
                    "Risk level": group["name"],
                    "Compliance guidance": group["compliance"],
                }
                for group in license_group_rows
            ] + [UNCLASSIFIED_LICENSE_GROUP_LEGEND]),
            min(data_view_height, 220),
            hide_index=True,
            use_container_width=True,
        )


def build_catalog_dataframe(rows):
    catalog_dataframe = pd.DataFrame(rows)
    return prepare_dataframe_link_columns(catalog_dataframe, CATALOG_LINK_COLUMNS)


def get_catalog_column_config():
    return {
        "model": st.column_config.TextColumn("Model", width="large"),
        "brand": st.column_config.TextColumn("Brand", width="small"),
        "task": st.column_config.TextColumn("Task", width="small"),
        "benchmark": st.column_config.TextColumn("Benchmark", width="small"),
        "license": st.column_config.TextColumn("License", width="small"),
        "risk_level": st.column_config.TextColumn("Risk", width="medium"),
        "downloads": st.column_config.NumberColumn("DL", format="%d", width="small"),
        "hf_url": get_hf_link_column(),
        "license_link": get_license_link_column(),
    }


def build_styled_risk_rows_dataframe(risk_dataframe, display_columns, styled_columns):
    license_group_ids = risk_dataframe["license_group"]
    display_dataframe = risk_dataframe[display_columns]

    def style_risk_row(row):
        row_styles = [""] * len(row)
        cell_style = build_license_group_cell_style(license_group_ids.loc[row.name])
        for column_name in styled_columns:
            row_styles[row.index.get_loc(column_name)] = cell_style
        return row_styles

    return display_dataframe.style.apply(style_risk_row, axis=1)


def build_styled_catalog_dataframe(catalog_dataframe):
    return build_styled_risk_rows_dataframe(
        catalog_dataframe,
        CATALOG_DISPLAY_COLUMNS,
        CATALOG_STYLED_COLUMNS,
    )


def load_top_models_by_risk(limit=30):
    return run_query(TOP_MODELS_BY_RISK_CYPHER, {"limit": limit})


def build_graph_canvas_height(data_view_height):
    return max(
        DATA_VIEW_HEIGHT_MIN - _GRAPH_CANVAS_HEIGHT_TRIM_PX,
        data_view_height - _GRAPH_CANVAS_HEIGHT_TRIM_PX,
    )


def render_pyvis_graph_html(net, data_view_height):
    graph_overflow_fix_css = """
        <style>
        body { margin: 0; overflow: hidden; }
        .card { border: none !important; margin: 0 !important; }
        .card-body { padding: 0 !important; }
        </style>
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as graph_file:
        graph_path = graph_file.name
    net.save_graph(graph_path)
    with open(graph_path, "r", encoding="utf-8") as graph_file:
        graph_html = graph_file.read().replace("</head>", f"{graph_overflow_fix_css}</head>")
    os.remove(graph_path)
    components.html(graph_html, height=data_view_height, scrolling=False)


def render_mini_graph(center_model, edges, data_view_height):
    graph_canvas_height = build_graph_canvas_height(data_view_height)
    net = Network(
        height=f"{graph_canvas_height}px",
        width="100%",
        bgcolor=GRAPH_BACKGROUND_COLOR,
        directed=True,
    )
    net.set_options(json.dumps(MINI_GRAPH_OPTIONS))
    net.add_node(center_model, label=center_model, color=CENTER_MODEL_NODE_COLOR, size=22)

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        relation = edge["relation"]
        color = RELATION_COLORS.get(relation, DEFAULT_RELATION_COLOR)
        if source != center_model:
            net.add_node(source, label=source, color=color, size=14)
        if target != center_model:
            net.add_node(target, label=target, color=color, size=14)
        net.add_edge(source, target, label=relation.replace("_", " "), color=color, arrows="to")

    render_pyvis_graph_html(net, data_view_height)


def render_catalog_page(query_limit, data_view_height):
    st.markdown("##### Model Catalog")
    st.caption("Search and filter models by license risk, task, and publisher.")
    render_license_group_legend(data_view_height)

    license_group_metadata_by_id = {
        group["id"]: group for group in get_license_group_metadata()
    }
    group_options = get_license_groups() + ["UNKNOWN"]

    search_col, risk_col, task_col, brand_col = st.columns(4)
    with search_col:
        search = st.text_input(
            "Search model",
            placeholder="e.g. bert, google bert",
            help="Token match on HF name parts (/ - _ .). Combine with Brand to narrow further.",
        )
    with risk_col:
        selected_groups = st.multiselect(
            "Risk level",
            group_options,
            default=[],
            format_func=lambda group_id: format_license_group_filter_label(
                group_id, license_group_metadata_by_id
            ),
        )
    with task_col:
        selected_tasks = st.multiselect("Task", get_tasks())
    with brand_col:
        selected_brands = st.multiselect("Brand", get_brands())

    in_benchmark = st.checkbox(
        "Benchmark only",
        value=False,
        help="Show only models with benchmark scores synced in ModelRoot (oll_rank on Model).",
    )

    catalog_filter_signature = (
        search.strip(),
        tuple(sorted(selected_groups)),
        tuple(sorted(selected_tasks)),
        tuple(sorted(selected_brands)),
        in_benchmark,
    )
    if st.session_state.get("catalog_filter_signature") != catalog_filter_signature:
        st.session_state["catalog_filter_signature"] = catalog_filter_signature

    rows = load_model_catalog(
        search,
        selected_groups,
        selected_tasks,
        selected_brands,
        in_benchmark,
        query_limit,
    )
    if not rows:
        st.warning("No models match the current filters.")
        return

    df = build_catalog_dataframe(rows)
    model_names = df["model"].tolist()

    st.markdown(
        f"**{len(df)}** models — filter above, select a table row, then open detail. "
        "Full compliance guidance is shown in Model Detail."
    )

    table_selection = st.dataframe(
        build_styled_catalog_dataframe(df),
        column_config=get_catalog_column_config(),
        hide_index=True,
        use_container_width=True,
        height=data_view_height,
        placeholder=DATAFRAME_PLACEHOLDER,
        on_select="rerun",
        selection_mode="single-row",
        key="catalog_table",
    )

    selected_model = None
    if table_selection.selection.rows:
        selected_model = df.iloc[table_selection.selection.rows[0]]["model"]
        st.session_state["detail_model"] = selected_model
    elif st.session_state.get("detail_model") in model_names:
        selected_model = st.session_state["detail_model"]

    if st.button("View detail", type="primary", disabled=selected_model is None):
        st.session_state["detail_model"] = selected_model
        st.session_state["pending_mode"] = "Model Detail"
        clear_detail_search_session_state()
        st.rerun()


def render_detail_page_body(model_name, query_limit, data_view_height):
    if model_name is None:
        return

    detail = load_model_detail(model_name)
    if not detail:
        st.error("Model not found.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Downloads", f"{detail['downloads']:,}")
    with c2:
        st.write("Brand", detail["brand"])

    st.markdown("**License**")
    st.markdown(
        render_license_with_risk(
            detail["license"],
            detail["license_group"],
            detail.get("license_group_name"),
        ),
        unsafe_allow_html=True,
    )
    if detail["license_group"] == "UNKNOWN":
        st.caption("This license is not mapped to a risk group yet.")
    elif detail.get("license_group_compliance"):
        st.caption(detail["license_group_compliance"])

    st.write(
        "Task",
        format_task_with_tech_domain(detail["task"], detail["tech_domain_names"]),
    )

    link_col1, link_col2 = st.columns(2)
    with link_col1:
        if detail.get("hf_url"):
            st.link_button("Open on Hugging Face", detail["hf_url"])
        else:
            st.link_button("Open on Hugging Face", f"https://huggingface.co/{detail['model']}")
    with link_col2:
        if detail.get("license_link"):
            st.link_button("Open license document", detail["license_link"])

    show_derived_models = st.toggle(
        "Show derived",
        value=False,
        help="Shows GGUF/AWQ/fork models (DERIVED_FROM) in the graph and in the derived table below.",
    )

    neighborhood_edges = load_model_neighborhood(model_name, query_limit)
    if not show_derived_models:
        neighborhood_edges = [
            neighborhood_edge
            for neighborhood_edge in neighborhood_edges
            if not (
                neighborhood_edge["relation"] == "DERIVED_FROM"
                and neighborhood_edge["direction"] == "in"
                and neighborhood_edge["entity_type"] == "Model"
            )
        ]
    neighborhood_edges = [
        neighborhood_edge
        for neighborhood_edge in neighborhood_edges
        if neighborhood_edge["relation"] != "PERFORMS"
    ]
    if neighborhood_edges:
        render_mini_graph(model_name, neighborhood_edges, data_view_height)

    render_benchmark_section(detail, model_name)

    if show_derived_models:
        derived_model_rows = load_derived_models(model_name, query_limit)
        if derived_model_rows:
            derived_models_dataframe = prepare_dataframe_link_columns(
                pd.DataFrame(derived_model_rows),
                ["hf_url"],
            )
            st.subheader("Derived models")
            render_scrollable_dataframe(
                derived_models_dataframe,
                data_view_height,
                column_config={
                    "model": st.column_config.TextColumn("Model", width="large"),
                    "downloads": st.column_config.NumberColumn("Downloads", format="%d"),
                    "license": st.column_config.TextColumn("License", width="small"),
                    "hf_url": get_hf_link_column(),
                },
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No derived models found for this base model.")

    st.subheader("Related entities (1-hop)")
    st.caption(f"Center: {model_name}")
    if neighborhood_edges:
        render_model_neighborhood_table(neighborhood_edges, data_view_height)
    else:
        st.info("No direct relationships found for this model.")


def render_license_intelligence_page(query_limit, data_view_height):
    st.subheader("License Intelligence")

    group_rows = run_query(MODELS_BY_LICENSE_GROUP_CYPHER)
    if group_rows:
        df_groups = pd.DataFrame(group_rows)
        st.subheader("Models by license group")
        st.bar_chart(df_groups.set_index("group_id"))

    license_rows = run_query(TOP_LICENSES_CYPHER, {"limit": query_limit})
    if license_rows:
        st.subheader("Top licenses")
        st.bar_chart(pd.DataFrame(license_rows).set_index("license"))

    top_models_by_risk = load_top_models_by_risk(limit=query_limit)
    if top_models_by_risk:
        st.subheader("Top models by license risk")
        top_models_dataframe = prepare_dataframe_link_columns(
            pd.DataFrame(top_models_by_risk),
            ["hf_url"],
        )
        render_scrollable_dataframe(
            build_styled_risk_rows_dataframe(
                top_models_dataframe,
                TOP_MODELS_BY_RISK_DISPLAY_COLUMNS,
                TOP_MODELS_BY_RISK_STYLED_COLUMNS,
            ),
            data_view_height,
            column_config={
                "risk_level": st.column_config.TextColumn("Risk level", width="medium"),
                "risk_guidance": st.column_config.TextColumn("Compliance guidance", width="large"),
                "model": st.column_config.TextColumn("Model", width="large"),
                "brand": st.column_config.TextColumn("Brand"),
                "license": st.column_config.TextColumn("License ID"),
                "downloads": st.column_config.NumberColumn("Downloads", format="%d"),
                "hf_url": get_hf_link_column(),
            },
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Top datasets by usage")
    dataset_rows = run_query(TOP_DATASETS_BY_USAGE_CYPHER, {"limit": query_limit})
    if dataset_rows:
        df_datasets = pd.DataFrame(dataset_rows)
        st.bar_chart(df_datasets.set_index("dataset"))
        ds_select = st.selectbox("Models using dataset", df_datasets["dataset"].tolist())
        if ds_select:
            model_rows = run_query(
                MODELS_USING_DATASET_CYPHER,
                {"dataset_name": ds_select, "limit": query_limit},
            )
            render_scrollable_dataframe(
                pd.DataFrame(model_rows), data_view_height, use_container_width=True
            )


def render_graph_explorer_page(query_limit, data_view_height):
    st.subheader("Graph Explorer")
    st.caption("Best for small, filtered subgraphs — not the full database.")

    center_model = render_model_picker("detail_model")
    rel_types = [r["rel"] for r in run_query(RELATIONSHIP_TYPES_CYPHER)]
    selected_rels = st.multiselect(
        "Relationships",
        rel_types,
        default=rel_types[:GRAPH_EXPLORER_DEFAULT_REL_LIMIT],
    )

    if center_model and selected_rels:
        results = run_query(
            GRAPH_EXPLORER_SUBGRAPH_CYPHER,
            {"center_model": center_model, "rels": selected_rels, "limit": query_limit},
        )
        
        if results:
            graph_canvas_height = build_graph_canvas_height(data_view_height)
            net = Network(
                height=f"{graph_canvas_height}px",
                width="100%",
                bgcolor=GRAPH_BACKGROUND_COLOR,
                directed=True,
            )
            net.set_options(json.dumps(GRAPH_EXPLORER_PHYSICS_OPTIONS))
            for row in results:
                rel_color = RELATION_COLORS.get(row["relation"], DEFAULT_RELATION_COLOR)
                net.add_node(row["model"], label=row["model"], color=CENTER_MODEL_NODE_COLOR)
                net.add_node(row["target_name"], label=row["target_name"], color=rel_color)
                net.add_edge(row["model"], row["target_name"], label=row["relation"], color=rel_color, arrows="to")
            render_pyvis_graph_html(net, data_view_height)
        else:
            st.warning("No relationships found.")


# --- UI ---
if "mode" not in st.session_state:
    st.session_state["mode"] = "Model Catalog"

if st.session_state.get("pending_mode"):
    pending_mode = st.session_state.pop("pending_mode")
    st.session_state["mode"] = pending_mode
    if pending_mode == "Model Detail":
        clear_detail_search_session_state()

st.sidebar.markdown("##### ModelRoot")
st.sidebar.markdown("##### AI Lineage & Compliance Knowledge Graph. #####") 
 

st.sidebar.radio("Mode", MODE_OPTIONS, key="mode")
mode = st.session_state["mode"]

if "query_limit" not in st.session_state:
    st.session_state["query_limit"] = QUERY_LIMIT_DEFAULT

query_limit = st.sidebar.slider(
    "Max results",
    min_value=QUERY_LIMIT_MIN,
    max_value=QUERY_LIMIT_MAX,
    key="query_limit",
)

if "data_view_height" not in st.session_state:
    st.session_state["data_view_height"] = DATA_VIEW_HEIGHT_DEFAULT

data_view_height = st.sidebar.slider(
    "Table height (px)",
    min_value=DATA_VIEW_HEIGHT_MIN,
    max_value=DATA_VIEW_HEIGHT_MAX,
    key="data_view_height",
    help="Fixed layout height for tables and graphs. Use fullscreen on a table for more space.",
)

database_snapshot_summary = load_database_snapshot_summary()

with st.sidebar.expander("Database snapshot"):
    if database_snapshot_summary:
        st.metric("Models", f"{database_snapshot_summary['models']:,}")
        st.metric("Licenses", f"{database_snapshot_summary['licenses']:,}")
        st.metric("License groups", database_snapshot_summary["groups"])

detail_active_model = None
if mode == "Model Detail":
    detail_active_model = render_detail_model_search_picker("detail_model")

inject_app_styles(show_detail_model_bar=(mode == "Model Detail" and detail_active_model))
render_main_header(mode, database_snapshot_summary)

if mode == "Model Detail":
    if detail_active_model:
        render_detail_model_bar(detail_active_model)
    render_detail_page_body(detail_active_model, query_limit, data_view_height)
elif mode == "Model Catalog":
    render_catalog_page(query_limit, data_view_height)
elif mode == "License Intelligence":
    render_license_intelligence_page(query_limit, data_view_height)
else:
    render_graph_explorer_page(query_limit, data_view_height)

render_main_footer()

