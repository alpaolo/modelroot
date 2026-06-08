"""
ModelRoot Streamlit app — catalog-first UX for license-aware model discovery.

Modes: Model Catalog (default), Model Detail (1-hop graph), License Intelligence, Graph Explorer.
Catalog: filter + table row selection only. Model name search uses HF token boundaries (/ - _ .), not raw substring (bert ≠ roberta).
Detail/Graph Explorer: find field + match list (detail_model).
Sidebar query_limit controls LIMIT on list/graph queries across all modes.
Sidebar data_view_height sets native st.dataframe / graph iframe height in pixels.
Minimal header: ModelRoot label in sidebar; native st.subheader per mode (no emoji).
"""
QUERY_LIMIT_MIN = 25
QUERY_LIMIT_MAX = 500
QUERY_LIMIT_DEFAULT = 100
DATA_VIEW_HEIGHT_MIN = 250
DATA_VIEW_HEIGHT_MAX = 900
DATA_VIEW_HEIGHT_DEFAULT = 450
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source", "scrapers"))
from neo4j_config import NEO4J_AUTH, NEO4J_URI

st.set_page_config(layout="wide", page_title="ModelRoot")


def inject_app_styles():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.25rem;
            padding-bottom: 1.25rem;
        }
        section[data-testid="stSidebar"] h5 {
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
        }
        .main h3 {
            margin-top: 0.25rem;
            line-height: 1.4;
        }
        div[data-testid="stDataFrame"] {
            font-size: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


LICENSE_GROUP_COLORS = {
    "GREEN": "#22c55e",
    "YELLOW": "#eab308",
    "ORANGE": "#f97316",
    "RED_COPYLEFT": "#dd22ce",
    "RED_RESTRICTED": "#ef4444",
}
UNKNOWN_LICENSE_GROUP_COLOR = "#94a3b8"
LICENSE_GROUP_CELL_STYLE = "color: white; font-weight: bold"

RELATION_COLORS = {
    "UNDER_LICENSE": "#6366f1",
    "PERFORMS": "#32CD32",
    "USED_DATASET": "#1C83E1",
    "CITED_IN": "#8A2BE2",
    "DERIVED_FROM": "#f59e0b",
    "PUBLISHED_BY": "#FF9900",
    "BASED_ON_PAPER": "#8B5CF6",
}


@st.cache_resource
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)


def run_query(query, params=None):
    with get_driver().session() as session:
        return [dict(record) for record in session.run(query, params or {})]


@st.cache_data
def get_license_group_metadata():
    return run_query("""
        MATCH (g:LicenseGroup)
        RETURN g.id AS id, g.name AS name, g.compliance AS compliance
        ORDER BY id
    """)


def get_license_groups():
    return [group["id"] for group in get_license_group_metadata()]


@st.cache_data
def get_tasks():
    return [r["task"] for r in run_query(
        "MATCH (m:Model)-[:PERFORMS]->(t:Task) RETURN DISTINCT t.name AS task ORDER BY task"
    )]


@st.cache_data
def get_brands():
    return [r["brand"] for r in run_query(
        "MATCH (m:Model)-[:PUBLISHED_BY]->(b:MainBrand) RETURN DISTINCT b.name AS brand ORDER BY brand"
    )]


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


def load_model_catalog(search, license_groups, tasks, brands, limit):
    query = """
    MATCH (m:Model)
    WHERE ($search_pattern IS NULL OR m.name =~ $search_pattern)
      AND (size($tasks) = 0 OR EXISTS {
        MATCH (m)-[:PERFORMS]->(t:Task)
        WHERE t.name IN $tasks
      })
      AND (size($brands) = 0 OR EXISTS {
        MATCH (m)-[:PUBLISHED_BY]->(b:MainBrand)
        WHERE b.name IN $brands
      })
    OPTIONAL MATCH (m)-[:UNDER_LICENSE]->(l:License)
    OPTIONAL MATCH (l)-[:BELONGS_TO]->(g:LicenseGroup)
    WITH m, l, g
    WHERE size($license_groups) = 0 OR coalesce(g.id, 'UNKNOWN') IN $license_groups
    OPTIONAL MATCH (m)-[:PERFORMS]->(t:Task)
    WITH m, l, g, head(collect(DISTINCT coalesce(t.name, '—'))) AS task
    OPTIONAL MATCH (m)-[:PUBLISHED_BY]->(b:MainBrand)
    WITH m, l, g, task, head(collect(DISTINCT coalesce(b.name, '—'))) AS brand
    RETURN m.name AS model,
           brand,
           task,
           coalesce(l.name, 'unknown') AS license,
           coalesce(g.id, 'UNKNOWN') AS license_group,
           coalesce(g.name, 'Unclassified') AS risk_level,
           coalesce(g.compliance, 'License not mapped to a risk group yet.') AS risk_guidance,
           coalesce(m.downloads, 0) AS downloads,
           m.hf_url AS hf_url,
           m.license_link AS license_link
    ORDER BY downloads DESC
    LIMIT $limit
    """
    return run_query(query, {
        "search_pattern": build_hf_token_search_regex(search),
        "license_groups": license_groups,
        "tasks": tasks,
        "brands": brands,
        "limit": limit,
    })


def load_model_detail(model_name):
    query = """
    MATCH (m:Model {name: $model_name})
    OPTIONAL MATCH (m)-[:UNDER_LICENSE]->(l:License)
    OPTIONAL MATCH (l)-[:BELONGS_TO]->(g:LicenseGroup)
    OPTIONAL MATCH (m)-[:PERFORMS]->(t:Task)
    OPTIONAL MATCH (m)-[:PUBLISHED_BY]->(b:MainBrand)
    RETURN m.name AS model,
           coalesce(l.name, 'unknown') AS license,
           coalesce(g.id, 'UNKNOWN') AS license_group,
           g.name AS license_group_name,
           g.compliance AS license_group_compliance,
           coalesce(t.name, '—') AS task,
           coalesce(b.name, '—') AS brand,
           coalesce(m.downloads, 0) AS downloads,
           m.hf_url AS hf_url,
           m.license_link AS license_link
    """
    rows = run_query(query, {"model_name": model_name})
    return rows[0] if rows else None


@st.cache_data
def load_all_model_names_by_downloads():
    return [
        row["model"]
        for row in run_query(
            "MATCH (m:Model) RETURN m.name AS model ORDER BY m.downloads DESC"
        )
    ]


def build_model_picker_options(session_key):
    model_options = list(load_all_model_names_by_downloads())
    current_model = st.session_state.get(session_key)
    if current_model and current_model not in model_options:
        model_options.insert(0, current_model)
    return model_options


MODEL_PICKER_BROWSE_LIMIT = 50
MODEL_PICKER_MATCH_LIMIT = 50


def filter_models_by_hf_token_search(model_options, search_text):
    if not search_text.strip():
        return model_options[:MODEL_PICKER_BROWSE_LIMIT]
    return [
        model_name
        for model_name in model_options
        if model_name_matches_hf_token_search(model_name, search_text)
    ][:MODEL_PICKER_MATCH_LIMIT]


def render_model_picker(session_key):
    model_options = build_model_picker_options(session_key)
    if not model_options:
        st.error("No models found.")
        return None

    find_query = st.text_input(
        "Find model",
        placeholder="e.g. bert, google bert, org/model",
        key=f"{session_key}_find",
        help="HF token match on / - _ . boundaries (google-bert matches bert; roberta does not).",
    )
    picker_options = filter_models_by_hf_token_search(model_options, find_query)
    if find_query.strip() and not picker_options:
        st.warning(f"No model token matches '{find_query.strip()}'.")
        picker_options = model_options[:MODEL_PICKER_BROWSE_LIMIT]

    if st.session_state.get(session_key) not in picker_options:
        st.session_state[session_key] = picker_options[0]

    picker_label = (
        f"Matching models ({len(picker_options)})"
        if find_query.strip()
        else "Top models by downloads"
    )
    st.selectbox(picker_label, picker_options, key=session_key)
    return st.session_state[session_key]


def load_model_neighborhood(model_name, limit):
    return run_query(
        """
        MATCH (m:Model {name: $model_name})-[r]->(n)
        RETURN type(r) AS relation,
               CASE
                   WHEN labels(n)[0] = 'Paper' THEN n.id
                   ELSE coalesce(n.name, n.id)
               END AS entity,
               labels(n)[0] AS entity_type,
               CASE
                   WHEN labels(n)[0] = 'Paper'
                   THEN coalesce(n.url, 'https://arxiv.org/abs/' + n.id)
                   ELSE null
               END AS entity_url,
               m.name AS source,
               coalesce(n.name, n.id) AS target,
               'out' AS direction
        UNION
        MATCH (n)-[r]->(m:Model {name: $model_name})
        RETURN type(r) AS relation,
               CASE
                   WHEN labels(n)[0] = 'Paper' THEN n.id
                   ELSE coalesce(n.name, n.id)
               END AS entity,
               labels(n)[0] AS entity_type,
               CASE
                   WHEN labels(n)[0] = 'Paper'
                   THEN coalesce(n.url, 'https://arxiv.org/abs/' + n.id)
                   ELSE null
               END AS entity_url,
               coalesce(n.name, n.id) AS source,
               m.name AS target,
               'in' AS direction
        LIMIT $limit
        """,
        {"model_name": model_name, "limit": limit},
    )


LINK_COLUMN_DISPLAY_TEXT = "Open"


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


LICENSE_OFFICIAL_DOCUMENT_URLS = {
    "mit": "https://spdx.org/licenses/MIT.html",
    "modified-mit": "https://spdx.org/licenses/MIT.html",
    "apache-2.0": "https://spdx.org/licenses/Apache-2.0.html",
    "bsd-2-clause": "https://spdx.org/licenses/BSD-2-Clause.html",
    "bsd-3-clause": "https://spdx.org/licenses/BSD-3-Clause.html",
    "gpl-3.0": "https://spdx.org/licenses/GPL-3.0-only.html",
    "agpl-3.0": "https://spdx.org/licenses/AGPL-3.0-only.html",
    "cc0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "cc-by-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "cc-by-sa-3.0": "https://creativecommons.org/licenses/by-sa/3.0/",
    "cc-by-sa-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "cc-by-nc-2.0": "https://creativecommons.org/licenses/by-nc/2.0/",
    "cc-by-nc-3.0": "https://creativecommons.org/licenses/by-nc/3.0/",
    "cc-by-nc-4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
    "cc-by-nc-sa-4.0": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "afl-3.0": "https://spdx.org/licenses/AFL-3.0.html",
    "cdla-permissive-2.0": "https://spdx.org/licenses/CDLA-Permissive-2.0.html",
    "creativeml-openrail-m": "https://huggingface.co/spaces/bigscience/license",
    "openrail": "https://huggingface.co/spaces/bigscience/license",
    "openrail++": "https://huggingface.co/spaces/bigscience/license",
    "bigcode-openrail-m": "https://www.licenses.ai/blog/2022/8/26/naming-convention-of-responsible-ai-licenses",
    "bigscience-bloom-rail-1.0": "https://bigscience.huggingface.co/blog/the-bigscience-rail-license",
    "llama2": "https://ai.meta.com/llama/license/",
    "llama3": "https://ai.meta.com/llama/license/",
    "llama3.1": "https://ai.meta.com/llama/license/",
    "llama3.2": "https://ai.meta.com/llama/license/",
    "llama3.3": "https://ai.meta.com/llama/license/",
    "llama-3-community": "https://ai.meta.com/llama/license/",
    "llama-3.1-community": "https://ai.meta.com/llama/license/",
    "llama-3.2-community": "https://ai.meta.com/llama/license/",
}

LICENSE_WIKIPEDIA_PAGES = {
    "gemma": "Gemma_(language_model)",
    "gemma-terms-of-use": "Gemma_(language_model)",
    "qwen-license": "Qwen",
    "qwen-commercial-license": "Qwen",
    "deepseek-license": "DeepSeek",
    "mistral-community-license": "Mistral_AI",
    "apple-amlr": "Apple_User_License_Agreement",
    "research-only-license": "Proprietary_software",
    "ai-by-nc-1.0": "Creative_Commons_license",
}


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
    if entity_kind in ("Dataset", "Publisher"):
        return f"{placeholder_link_base}{entity_label}"
    if entity_kind == "License":
        license_document_url = resolve_license_documentation_url(entity_label)
        if license_document_url:
            return f"{license_document_url}#{entity_label}"
    return pd.NA


def render_model_neighborhood_table(neighborhood_edges, data_view_height):
    neighborhood_kind_labels = {
        "BASED_ON_PAPER": "Paper",
        "USED_DATASET": "Dataset",
        "PUBLISHED_BY": "Publisher",
        "UNDER_LICENSE": "License",
        "PERFORMS": "Task",
        "DERIVED_FROM": "Model",
        "CITED_IN": "Citation",
        "MainBrand": "Publisher",
        "Organization": "Publisher",
    }
    neighborhood_link_display_regex = r".*#(.*)$"
    neighborhood_placeholder_link_base = "https://modelroot.pending/#"
    neighborhood_rows = []
    for neighborhood_edge in neighborhood_edges:
        relation = neighborhood_edge["relation"]
        entity_type = neighborhood_edge["entity_type"]
        entity_label = neighborhood_edge["entity"]
        entity_kind = (
            neighborhood_kind_labels.get(relation)
            or neighborhood_kind_labels.get(entity_type)
            or relation.replace("_", " ").title()
        )
        entity_url = normalize_optional_url(neighborhood_edge.get("entity_url"))
        entity_link = build_neighborhood_link_cell(
            entity_label,
            entity_kind,
            entity_url,
            neighborhood_placeholder_link_base,
        )
        neighborhood_rows.append({
            "Kind": entity_kind,
            "Entity": entity_label,
            "Link": entity_link,
        })
    render_scrollable_dataframe(
        pd.DataFrame(neighborhood_rows),
        data_view_height,
        column_config={
            "Kind": st.column_config.TextColumn("Kind", width="small"),
            "Entity": st.column_config.TextColumn("Entity", width="large"),
            "Link": st.column_config.LinkColumn(
                "Link",
                display_text=neighborhood_link_display_regex,
                width="small",
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
    dataframe_kwargs.setdefault("placeholder", "")
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
            ] + [{
                "Risk level": "Unclassified",
                "Compliance guidance": "License not mapped to a risk group yet.",
            }]),
            min(data_view_height, 220),
            hide_index=True,
            use_container_width=True,
        )


CATALOG_DISPLAY_COLUMNS = [
    "model",
    "brand",
    "task",
    "license",
    "risk_level",
    "downloads",
    "hf_url",
    "license_link",
]


def build_catalog_dataframe(rows):
    catalog_dataframe = pd.DataFrame(rows)
    return prepare_dataframe_link_columns(catalog_dataframe, ["hf_url", "license_link"])


def get_catalog_column_config():
    return {
        "model": st.column_config.TextColumn("Model", width="large"),
        "brand": st.column_config.TextColumn("Brand", width="small"),
        "task": st.column_config.TextColumn("Task", width="small"),
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


CATALOG_STYLED_COLUMNS = ["model", "license", "risk_level"]


def build_styled_catalog_dataframe(catalog_dataframe):
    return build_styled_risk_rows_dataframe(
        catalog_dataframe,
        CATALOG_DISPLAY_COLUMNS,
        CATALOG_STYLED_COLUMNS,
    )


TOP_MODELS_BY_RISK_DISPLAY_COLUMNS = [
    "risk_level",
    "risk_guidance",
    "model",
    "brand",
    "license",
    "downloads",
    "hf_url",
]
TOP_MODELS_BY_RISK_STYLED_COLUMNS = ["risk_level", "risk_guidance", "model"]


def load_top_models_by_risk(limit=30):
    return run_query("""
        MATCH (m:Model)-[:UNDER_LICENSE]->(l:License)
        OPTIONAL MATCH (l)-[:BELONGS_TO]->(g:LicenseGroup)
        OPTIONAL MATCH (m)-[:PUBLISHED_BY]->(b:MainBrand)
        RETURN coalesce(g.id, 'UNKNOWN') AS license_group,
               coalesce(g.name, 'Unclassified') AS risk_level,
               coalesce(g.compliance, 'License not mapped to a risk group yet.') AS risk_guidance,
               m.name AS model,
               coalesce(b.name, '—') AS brand,
               coalesce(l.name, 'unknown') AS license,
               coalesce(m.downloads, 0) AS downloads,
               m.hf_url AS hf_url
        ORDER BY m.downloads DESC
        LIMIT $limit
    """, {"limit": limit})


def render_mini_graph(center_model, edges, data_view_height):
    net = Network(height=f"{data_view_height}px", width="100%", bgcolor="#ffffff", directed=True)
    options = {
        "nodes": {"font": {"size": 14}},
        "edges": {"font": {"size": 10, "align": "middle"}},
        "physics": {"enabled": True, "stabilization": {"iterations": 80}},
    }
    net.set_options(json.dumps(options))
    net.add_node(center_model, label=center_model, color="#FF4B4B", size=22)

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        relation = edge["relation"]
        color = RELATION_COLORS.get(relation, "#999999")
        if source != center_model:
            net.add_node(source, label=source, color=color, size=14)
        if target != center_model:
            net.add_node(target, label=target, color=color, size=14)
        net.add_edge(source, target, label=relation.replace("_", " "), color=color, arrows="to")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as graph_file:
        graph_path = graph_file.name
    net.save_graph(graph_path)
    with open(graph_path, "r", encoding="utf-8") as graph_file:
        components.html(graph_file.read(), height=data_view_height, scrolling=True)
    os.remove(graph_path)


def render_catalog_page(query_limit, data_view_height):
    st.markdown("###### Model Catalog ######")
    st.markdown("###### Search and filter models by license risk, task, and publisher. ######")
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

    catalog_filter_signature = (
        search.strip(),
        tuple(sorted(selected_groups)),
        tuple(sorted(selected_tasks)),
        tuple(sorted(selected_brands)),
    )
    if st.session_state.get("catalog_filter_signature") != catalog_filter_signature:
        st.session_state["catalog_filter_signature"] = catalog_filter_signature

    rows = load_model_catalog(search, selected_groups, selected_tasks, selected_brands, query_limit)
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
        placeholder="",
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
        st.rerun()


def render_detail_page(query_limit, data_view_height):
    st.subheader("Model Detail")

    model_name = render_model_picker("detail_model")
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

    st.write("Task", detail["task"])

    link_col1, link_col2 = st.columns(2)
    with link_col1:
        if detail.get("hf_url"):
            st.link_button("Open on Hugging Face", detail["hf_url"])
        else:
            st.link_button("Open on Hugging Face", f"https://huggingface.co/{detail['model']}")
    with link_col2:
        if detail.get("license_link"):
            st.link_button("Open license document", detail["license_link"])

    neighborhood_edges = load_model_neighborhood(model_name, query_limit)
    hide_derivative_models = st.checkbox(
        "Hide derivative models (GGUF/forks)",
        value=True,
        help="Hides incoming DERIVED_FROM links from quantized or repackaged forks.",
    )
    if hide_derivative_models:
        neighborhood_edges = [
            neighborhood_edge
            for neighborhood_edge in neighborhood_edges
            if not (
                neighborhood_edge["relation"] == "DERIVED_FROM"
                and neighborhood_edge["direction"] == "in"
                and neighborhood_edge["entity_type"] == "Model"
            )
        ]
    if neighborhood_edges:
        render_mini_graph(model_name, neighborhood_edges, data_view_height)

    st.subheader("Related entities (1-hop)")
    st.caption(f"Center: {model_name}")
    if neighborhood_edges:
        render_model_neighborhood_table(neighborhood_edges, data_view_height)
    else:
        st.info("No direct relationships found for this model.")


def render_license_intelligence_page(query_limit, data_view_height):
    st.subheader("License Intelligence")

    group_rows = run_query("""
        MATCH (m:Model)-[:UNDER_LICENSE]->(l:License)
        OPTIONAL MATCH (l)-[:BELONGS_TO]->(g:LicenseGroup)
        RETURN coalesce(g.id, 'UNKNOWN') AS group_id, count(m) AS models
        ORDER BY models DESC
    """)
    if group_rows:
        df_groups = pd.DataFrame(group_rows)
        st.subheader("Models by license group")
        st.bar_chart(df_groups.set_index("group_id"))

    license_rows = run_query("""
        MATCH (m:Model)-[:UNDER_LICENSE]->(l:License)
        RETURN l.name AS license, count(m) AS models
        ORDER BY models DESC
        LIMIT $limit
    """, {"limit": query_limit})
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
    dataset_rows = run_query("""
        MATCH (m:Model)-[:USED_DATASET]->(d:Dataset)
        RETURN d.name AS dataset, count(m) AS uses
        ORDER BY uses DESC
        LIMIT $limit
    """, {"limit": query_limit})
    if dataset_rows:
        df_datasets = pd.DataFrame(dataset_rows)
        st.bar_chart(df_datasets.set_index("dataset"))
        ds_select = st.selectbox("Models using dataset", df_datasets["dataset"].tolist())
        if ds_select:
            model_rows = run_query(
                """
                MATCH (m:Model)-[:USED_DATASET]->(d:Dataset {name: $dataset_name})
                RETURN m.name AS model
                ORDER BY m.downloads DESC
                LIMIT $limit
                """,
                {"dataset_name": ds_select, "limit": query_limit},
            )
            render_scrollable_dataframe(
                pd.DataFrame(model_rows), data_view_height, use_container_width=True
            )


def render_graph_explorer_page(query_limit, data_view_height):
    st.subheader("Graph Explorer")
    st.caption("Best for small, filtered subgraphs — not the full database.")

    center_model = render_model_picker("detail_model")
    rel_types = [r["rel"] for r in run_query("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType AS rel")]
    selected_rels = st.multiselect("Relationships", rel_types, default=rel_types[:6])

    if center_model and selected_rels:
        results = run_query("""
        MATCH (m:Model {name: $center_model})-[r]->(target)
        WHERE type(r) IN $rels
        RETURN m.name AS model,
               coalesce(target.name, target.id) AS target_name,
               type(r) AS relation
        UNION
        MATCH (source)-[r]->(m:Model {name: $center_model})
        WHERE type(r) IN $rels
        RETURN m.name AS model,
               coalesce(source.name, source.id) AS target_name,
               type(r) AS relation
        LIMIT $limit
        """, {"center_model": center_model, "rels": selected_rels, "limit": query_limit})
        
        if results:
            net = Network(height=f"{data_view_height}px", width="100%", bgcolor="#ffffff", directed=True)
            net.set_options(json.dumps({
                "physics": {"enabled": True, "stabilization": {"iterations": 100}},
            }))
            for row in results:
                rel_color = RELATION_COLORS.get(row["relation"], "#999999")
                net.add_node(row["model"], label=row["model"], color="#FF4B4B")
                net.add_node(row["target_name"], label=row["target_name"], color=rel_color)
                net.add_edge(row["model"], row["target_name"], label=row["relation"], color=rel_color, arrows="to")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as graph_file:
                graph_path = graph_file.name
            net.save_graph(graph_path)
            with open(graph_path, "r", encoding="utf-8") as graph_file:
                components.html(graph_file.read(), height=data_view_height, scrolling=True)
            os.remove(graph_path)
        else:
            st.warning("No relationships found.")


# --- UI ---
inject_app_styles()
st.markdown("##### AI Lineage & Compliance Knowledge Graph. #####") 
if "mode" not in st.session_state:
    st.session_state["mode"] = "Model Catalog"

if st.session_state.get("pending_mode"):
    st.session_state["mode"] = st.session_state.pop("pending_mode")

st.sidebar.markdown("##### ModelRoot")
st.sidebar.markdown("##### AI Lineage & Compliance Knowledge Graph. #####") 
 

mode_options = ["Model Catalog", "Model Detail", "License Intelligence", "Graph Explorer"]
st.sidebar.radio("Mode", mode_options, key="mode")
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

with st.sidebar.expander("Database snapshot"):
    snapshot = run_query("""
        MATCH (m:Model) WITH count(m) AS models
        MATCH (l:License) WITH models, count(l) AS licenses
        MATCH (g:LicenseGroup) RETURN models, licenses, count(g) AS groups
    """)
    if snapshot:
        st.metric("Models", f"{snapshot[0]['models']:,}")
        st.metric("Licenses", f"{snapshot[0]['licenses']:,}")
        st.metric("License groups", snapshot[0]["groups"])

if mode == "Model Catalog":
    render_catalog_page(query_limit, data_view_height)
elif mode == "Model Detail":
    render_detail_page(query_limit, data_view_height)
elif mode == "License Intelligence":
    render_license_intelligence_page(query_limit, data_view_height)
else:
    render_graph_explorer_page(query_limit, data_view_height)

