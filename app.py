"""
ModelRoot Streamlit app — catalog-first UX for license-aware model discovery.

Modes: Model Catalog (default), Model Detail (1-hop graph), License Intelligence, Graph Explorer.
"""
import json
import os
import sys
import tempfile

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from neo4j import GraphDatabase
from pyvis.network import Network

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source", "scrapers"))
from neo4j_config import NEO4J_AUTH, NEO4J_URI

st.set_page_config(layout="wide", page_title="ModelRoot Pro", page_icon="🌐")

LICENSE_GROUP_COLORS = {
    "GREEN": "#22c55e",
    "YELLOW": "#eab308",
    "ORANGE": "#f97316",
    "RED_COPYLEFT": "#a855f7",
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


def render_license_badge(group_id):
    if not group_id:
        return "—"
    return (
        f'<span style="background:{get_license_group_color(group_id)};'
        f'{LICENSE_GROUP_CELL_STYLE};padding:2px 8px;border-radius:4px;">{group_id}</span>'
    )


def load_model_catalog(search, license_groups, tasks, brands, limit):
    query = """
    MATCH (m:Model)
    OPTIONAL MATCH (m)-[:UNDER_LICENSE]->(l:License)
    OPTIONAL MATCH (l)-[:BELONGS_TO]->(g:LicenseGroup)
    OPTIONAL MATCH (m)-[:PERFORMS]->(t:Task)
    OPTIONAL MATCH (m)-[:PUBLISHED_BY]->(b:MainBrand)
    WHERE ($search = '' OR toLower(m.name) CONTAINS toLower($search))
      AND (size($license_groups) = 0 OR g.id IN $license_groups OR (g IS NULL AND 'UNKNOWN' IN $license_groups))
      AND (size($tasks) = 0 OR t.name IN $tasks)
      AND (size($brands) = 0 OR b.name IN $brands)
    RETURN m.name AS model,
           coalesce(b.name, '—') AS brand,
           coalesce(t.name, '—') AS task,
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
        "search": search.strip(),
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


def load_model_neighborhood(model_name, limit=25):
    query = """
    MATCH (m:Model {name: $model_name})-[r]->(n)
    RETURN m.name AS source,
           coalesce(n.name, n.id) AS target,
           labels(n)[0] AS target_type,
           type(r) AS relation
    UNION
    MATCH (n)-[r]->(m:Model {name: $model_name})
    RETURN coalesce(n.name, n.id) AS source,
           m.name AS target,
           labels(n)[0] AS target_type,
           type(r) AS relation
    LIMIT $limit
    """
    return run_query(query, {"model_name": model_name, "limit": limit})


def format_license_group_filter_label(group_id, license_group_metadata_by_id):
    if group_id == "UNKNOWN":
        return "UNKNOWN — No risk group assigned"
    group_metadata = license_group_metadata_by_id.get(group_id)
    if not group_metadata:
        return group_id
    return f"{group_id} — {group_metadata['name']}"


def render_license_group_legend():
    license_group_rows = get_license_group_metadata()
    with st.expander("License risk levels reference", expanded=False):
        st.dataframe(
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
            hide_index=True,
            use_container_width=True,
        )


CATALOG_DISPLAY_COLUMNS = [
    "model",
    "brand",
    "task",
    "license",
    "risk_level",
    "risk_guidance",
    "downloads",
    "hf_url",
    "license_link",
]


def build_catalog_dataframe(rows):
    catalog_dataframe = pd.DataFrame(rows)
    catalog_dataframe["hf_url"] = catalog_dataframe.apply(
        lambda row: row["hf_url"] or f"https://huggingface.co/{row['model']}",
        axis=1,
    )
    return catalog_dataframe


def get_catalog_column_config():
    return {
        "model": st.column_config.TextColumn("Model", width="large"),
        "brand": st.column_config.TextColumn("Brand"),
        "task": st.column_config.TextColumn("Task"),
        "license": st.column_config.TextColumn("License ID"),
        "risk_level": st.column_config.TextColumn("Risk level", width="medium"),
        "risk_guidance": st.column_config.TextColumn("Compliance guidance", width="large"),
        "downloads": st.column_config.NumberColumn("Downloads", format="%d"),
        "hf_url": st.column_config.LinkColumn("HF", display_text="Open"),
        "license_link": st.column_config.LinkColumn("License doc", display_text="Open"),
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
        ["risk_level"],
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
               coalesce(m.hf_url, 'https://huggingface.co/' + m.name) AS hf_url
        ORDER BY m.downloads DESC
        LIMIT $limit
    """, {"limit": limit})


def render_mini_graph(center_model, edges):
    net = Network(height="420px", width="100%", bgcolor="#ffffff", directed=True)
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
        components.html(graph_file.read(), height=440)
    os.remove(graph_path)


def render_catalog_page():
    st.header("📚 Model Catalog")
    st.caption("Search and filter models by license risk, task, and publisher.")
    render_license_group_legend()

    col1, col2, col3 = st.columns(3)
    with col1:
        search = st.text_input("Search model", placeholder="e.g. Llama, bert, embedding")
    with col2:
        license_group_metadata_by_id = {
            group["id"]: group for group in get_license_group_metadata()
        }
        group_options = get_license_groups() + ["UNKNOWN"]
        selected_groups = st.multiselect(
            "Risk level",
            group_options,
            default=[],
            format_func=lambda group_id: format_license_group_filter_label(
                group_id, license_group_metadata_by_id
            ),
        )
    with col3:
        limit = st.slider("Max results", 25, 500, 100)

    col4, col5 = st.columns(2)
    with col4:
        selected_tasks = st.multiselect("Task", get_tasks())
    with col5:
        selected_brands = st.multiselect("Brand", get_brands())

    rows = load_model_catalog(search, selected_groups, selected_tasks, selected_brands, limit)
    if not rows:
        st.warning("No models match the current filters.")
        return

    df = build_catalog_dataframe(rows)
    st.markdown(f"**{len(df)}** models — select a row or use the button below to open detail.")

    table_selection = st.dataframe(
        build_styled_catalog_dataframe(df),
        column_config=get_catalog_column_config(),
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key="catalog_table",
    )

    model_names = df["model"].tolist()
    default_model = st.session_state.get("detail_model")
    default_index = model_names.index(default_model) if default_model in model_names else 0
    selected_model = st.selectbox("Select model", model_names, index=default_index)

    if table_selection.selection.rows:
        selected_model = df.iloc[table_selection.selection.rows[0]]["model"]

    if st.button("View detail", type="primary"):
        st.session_state["detail_model"] = selected_model
        st.session_state["mode"] = "Model Detail"
        st.rerun()


def render_detail_page():
    st.header("🔎 Model Detail")

    models = [r["model"] for r in run_query(
        "MATCH (m:Model) RETURN m.name AS model ORDER BY m.downloads DESC LIMIT 500"
    )]
    default_model = st.session_state.get("detail_model")
    default_index = models.index(default_model) if default_model in models else 0
    model_name = st.selectbox("Select model", models, index=default_index)
    st.session_state["detail_model"] = model_name

    detail = load_model_detail(model_name)
    if not detail:
        st.error("Model not found.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Downloads", f"{detail['downloads']:,}")
    with c2:
        st.write("License", detail["license"])

    if detail["license_group"] == "UNKNOWN":
        st.info("This license is not mapped to a risk group yet.")
    else:
        st.markdown(render_license_badge(detail["license_group"]), unsafe_allow_html=True)
        if detail.get("license_group_name"):
            st.markdown(f"**{detail['license_group_name']}**")
        if detail.get("license_group_compliance"):
            st.caption(detail["license_group_compliance"])

    st.write("Brand", detail["brand"])
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

    st.subheader("Neighborhood (1-hop)")
    edges = load_model_neighborhood(model_name)
    if edges:
        st.dataframe(pd.DataFrame(edges), use_container_width=True)
        render_mini_graph(model_name, edges)
    else:
        st.info("No direct relationships found for this model.")


def render_license_intelligence_page():
    st.header("📊 License Intelligence")

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
        LIMIT 15
    """)
    if license_rows:
        st.subheader("Top licenses")
        st.bar_chart(pd.DataFrame(license_rows).set_index("license"))

    top_models_by_risk = load_top_models_by_risk(limit=30)
    if top_models_by_risk:
        st.subheader("Top models by license risk")
        top_models_dataframe = pd.DataFrame(top_models_by_risk)
        st.dataframe(
            build_styled_risk_rows_dataframe(
                top_models_dataframe,
                TOP_MODELS_BY_RISK_DISPLAY_COLUMNS,
                TOP_MODELS_BY_RISK_STYLED_COLUMNS,
            ),
            column_config={
                "risk_level": st.column_config.TextColumn("Risk level", width="medium"),
                "risk_guidance": st.column_config.TextColumn("Compliance guidance", width="large"),
                "model": st.column_config.TextColumn("Model", width="large"),
                "brand": st.column_config.TextColumn("Brand"),
                "license": st.column_config.TextColumn("License ID"),
                "downloads": st.column_config.NumberColumn("Downloads", format="%d"),
                "hf_url": st.column_config.LinkColumn("HF", display_text="Open"),
            },
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Top datasets by usage")
    dataset_rows = run_query("""
        MATCH (m:Model)-[:USED_DATASET]->(d:Dataset)
        RETURN d.name AS dataset, count(m) AS uses
        ORDER BY uses DESC
        LIMIT 10
    """)
    if dataset_rows:
        df_datasets = pd.DataFrame(dataset_rows)
        st.bar_chart(df_datasets.set_index("dataset"))
        ds_select = st.selectbox("Models using dataset", df_datasets["dataset"].tolist())
        if ds_select:
            model_rows = run_query(
                "MATCH (m:Model)-[:USED_DATASET]->(d:Dataset {name: $ds}) RETURN m.name AS model ORDER BY m.downloads DESC LIMIT 50",
                {"ds": ds_select},
            )
            st.dataframe(pd.DataFrame(model_rows), use_container_width=True)


def render_graph_explorer_page():
    st.header("🕸️ Graph Explorer (advanced)")
    st.caption("Best for small, filtered subgraphs — not the full database.")

    search = st.text_input("Search model or related node:", placeholder="e.g. Llama")
    limit = st.slider("Number of edges", 10, 100, 40)
    rel_types = [r["rel"] for r in run_query("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType AS rel")]
    selected_rels = st.multiselect("Relationships", rel_types, default=rel_types[:6])

    if search and selected_rels:
        results = run_query("""
            MATCH (m:Model)-[r]->(target)
            WHERE (toLower(m.name) CONTAINS toLower($s) OR toLower(target.name) CONTAINS toLower($s))
              AND type(r) IN $rels
            RETURN m.name AS model,
                   coalesce(target.name, target.id) AS target_name,
                   type(r) AS relation
            LIMIT $limit
        """, {"s": search, "rels": selected_rels, "limit": limit})

        if results:
            net = Network(height="600px", width="100%", bgcolor="#ffffff", directed=True)
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
                components.html(graph_file.read(), height=620)
            os.remove(graph_path)
        else:
            st.warning("No relationships found.")


# --- UI ---
st.title("🌐 ModelRoot Pro")

if "mode" not in st.session_state:
    st.session_state["mode"] = "Model Catalog"

mode_options = ["Model Catalog", "Model Detail", "License Intelligence", "Graph Explorer"]
mode = st.sidebar.radio(
    "Mode",
    mode_options,
    index=mode_options.index(st.session_state["mode"]),
)
st.session_state["mode"] = mode

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
    render_catalog_page()
elif mode == "Model Detail":
    render_detail_page()
elif mode == "License Intelligence":
    render_license_intelligence_page()
else:
    render_graph_explorer_page()
