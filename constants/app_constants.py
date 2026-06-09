"""
ModelRoot UI constants — colors, column layouts, document URLs, and query limits.
Brand HF URLs live in constants/brand_urls.py.
"""

# Sidebar query / layout limits
QUERY_LIMIT_MIN = 25
QUERY_LIMIT_MAX = 500
QUERY_LIMIT_DEFAULT = 100
DATA_VIEW_HEIGHT_MIN = 250
DATA_VIEW_HEIGHT_MAX = 900
DATA_VIEW_HEIGHT_DEFAULT = 450

# License risk badge colors
LICENSE_GROUP_COLORS = {
    "GREEN": "#22c55e",
    "YELLOW": "#eab308",
    "ORANGE": "#f97316",
    "RED_COPYLEFT": "#dd22ce",
    "RED_RESTRICTED": "#ef4444",
}
UNKNOWN_LICENSE_GROUP_COLOR = "#94a3b8"
LICENSE_GROUP_CELL_STYLE = "color: white; font-weight: bold"

# Graph relation edge / node colors
RELATION_COLORS = {
    "UNDER_LICENSE": "#6366f1",
    "PERFORMS": "#32CD32",
    "USED_DATASET": "#1C83E1",
    "CITED_IN": "#8A2BE2",
    "DERIVED_FROM": "#f59e0b",
    "PUBLISHED_BY": "#FF9900",
    "BASED_ON_PAPER": "#8B5CF6",
}
DEFAULT_RELATION_COLOR = "#999999"
CENTER_MODEL_NODE_COLOR = "#FF4B4B"
GRAPH_BACKGROUND_COLOR = "#ffffff"
GRAPH_IFRAME_HEIGHT_TRIM_PX = 18

# Model picker
MODEL_PICKER_BROWSE_LIMIT = 50
MODEL_PICKER_MATCH_LIMIT = 50

# Dataframe link columns
LINK_COLUMN_DISPLAY_TEXT = "Open"
CATALOG_LINK_COLUMNS = ["hf_url", "license_link"]
DATAFRAME_PLACEHOLDER = ""

# License documentation URLs (SPDX, CC, vendor pages)
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

# Benchmark panel (Model Detail)
BENCHMARK_PANEL_BORDER_COLOR = "#2563eb"
BENCHMARK_PANEL_BACKGROUND_COLOR = "#f8fafc"
OPEN_LLM_BENCHMARK_SOURCE_URL = (
    "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard"
)

# Catalog / license-intelligence table columns
CATALOG_DISPLAY_COLUMNS = [
    "model",
    "brand",
    "task",
    "benchmark",
    "license",
    "risk_level",
    "downloads",
    "hf_url",
    "license_link",
]
CATALOG_STYLED_COLUMNS = ["model", "license", "risk_level"]

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

# Model detail neighborhood table
NEIGHBORHOOD_KIND_LABELS = {
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
NEIGHBORHOOD_LINK_DISPLAY_REGEX = r".*#(.*)$"
NEIGHBORHOOD_PLACEHOLDER_LINK_BASE = "https://modelroot.pending/#"

# Pyvis graph options
MINI_GRAPH_OPTIONS = {
    "nodes": {"font": {"size": 14}},
    "edges": {"font": {"size": 10, "align": "middle"}},
    "physics": {"enabled": True, "stabilization": {"iterations": 80}},
}
GRAPH_EXPLORER_PHYSICS_OPTIONS = {
    "physics": {"enabled": True, "stabilization": {"iterations": 100}},
}
GRAPH_EXPLORER_DEFAULT_REL_LIMIT = 6

# App navigation
MODE_OPTIONS = ["Model Catalog", "Model Detail", "License Intelligence", "Graph Explorer"]

# License group legend fallback row
UNCLASSIFIED_LICENSE_GROUP_LEGEND = {
    "Risk level": "Unclassified",
    "Compliance guidance": "License not mapped to a risk group yet.",
}
