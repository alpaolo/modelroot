"""ModelRoot Neo4j schema allowlists for chat Cypher validation (aligned with constants/query.py)."""

ALLOWED_CHAT_NODE_LABELS = frozenset({
    "Model",
    "License",
    "LicenseGroup",
    "Task",
    "TechDomain",
    "MainBrand",
    "Dataset",
    "Paper",
})

ALLOWED_CHAT_RELATIONSHIP_TYPES = frozenset({
    "UNDER_LICENSE",
    "BELONGS_TO",
    "PERFORMS",
    "GROUPS",
    "PUBLISHED_BY",
    "DERIVED_FROM",
    "USED_DATASET",
    "CITED_IN",
    "BASED_ON_PAPER",
})

FORBIDDEN_LEGACY_RELATIONSHIP_TYPES = frozenset({
    "LICENSED_AS",
    "PERFORMS_TASK",
    "USED_FOR_DOMAIN",
})
