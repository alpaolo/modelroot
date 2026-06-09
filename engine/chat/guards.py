"""Read-only and schema validation for chat-generated Cypher."""

import re

from engine.chat.schema_rules import (
    ALLOWED_CHAT_NODE_LABELS,
    ALLOWED_CHAT_RELATIONSHIP_TYPES,
    FORBIDDEN_LEGACY_RELATIONSHIP_TYPES,
)

FORBIDDEN_CYPHER_PATTERN = re.compile(
    r"\b("
    r"CREATE|MERGE|DELETE|DETACH\s+DELETE|SET|REMOVE|DROP|FOREACH|"
    r"LOAD\s+CSV|LOAD\s+XML|START\s+DATABASE|STOP\s+DATABASE|"
    r"GRANT|DENY|REVOKE|ALTER|RENAME|INDEX\s+ON|CONSTRAINT\s+ON"
    r")\b",
    re.IGNORECASE,
)
FORBIDDEN_CALL_PROCEDURE_PATTERN = re.compile(
    r"\bCALL\s+(?!(\s*\{))",
    re.IGNORECASE,
)
FORBIDDEN_DBMS_PATTERN = re.compile(r"\bdbms\.", re.IGNORECASE)
CYPHER_FENCE_PATTERN = re.compile(r"```(?:cypher)?\s*([\s\S]*?)```", re.IGNORECASE)
LIMIT_PATTERN = re.compile(r"\bLIMIT\s+(\d+)\b", re.IGNORECASE)
RELATIONSHIP_BRACKET_PATTERN = re.compile(r"-\[[^\]]+\]-?", re.IGNORECASE)
RELATIONSHIP_TYPE_PATTERN = re.compile(
    r":([A-Za-z_][A-Za-z0-9_]*(?:\|[A-Za-z_][A-Za-z0-9_]*)*)",
)
NODE_LABEL_PATTERN = re.compile(
    r"\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:([A-Za-z][A-Za-z0-9_]*)(?=\s|:|\)|\{)",
)
TYPE_FUNCTION_LITERAL_PATTERN = re.compile(
    r"type\s*\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\)\s*=\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)


class CypherGuardError(ValueError):
    """Raised when generated Cypher fails read-only or schema validation."""


def extract_cypher_from_llm_text(llm_text: str) -> str:
    fenced_match = CYPHER_FENCE_PATTERN.search(llm_text or "")
    if fenced_match:
        return fenced_match.group(1).strip()
    return (llm_text or "").strip()


def _split_relationship_type_tokens(relationship_type_token: str) -> list[str]:
    return [
        relationship_type.strip()
        for relationship_type in relationship_type_token.split("|")
        if relationship_type.strip()
    ]


def extract_relationship_types(cypher_query: str) -> list[str]:
    relationship_types = []

    for relationship_bracket in RELATIONSHIP_BRACKET_PATTERN.findall(cypher_query):
        if "*" in relationship_bracket:
            raise CypherGuardError("Variable-length relationships are not allowed.")

        if ":" not in relationship_bracket:
            raise CypherGuardError(
                "Untyped relationships are not allowed. Use explicit ModelRoot relationship types."
            )

        for relationship_type_match in RELATIONSHIP_TYPE_PATTERN.findall(relationship_bracket):
            relationship_types.extend(_split_relationship_type_tokens(relationship_type_match))

    for literal_relationship_type in TYPE_FUNCTION_LITERAL_PATTERN.findall(cypher_query):
        relationship_types.append(literal_relationship_type.strip())

    return relationship_types


def extract_node_labels(cypher_query: str) -> list[str]:
    return NODE_LABEL_PATTERN.findall(cypher_query)


def validate_graph_schema_cypher(cypher_query: str) -> None:
    if not re.search(r"\bMATCH\b", cypher_query, re.IGNORECASE):
        raise CypherGuardError("Query must contain at least one MATCH clause.")

    relationship_types = extract_relationship_types(cypher_query)
    if not relationship_types:
        raise CypherGuardError(
            "Query must use explicit ModelRoot relationships "
            f"({', '.join(sorted(ALLOWED_CHAT_RELATIONSHIP_TYPES))})."
        )

    for relationship_type in relationship_types:
        if relationship_type in FORBIDDEN_LEGACY_RELATIONSHIP_TYPES:
            raise CypherGuardError(
                f"Relationship '{relationship_type}' is not part of the ModelRoot graph. "
                "Use UNDER_LICENSE, PERFORMS, USED_DATASET, etc."
            )
        if relationship_type not in ALLOWED_CHAT_RELATIONSHIP_TYPES:
            raise CypherGuardError(
                f"Relationship '{relationship_type}' is not allowed. "
                f"Allowed: {', '.join(sorted(ALLOWED_CHAT_RELATIONSHIP_TYPES))}."
            )

    for node_label in extract_node_labels(cypher_query):
        if node_label not in ALLOWED_CHAT_NODE_LABELS:
            raise CypherGuardError(
                f"Node label '{node_label}' is not allowed. "
                f"Allowed: {', '.join(sorted(ALLOWED_CHAT_NODE_LABELS))}."
            )


def validate_read_only_cypher(cypher_text: str, max_rows: int) -> str:
    cypher_query = extract_cypher_from_llm_text(cypher_text).strip().rstrip(";")
    if not cypher_query:
        raise CypherGuardError("Empty Cypher query.")

    if FORBIDDEN_CYPHER_PATTERN.search(cypher_query):
        raise CypherGuardError("Write or admin Cypher statements are not allowed.")

    if FORBIDDEN_CALL_PROCEDURE_PATTERN.search(cypher_query):
        raise CypherGuardError("CALL procedures are not allowed (read-only subqueries only).")

    if FORBIDDEN_DBMS_PATTERN.search(cypher_query):
        raise CypherGuardError("dbms.* procedures are not allowed.")

    validate_graph_schema_cypher(cypher_query)

    limit_match = LIMIT_PATTERN.search(cypher_query)
    if limit_match:
        requested_limit = int(limit_match.group(1))
        if requested_limit > max_rows:
            cypher_query = LIMIT_PATTERN.sub(f"LIMIT {max_rows}", cypher_query, count=1)
    else:
        cypher_query = f"{cypher_query}\nLIMIT {max_rows}"

    return cypher_query
