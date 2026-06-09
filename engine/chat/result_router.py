"""Classify Neo4j rows into catalog, detail, or text chat outputs."""

import json
from typing import Any

import pandas as pd

from engine.chat.types import ChatOutputKind, ChatResult

MODEL_FIELD_CANDIDATES = (
    "model",
    "model_name",
    "name",
    "best_llm",
    "best_emb",
    "derivative",
    "base_model",
)

CATALOG_COLUMN_ALIASES = {
    "model": ("model", "model_name", "name", "best_llm", "best_emb"),
    "brand": ("brand", "publisher", "main_brand"),
    "task": ("task", "pipeline_tag"),
    "license": ("license", "license_name", "license_id"),
    "risk_level": ("risk_level", "license_group_name", "group_name"),
    "license_group": ("license_group", "group_id", "license_group_id"),
    "downloads": ("downloads", "dl", "model_downloads"),
    "benchmark": ("benchmark", "oll_rank"),
    "dataset": ("dataset", "dataset_name"),
    "hf_url": ("hf_url",),
}


def _first_present_value(row: dict[str, Any], field_names: tuple[str, ...]) -> Any:
    for field_name in field_names:
        if field_name in row and row[field_name] not in (None, ""):
            return row[field_name]
    return None


def extract_model_names_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    model_names = []
    for row in rows:
        model_value = _first_present_value(row, MODEL_FIELD_CANDIDATES)
        if isinstance(model_value, str) and model_value.strip():
            model_names.append(model_value.strip())
    return list(dict.fromkeys(model_names))


def normalize_rows_to_catalog_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    normalized_rows = []
    for row in rows:
        normalized_row = {}
        for target_column, aliases in CATALOG_COLUMN_ALIASES.items():
            value = _first_present_value(row, aliases)
            if value is not None:
                normalized_row[target_column] = value
        if not normalized_row:
            normalized_row = dict(row)
        normalized_rows.append(normalized_row)
    return pd.DataFrame(normalized_rows)


def classify_chat_rows(
    rows: list[dict[str, Any]],
    summary_text: str,
    cypher: str,
) -> ChatResult:
    if not rows:
        return ChatResult(
            answer_text=summary_text,
            output_kind=ChatOutputKind.TEXT,
            cypher=cypher,
            rows=rows,
        )

    model_names = extract_model_names_from_rows(rows)

    if len(model_names) == 1:
        return ChatResult(
            answer_text=summary_text,
            output_kind=ChatOutputKind.DETAIL,
            cypher=cypher,
            rows=rows,
            detail_model_name=model_names[0],
        )

    if len(model_names) >= 2:
        catalog_dataframe = normalize_rows_to_catalog_dataframe(rows)
        return ChatResult(
            answer_text=summary_text,
            output_kind=ChatOutputKind.CATALOG,
            cypher=cypher,
            rows=rows,
            catalog_dataframe=catalog_dataframe,
        )

    return ChatResult(
        answer_text=summary_text,
        output_kind=ChatOutputKind.TEXT,
        cypher=cypher,
        rows=rows,
    )


def rows_to_summary_json(
    rows: list[dict[str, Any]],
    max_rows: int = 25,
    max_json_chars: int = 8000,
) -> str:
    trimmed_rows = rows[:max_rows]
    rows_json = json.dumps(trimmed_rows, default=str, ensure_ascii=False)
    if len(rows_json) <= max_json_chars:
        return rows_json

    compact_json = json.dumps(
        trimmed_rows,
        default=str,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if len(compact_json) <= max_json_chars:
        return compact_json

    return compact_json[: max_json_chars - 3] + "..."

