"""Chat query result types for ModelRoot graph Q&A."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import pandas as pd


class ChatOutputKind(str, Enum):
    TEXT = "text"
    CATALOG = "catalog"
    DETAIL = "detail"


@dataclass
class ChatResult:
    answer_text: str
    output_kind: ChatOutputKind
    cypher: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)
    catalog_dataframe: Optional[pd.DataFrame] = None
    detail_model_name: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None
