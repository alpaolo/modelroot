"""
Open LLM Leaderboard scores keyed by Model.name.
Loaded from open_llm_leaderboard.json (built by enrich_open_llm_leaderboard.py).
"""

import json
from functools import lru_cache
from pathlib import Path

_OPEN_LLM_LEADERBOARD_JSON_PATH = Path(__file__).with_name("open_llm_leaderboard.json")


@lru_cache(maxsize=1)
def _load_open_llm_leaderboard_payload():
    if not _OPEN_LLM_LEADERBOARD_JSON_PATH.exists():
        return {"meta": {}, "models": {}}
    with _OPEN_LLM_LEADERBOARD_JSON_PATH.open(encoding="utf-8") as leaderboard_file:
        return json.load(leaderboard_file)


def get_open_llm_leaderboard_entry(model_name):
    models = _load_open_llm_leaderboard_payload().get("models", {})
    return models.get(model_name)
