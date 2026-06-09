"""
Dataset documentation URLs keyed by Dataset.name from Neo4j.
Loaded from dataset_urls.json (built by engine/scrapers/enrich_dataset_urls.py).
"""

import json
from functools import lru_cache
from pathlib import Path

_DATASET_URLS_JSON_PATH = Path(__file__).with_name("dataset_urls.json")


@lru_cache(maxsize=1)
def _load_dataset_documentation_urls():
    if not _DATASET_URLS_JSON_PATH.exists():
        return {}
    with _DATASET_URLS_JSON_PATH.open(encoding="utf-8") as dataset_urls_file:
        return json.load(dataset_urls_file)


DATASET_DOCUMENTATION_URLS = _load_dataset_documentation_urls()
