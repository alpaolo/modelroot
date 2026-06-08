"""
Hugging Face organization URLs keyed by MainBrand name (from Neo4j).
Loaded from brand_urls.json to keep imports fast and reliable.
"""

import json
from functools import lru_cache
from pathlib import Path

_BRAND_URLS_JSON_PATH = Path(__file__).with_name("brand_urls.json")


@lru_cache(maxsize=1)
def _load_brand_hf_urls():
    with _BRAND_URLS_JSON_PATH.open(encoding="utf-8") as brand_urls_file:
        return json.load(brand_urls_file)


BRAND_HF_URLS = _load_brand_hf_urls()
