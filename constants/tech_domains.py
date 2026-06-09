"""
Tech domain display names for HF pipeline tasks.
Neo4j is primary; JSON map is fallback when GROUPS links are missing.
"""

import json
from functools import lru_cache
from pathlib import Path

_TECH_DOMAINS_JSON_PATH = Path(__file__).with_name("tech_domains.json")
_TASK_TECH_DOMAIN_MAP_JSON_PATH = Path(__file__).with_name("task_tech_domain_map.json")


@lru_cache(maxsize=1)
def _load_tech_domain_by_id():
    with _TECH_DOMAINS_JSON_PATH.open(encoding="utf-8") as tech_domains_file:
        tech_domains = json.load(tech_domains_file)
    return {tech_domain["id"]: tech_domain for tech_domain in tech_domains}


@lru_cache(maxsize=1)
def _load_task_tech_domain_map():
    with _TASK_TECH_DOMAIN_MAP_JSON_PATH.open(encoding="utf-8") as task_map_file:
        return json.load(task_map_file)


def resolve_tech_domain_names_for_task(task_name, neo4j_tech_domain_names=None):
    if neo4j_tech_domain_names:
        resolved_from_neo4j = [name for name in neo4j_tech_domain_names if name]
        if resolved_from_neo4j:
            return resolved_from_neo4j

    if not task_name or task_name == "—":
        return []

    tech_domain_by_id = _load_tech_domain_by_id()
    task_tech_domain_map = _load_task_tech_domain_map()
    tech_domain_ids = task_tech_domain_map.get(task_name, [])
    return [
        tech_domain_by_id[tech_domain_id]["name"]
        for tech_domain_id in tech_domain_ids
        if tech_domain_id in tech_domain_by_id
    ]
