"""
Import TechDomain macro-areas into Neo4j and link HF pipeline Task nodes.

Sources:
- constants/tech_domains.json — TechDomain definitions
- constants/task_tech_domain_map.json — Task.name → [tech_domain_id, ...]

Creates:
  (td:TechDomain {id, name, compliance_note})-[:GROUPS]->(t:Task {name})

Idempotent: safe to re-run (MERGE only).
"""
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / ".env"))
import config as env

NEO4J_URI = env.NEO4J_URI
NEO4J_AUTH = env.NEO4J_AUTH
TECH_DOMAINS_JSON_PATH = PROJECT_ROOT / "constants" / "tech_domains.json"
TASK_TECH_DOMAIN_MAP_JSON_PATH = PROJECT_ROOT / "constants" / "task_tech_domain_map.json"

MERGE_TECH_DOMAIN_QUERY = """
MERGE (td:TechDomain {id: $tech_domain_id})
SET td.name = $tech_domain_name,
    td.compliance_note = $compliance_note
RETURN td.id AS tech_domain_id
"""

MERGE_TASK_TECH_DOMAIN_LINK_QUERY = """
MERGE (t:Task {name: $task_name})
MERGE (td:TechDomain {id: $tech_domain_id})
MERGE (td)-[:GROUPS]->(t)
RETURN td.id AS tech_domain_id, t.name AS task_name
"""


def log(message):
    print(message, flush=True)


def load_json_array(json_path):
    with json_path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def load_task_tech_domain_map():
    with TASK_TECH_DOMAIN_MAP_JSON_PATH.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def merge_tech_domains(driver, tech_domains):
    merged_tech_domain_count = 0
    with driver.session() as session:
        for tech_domain in tech_domains:
            session.run(
                MERGE_TECH_DOMAIN_QUERY,
                tech_domain_id=tech_domain["id"],
                tech_domain_name=tech_domain["name"],
                compliance_note=tech_domain.get("compliance_note"),
            )
            merged_tech_domain_count += 1
    return merged_tech_domain_count


def merge_task_tech_domain_links(driver, task_tech_domain_map):
    linked_relationship_count = 0
    unmapped_task_names = []

    with driver.session() as session:
        existing_task_names = {
            record["task_name"]
            for record in session.run("MATCH (t:Task) RETURN t.name AS task_name")
        }

        for task_name, tech_domain_ids in task_tech_domain_map.items():
            if task_name not in existing_task_names:
                unmapped_task_names.append(task_name)

            for tech_domain_id in tech_domain_ids:
                session.run(
                    MERGE_TASK_TECH_DOMAIN_LINK_QUERY,
                    task_name=task_name,
                    tech_domain_id=tech_domain_id,
                )
                linked_relationship_count += 1

        orphan_tasks = [
            record["task_name"]
            for record in session.run("""
                MATCH (t:Task)
                WHERE NOT EXISTS { MATCH (:TechDomain)-[:GROUPS]->(t) }
                RETURN t.name AS task_name
                ORDER BY task_name
            """)
        ]

    return linked_relationship_count, unmapped_task_names, orphan_tasks


def print_summary(driver):
    with driver.session() as session:
        tech_domain_count = session.run(
            "MATCH (td:TechDomain) RETURN count(td) AS c"
        ).single()["c"]
        groups_relationship_count = session.run(
            "MATCH (:TechDomain)-[r:GROUPS]->(:Task) RETURN count(r) AS c"
        ).single()["c"]
        grouped_task_count = session.run("""
            MATCH (t:Task)
            WHERE EXISTS { MATCH (:TechDomain)-[:GROUPS]->(t) }
            RETURN count(t) AS c
        """).single()["c"]
        task_count = session.run("MATCH (t:Task) RETURN count(t) AS c").single()["c"]

    log(f"TechDomain nodes:      {tech_domain_count}")
    log(f"GROUPS relationships:  {groups_relationship_count}")
    log(f"Tasks grouped:         {grouped_task_count}/{task_count}")


def main():
    log("=== enrich_tech_domains.py ===")

    tech_domains = load_json_array(TECH_DOMAINS_JSON_PATH)
    task_tech_domain_map = load_task_tech_domain_map()

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    merged_tech_domain_count = merge_tech_domains(driver, tech_domains)
    log(f"[OK] TechDomain nodes merged: {merged_tech_domain_count}")

    linked_relationship_count, unmapped_in_db, orphan_tasks = merge_task_tech_domain_links(
        driver,
        task_tech_domain_map,
    )
    log(f"[OK] GROUPS links merged: {linked_relationship_count}")

    if unmapped_in_db:
        log(f"[WARN] Tasks in map but not yet in DB: {unmapped_in_db}")
    if orphan_tasks:
        log(f"[WARN] Tasks in DB without TechDomain: {orphan_tasks}")

    log("\n=== SNAPSHOT ===")
    print_summary(driver)

    driver.close()
    log("\n=== COMPLETATO ===")


if __name__ == "__main__":
    main()
