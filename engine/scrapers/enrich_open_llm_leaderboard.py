"""
Open LLM Leaderboard — FETCH → MERGE Neo4j → CACHE JSON.

Source: open-llm-leaderboard/contents (HF dataset parquet)
Join key: fullname = Model.name
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / ".env"))
import config as env

NEO4J_URI = env.NEO4J_URI
NEO4J_AUTH = env.NEO4J_AUTH

OPEN_LLM_PARQUET_URL = (
    "https://huggingface.co/api/datasets/"
    "open-llm-leaderboard/contents/parquet/default/train/0.parquet"
)
OUTPUT_JSON_PATH = PROJECT_ROOT / "constants" / "open_llm_leaderboard.json"
MERGE_BATCH_SIZE = 200

BENCHMARK_COLUMNS = {
    "ifeval": "IFEval",
    "bbh": "BBH",
    "math_lvl5": "MATH Lvl 5",
    "gpqa": "GPQA",
    "musr": "MUSR",
    "mmlu_pro": "MMLU-PRO",
}


def log(message):
    print(message, flush=True)


def fetch_open_llm_leaderboard_rows():
    log(f"FETCH {OPEN_LLM_PARQUET_URL}")
    dataframe = pd.read_parquet(OPEN_LLM_PARQUET_URL)
    log(f"Rows downloaded: {len(dataframe)}")

    if dataframe.empty:
        return []

    best_row_per_model = (
        dataframe.sort_values("Average ⬆️", ascending=False)
        .groupby("fullname", as_index=False)
        .first()
    )

    leaderboard_rows = []
    for _, row in best_row_per_model.iterrows():
        model_name = str(row["fullname"]).strip()
        if not model_name or model_name == "nan":
            continue

        benchmarks = {}
        for benchmark_key, column_name in BENCHMARK_COLUMNS.items():
            value = row.get(column_name)
            if pd.notna(value):
                benchmarks[benchmark_key] = round(float(value), 4)

        average_value = row.get("Average ⬆️")
        if pd.isna(average_value):
            continue

        params_value = row.get("#Params (B)")
        params_b = round(float(params_value), 4) if pd.notna(params_value) else None

        submission_date = row.get("Submission Date")
        submission_date_text = (
            str(submission_date)[:10]
            if pd.notna(submission_date)
            else None
        )

        hub_license = row.get("Hub License")
        license_name = str(hub_license) if pd.notna(hub_license) else None

        leaderboard_rows.append({
            "model_name": model_name,
            "average": round(float(average_value), 4),
            "benchmarks": benchmarks,
            "params_b": params_b,
            "license": license_name,
            "submission_date": submission_date_text,
        })

    leaderboard_rows.sort(key=lambda entry: entry["average"], reverse=True)
    for rank, entry in enumerate(leaderboard_rows, start=1):
        entry["rank"] = rank

    log(f"Unique models in leaderboard: {len(leaderboard_rows)}")
    return leaderboard_rows


def build_json_payload(leaderboard_rows):
    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    models_payload = {}
    for entry in leaderboard_rows:
        models_payload[entry["model_name"]] = {
            "average": entry["average"],
            "rank": entry["rank"],
            "benchmarks": entry["benchmarks"],
            "params_b": entry["params_b"],
            "license": entry["license"],
            "submission_date": entry["submission_date"],
        }

    return {
        "meta": {
            "source": "open-llm-leaderboard/contents",
            "fetched_at": fetched_at,
            "model_count": len(models_payload),
        },
        "models": models_payload,
    }


def write_open_llm_leaderboard_json(json_payload):
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_PATH.write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log(f"CACHE {OUTPUT_JSON_PATH}")


def merge_open_llm_leaderboard_to_neo4j(driver, leaderboard_rows):
    merge_query = """
    UNWIND $batch AS row
    MATCH (m:Model {name: row.model_name})
    SET m.oll_average = row.average,
        m.oll_rank = row.rank,
        m.oll_ifeval = row.oll_ifeval,
        m.oll_bbh = row.oll_bbh,
        m.oll_math_lvl5 = row.oll_math_lvl5,
        m.oll_gpqa = row.oll_gpqa,
        m.oll_musr = row.oll_musr,
        m.oll_mmlu_pro = row.oll_mmlu_pro,
        m.oll_params_b = row.params_b,
        m.oll_submission_date = row.submission_date,
        m.oll_synced_at = timestamp()
    RETURN count(m) AS merged_count
  """

    merged_total = 0

    with driver.session() as session:
        for batch_start in range(0, len(leaderboard_rows), MERGE_BATCH_SIZE):
            batch = leaderboard_rows[batch_start : batch_start + MERGE_BATCH_SIZE]
            neo4j_batch = []
            for entry in batch:
                benchmarks = entry["benchmarks"]
                neo4j_batch.append({
                    "model_name": entry["model_name"],
                    "average": entry["average"],
                    "rank": entry["rank"],
                    "oll_ifeval": benchmarks.get("ifeval"),
                    "oll_bbh": benchmarks.get("bbh"),
                    "oll_math_lvl5": benchmarks.get("math_lvl5"),
                    "oll_gpqa": benchmarks.get("gpqa"),
                    "oll_musr": benchmarks.get("musr"),
                    "oll_mmlu_pro": benchmarks.get("mmlu_pro"),
                    "params_b": entry["params_b"],
                    "submission_date": entry["submission_date"],
                })

            result = session.run(merge_query, batch=neo4j_batch).single()
            merged_total += result["merged_count"]

    not_in_graph = len(leaderboard_rows) - merged_total
    log(f"[OK] MERGE Neo4j: {merged_total} models updated")
    if not_in_graph:
        log(f"[WARN] {not_in_graph} leaderboard models not found in graph")

    return merged_total, not_in_graph


def main():
    log("=== enrich_open_llm_leaderboard.py ===")

    leaderboard_rows = fetch_open_llm_leaderboard_rows()
    if not leaderboard_rows:
        log("[FAIL] No leaderboard rows fetched")
        raise SystemExit(1)

    json_payload = build_json_payload(leaderboard_rows)
    write_open_llm_leaderboard_json(json_payload)

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    try:
        merge_open_llm_leaderboard_to_neo4j(driver, leaderboard_rows)
    finally:
        driver.close()

    log("\n=== COMPLETATO ===")


if __name__ == "__main__":
    main()
