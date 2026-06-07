"""
Fase A — arricchimento link su Model:
- hf_url: pagina Hugging Face (derivabile, senza API)
- license_link: URL documento licenza da cardData (solo navigazione DS)
- REQUEST_DELAY_SECONDS=1.0 per ridurre errori 429 su API HF
"""
import time

import requests
from neo4j import GraphDatabase

from neo4j_config import NEO4J_AUTH as AUTH, NEO4J_URI as URI

HF_MODEL_BASE_URL = "https://huggingface.co/"
REQUEST_DELAY_SECONDS = 1.0
PROGRESS_SUMMARY_EVERY = 50


def log(message):
    print(message, flush=True)


def set_hf_urls_on_all_models(driver):
    """Imposta hf_url su tutti i Model che non ce l'hanno ancora."""
    with driver.session() as session:
        pending_count = session.run(
            "MATCH (m:Model) WHERE m.hf_url IS NULL RETURN count(m) AS c"
        ).single()["c"]
        already_set_count = session.run(
            "MATCH (m:Model) WHERE m.hf_url IS NOT NULL RETURN count(m) AS c"
        ).single()["c"]

    log(f"[HF_URL] Modelli senza hf_url: {pending_count}")
    log(f"[HF_URL] Modelli gia impostati: {already_set_count}")

    if pending_count == 0:
        log("[HF_URL] Nessun aggiornamento necessario.")
        return 0

    query = """
    MATCH (m:Model)
    WHERE m.hf_url IS NULL
    SET m.hf_url = $hf_base_url + m.name
    RETURN count(m) AS updated_count
    """
    with driver.session() as session:
        result = session.run(query, hf_base_url=HF_MODEL_BASE_URL)
        updated_count = result.single()["updated_count"]

    log(f"[HF_URL] Aggiornati {updated_count} modelli.")
    return updated_count


def fetch_license_link_from_hf(model_id):
    """Legge cardData.license_link dall'API HF."""
    response = requests.get(f"{HF_MODEL_BASE_URL}api/models/{model_id}", timeout=10)
    if response.status_code != 200:
        return None, response.status_code

    card_data = response.json().get("cardData") or {}
    license_link = card_data.get("license_link")
    if not license_link:
        return None, 200

    return str(license_link).strip(), 200


def enrich_license_links(driver):
    """Aggiunge license_link ai Model che ne sono privi."""
    query_all_models = """
    MATCH (m:Model)
    RETURN m.name AS model_id, m.license_link AS license_link
    ORDER BY m.downloads DESC
    """

    update_query = """
    MATCH (m:Model {name: $model_id})
    SET m.license_link = $license_link
    RETURN m.name AS model_id
    """

    log("[LICENSE_LINK] Caricamento modelli dal grafo...")
    with driver.session() as session:
        records = list(session.run(query_all_models))
        model_ids = [
            record["model_id"]
            for record in records
            if not record.get("license_link")
        ]
        already_linked = len(records) - len(model_ids)

    total = len(model_ids)
    log(f"[LICENSE_LINK] Totale modelli nel DB: {len(records)}")
    log(f"[LICENSE_LINK] Gia con license_link: {already_linked}")
    log(f"[LICENSE_LINK] Da analizzare su HF: {total}\n")

    if total == 0:
        log("[LICENSE_LINK] Nessun modello da elaborare.")
        return {"total": 0, "updated": 0, "skipped": 0, "errors": 0}

    updated_count = 0
    skipped_count = 0
    error_count = 0

    for index, model_id in enumerate(model_ids, start=1):
        log(f"[CHECK {index}/{total}] {model_id}")

        try:
            license_link, status_code = fetch_license_link_from_hf(model_id)

            if status_code != 200:
                error_count += 1
                log(f"  [ERRORE API] status {status_code}")
                time.sleep(REQUEST_DELAY_SECONDS)
                continue

            if not license_link:
                skipped_count += 1
                log("  [SKIP] Nessun license_link in cardData")
                time.sleep(REQUEST_DELAY_SECONDS)
                continue

            with driver.session() as session:
                result = session.run(update_query, model_id=model_id, license_link=license_link)
                if result.single():
                    updated_count += 1
                    log(f"  [OK] {license_link}")

        except requests.RequestException as error:
            error_count += 1
            log(f"  [ERRORE] {error}")

        if index % PROGRESS_SUMMARY_EVERY == 0:
            log(
                f"[RIEPILOGO {index}/{total}] "
                f"ok={updated_count} | skip={skipped_count} | errori={error_count}"
            )

        time.sleep(REQUEST_DELAY_SECONDS)

    return {
        "total": total,
        "updated": updated_count,
        "skipped": skipped_count,
        "errors": error_count,
    }


def main():
    log("=== enrich_model_links.py ===\n")
    driver = GraphDatabase.driver(URI, auth=AUTH)

    log("--- Step 1/2: hf_url ---")
    hf_url_count = set_hf_urls_on_all_models(driver)
    log(f"Step 1 completato: {hf_url_count} hf_url scritti.\n")

    log("--- Step 2/2: license_link ---")
    stats = enrich_license_links(driver)
    log(
        f"\n=== COMPLETATO ===\n"
        f"license_link aggiornati: {stats['updated']}\n"
        f"senza link su HF (skip):  {stats['skipped']}\n"
        f"errori API/rete:          {stats['errors']}\n"
        f"modelli analizzati:       {stats['total']}"
    )

    driver.close()


if __name__ == "__main__":
    main()
