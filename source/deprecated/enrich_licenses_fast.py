import concurrent.futures
import requests
from neo4j import GraphDatabase

# Configurazione Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "y+8B0fxIcrist"

# Numero di richieste parallele (25 è un ottimo compromesso per la velocità senza ban)
MAX_WORKERS = 25 

def get_real_license_from_hf(model_id):
    """Interroga l'API del singolo modello in modo rapido"""
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        response = requests.get(url, timeout=5) # Timeout ridotto a 5s per evitare blocchi
        if response.status_code == 200:
            data = response.json()
            card_data = data.get("cardData", {})
            license_type = card_data.get("license") or data.get("license")
            if isinstance(license_type, list):
                license_type = license_type[0] if license_type else "unknown"
            return str(license_type).strip() if license_type else "unknown"
        elif response.status_code == 404:
            return "deleted_or_private"
    except Exception:
        pass
    return "unknown"

def process_model(model_id, driver, update_query):
    """Funzione eseguita in parallelo per ogni singolo modello"""
    real_license = get_real_license_from_hf(model_id)
    
    # Aggiorna il DB solo se troviamo una licenza reale e utile
    if real_license and real_license != "unknown":
        with driver.session() as session:
            session.run(update_query, model_id=model_id, license_name=real_license)
        return f"✅ {model_id} -> {real_license}"
    return None

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Prendiamo TUTTI i modelli rimasti sotto la licenza sconosciuta
    get_models_query = """
    MATCH (m:Model)-[:UNDER_LICENSE]->(l:License {name: "Unknown / Unspecified"})
    RETURN m.name AS model_id
    ORDER BY m.downloads DESC
    """
    
    update_query = """
    MATCH (m:Model {name: $model_id})-[r:UNDER_LICENSE]->(:License)
    DELETE r
    WITH m
    MERGE (l:License {name: $license_name})
    MERGE (m)-[:UNDER_LICENSE]->(l)
    """
    
    print("🔍 Recupero di tutta la flotta rimasta dal database...")
    with driver.session() as session:
        result = session.run(get_models_query)
        models_to_enrich = [record["model_id"] for record in result]
    
    total_models = len(models_to_enrich)
    if total_models == 0:
        print("🎉 Ottimo! Non ci sono più modelli 'Unknown' da elaborare.")
        driver.close()
        return

    print(f"🚀 Avvio scraping in PARALLELO con {MAX_WORKERS} worker su {total_models} modelli...")
    
    # Il motore della velocità: esegue i task in parallelo su più thread
    count_updated = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Lancio tutti i modelli nel pool di esecuzione
        futures = {executor.submit(process_model, m_id, driver, update_query): m_id for m_id in models_to_enrich}
        
        for idx, future in enumerate(concurrent.futures.as_completed(futures), 1):
            res = future.result()
            if res:
                count_updated += 1
            
            # Un feedback visivo ogni 100 modelli per vedere l'avanzamento rapidissimo
            if idx % 100 == 0 or idx == total_models:
                print(f"⚡ Elaborati [{idx}/{total_models}] modelli... (Trovate {count_updated} nuove licenze reali)")

    driver.close()
    print(f"🏁 Fatto! Il database è stato arricchito con {count_updated} licenze specifiche.")

if __name__ == "__main__":
    main()