import sys
import time
from pathlib import Path

import requests
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".env"))
import config as env

NEO4J_URI = env.NEO4J_URI
NEO4J_AUTH = env.NEO4J_AUTH

def get_real_license_from_hf(model_id):
    """Interroga l'API del singolo modello per avere il JSON completo al 100%"""
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            card_data = data.get("cardData", {})
            
            # Hugging Face può salvare la licenza in modi diversi nel JSON completo
            license_type = card_data.get("license") or data.get("license")
            if isinstance(license_type, list):
                license_type = license_type[0] if license_type else "unknown"
            return str(license_type).strip() if license_type else "unknown"
        elif response.status_code == 404:
            return "deleted_or_private"
    except Exception as e:
        print(f"⚠️ Errore durante la chiamata per {model_id}: {e}")
    return "unknown"

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    
    # 1. Estraiamo i primi 200 modelli più scaricati che sono ancora "Unknown"
    # Facciamo blocchi da 200 alla volta per testare ed evitare ban da HF
    get_models_query = """
    MATCH (m:Model)-[:UNDER_LICENSE]->(l:License {name: "Unknown / Unspecified"})
    RETURN m.name AS model_id
    ORDER BY m.downloads DESC
    LIMIT 200
    """
    
    # Query di aggiornamento: stacca dal vecchio Unknown e attacca alla licenza reale
    update_query = """
    MATCH (m:Model {name: $model_id})-[r:UNDER_LICENSE]->(:License)
    DELETE r
    WITH m
    MERGE (l:License {name: $license_name})
    MERGE (m)-[:UNDER_LICENSE]->(l)
    """
    
    print("🔍 Recupero modelli da arricchire dal database...")
    with driver.session() as session:
        result = session.run(get_models_query)
        models_to_enrich = [record["model_id"] for record in result]
        
        if not models_to_enrich:
            print("🎉 Tutti i modelli nel DB hanno già una licenza definita!")
            return

        print(f"🚀 Inizio carotaggio su Hugging Face per {len(models_to_enrich)} modelli...")
        
        for idx, model_id in enumerate(models_to_enrich, 1):
            # Chiamata a Hugging Face
            real_license = get_real_license_from_hf(model_id)
            
            # Se troviamo una licenza valida e diversa da unknown, aggiorniamo il grafo
            if real_license and real_license != "unknown":
                session.run(update_query, model_id=model_id, license_name=real_license)
                print(f"✅ [{idx}/{len(models_to_enrich)}] {model_id} -> Aggiornato a licenza: '{real_license}'")
            else:
                print(f"⏳ [{idx}/{len(models_to_enrich)}] {model_id} -> Rimasto Sconosciuto (Licenza non trovata)")
            
            # Un piccolo sleep per essere gentili con i server di Hugging Face
            time.sleep(0.5)

    driver.close()
    print("🏁 Operazione di arricchimento completata!")

if __name__ == "__main__":
    main()