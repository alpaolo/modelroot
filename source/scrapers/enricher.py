import os
from neo4j import GraphDatabase
from huggingface_hub import model_info

# Configurazione connessione a Neo4j (usa le tue credenziali attuali)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "y+8B0fxIcrist"  # <--- METTI LA TUA PASSWORD QUI

def get_advanced_model_data(model_id):
    """Interroga Hugging Face per estrarre licenza, dataset e paper ArXiv"""
    try:
        print(f"🔄 Controllo metadati su Hugging Face per: {model_id}...")
        info = model_info(model_id)
        card_data = info.card_data if info.card_data else {}
        
        # 1. Licenza
        license_type = card_data.get("license", "unknown")
        
        # 2. Dataset utilizzati
        datasets = card_data.get("datasets", [])
        if isinstance(datasets, str):
            datasets = [datasets]
        elif datasets is None:
            datasets = []
            
        # 3. ID Paper ArXiv
        arxiv_ids = []
        # Cerca nelle informazioni dei transformers
        if hasattr(info, 'transformers_info') and info.transformers_info:
            arxiv_ids = info.transformers_info.get("arxiv", [])
        
        # Se vuoto, cerca nei tag generici dello YAML
        if not arxiv_ids and 'arxiv' in card_data:
            raw_arxiv = card_data['arxiv']
            if isinstance(raw_arxiv, list):
                arxiv_ids = raw_arxiv
            else:
                arxiv_ids = [raw_arxiv]
                
        # Pulizia degli ID (rimuoviamo eventuali spazi o formati strani)
        clean_arxiv_ids = []
        for a_id in arxiv_ids:
            if isinstance(a_id, str):
                # Se l'id contiene l'url intero, estraiamo solo la parte finale
                a_id = a_id.split('/')[-1]
            clean_arxiv_ids.append(str(a_id).strip())

        return {
            "license": license_type,
            "datasets": datasets,
            "arxiv_ids": clean_arxiv_ids
        }
    except Exception as e:
        print(f"⚠️ Impossibile recuperare dati per {model_id}: {e}")
        return {"license": "unknown", "datasets": [], "arxiv_ids": []}

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        # 1. Recupera la lista dei modelli attualmente presenti nel database
        print("🔍 Leggo i modelli presenti nel database...")
        result = session.run("MATCH (m:Model) RETURN m.name AS name")
        models = [record["name"] for record in result]
        print(f"📋 Trovati {len(models)} modelli da arricchire.\n")
        
        # 2. Ciclo su ogni modello per estrarre e salvare i dati avanzati
        for model_id in models:
            data = get_advanced_model_data(model_id)
            
            # Query Cypher per iniettare i nuovi dati e creare le relazioni
            update_query = """
            MATCH (m:Model {name: $model_id})
            SET m.license = $license
            
            FOREACH (d_name IN $datasets |
                MERGE (d:Dataset {name: d_name})
                MERGE (m)-[:USED_DATASET]->(d)
            )
            
            FOREACH (arxiv_id IN $arxiv_ids |
                MERGE (p:Paper {id: arxiv_id})
                ON CREATE SET p.url = "https://arxiv.org/abs/" + arxiv_id
                MERGE (m)-[:CITED_IN]->(p)
            )
            """
            
            session.run(update_query, 
                        model_id=model_id, 
                        license=data["license"], 
                        datasets=data["datasets"], 
                        arxiv_ids=data["arxiv_ids"])
            
            print(f"✅ Modello {model_id} aggiornato! (Licenza: {data['license']}, Dataset: {len(data['datasets'])}, Paper: {len(data['arxiv_ids'])})\n")
            
    driver.close()
    print("🎉 Arricchimento completato con successo!")

if __name__ == "__main__":
    main()