import sys
import time
from pathlib import Path

from huggingface_hub import HfApi
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".env"))
import config as env

NEO4J_URI = env.NEO4J_URI
NEO4J_AUTH = env.NEO4J_AUTH

def main():
    print("🚀 Avvio dello Scraper Massivo Centripeto (Target: 10.000 modelli)...")
    start_time = time.time()
    
    # 1. Connessione a Hugging Face
    api = HfApi()
    print("📥 Scaricamento dati da Hugging Face (richiesta massiva)...")
    
    models_chunk = api.list_models(
        sort="downloads", 
        limit=10000,
        full=True, 
        fetch_config=False
    )
    
    hf_time = time.time() - start_time
    print(f"✅ Dati Hugging Face scaricati in {hf_time:.2f} secondi. Inizio inserimento in Neo4j...")

    # 2. Connessione a Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    
    # Query ottimizzata secondo la logica del tuo grafo
    query = """
    UNWIND $batch AS item
    
    // 1. Gestione Modello e Brand (Logica Passiva: Modello -> Brand)
    MERGE (brand:MainBrand {name: item.author})
    MERGE (m:Model {name: item.model_id})
    ON CREATE SET m.created_at = timestamp()
    SET m.downloads = item.downloads,
        m.trending_downloads = item.trending_downloads
    MERGE (m)-[:PUBLISHED_BY]->(brand)
    
    // 2. Gestione della Licenza come Nodo Autonomo
    MERGE (lic:License {name: item.license})
    MERGE (m)-[:UNDER_LICENSE]->(lic)
    
    // 3. Gestione della Derivazione (Satelliti GGUF/AWQ che puntano al Modello Base)
    FOREACH (base IN CASE WHEN item.base_model IS NOT NULL THEN [item.base_model] ELSE [] END |
        MERGE (bm:Model {name: base})
        MERGE (m)-[:DERIVED_FROM]->(bm)
    )
    
    // 4. Gestione Task
    FOREACH (t IN CASE WHEN item.task IS NOT NULL THEN [item.task] ELSE [] END |
        MERGE (pipeline:Task {name: t})
        MERGE (m)-[:PERFORMS]->(pipeline)
    )
    
    // 5. Gestione Dataset Collegati
    FOREACH (d_name IN item.datasets |
        MERGE (d:Dataset {name: d_name})
        MERGE (m)-[:USED_DATASET]->(d)
    )
    
    // 6. Gestione Paper ArXiv Collegati
    FOREACH (arxiv_id IN item.arxiv_ids |
        MERGE (p:Paper {id: toString(arxiv_id)})
        ON CREATE SET p.url = "https://arxiv.org/abs/" + toString(arxiv_id)
        MERGE (m)-[:CITED_IN]->(p)
    )
    """

    batch = []
    count = 0
    
    with driver.session() as session:
        for model in models_chunk:
            model_id = model.modelId
            
            if "/" in model_id:
                author = model_id.split("/")[0]
            else:
                author = "Unknown_Author"
                
            downloads = getattr(model, "downloads", 0)
            trending_downloads = getattr(model, "trending_downloads", 0)
            task = getattr(model, "pipeline_tag", None)
            
            # Estrazione Licenza
            card_data = model.cardData if model.cardData else {}
            license_type = card_data.get("license", "unknown")
            if isinstance(license_type, list): 
                license_type = license_type[0] if license_type else "unknown"
            
            # Estrazione Modello Base (Se esiste, serve per il link DERIVED_FROM)
            base_model = card_data.get("base_model", None)
            if isinstance(base_model, list):
                base_model = base_model[0] if base_model else None
            if base_model == model_id: # Evita loop infiniti di un modello derivato da se stesso
                base_model = None
            
            # Pulizia Dataset
            datasets = card_data.get("datasets", [])
            if isinstance(datasets, str): datasets = [datasets]
            elif datasets is None: datasets = []
            datasets = [str(d)[:50] for d in datasets if d] 
            
            # Estrazione ArXiv
            arxiv_ids = []
            if hasattr(model, 'transformers_info') and model.transformers_info:
                arxiv_ids = model.transformers_info.get("arxiv", [])
            if not arxiv_ids and 'arxiv' in card_data:
                raw_arxiv = card_data['arxiv']
                arxiv_ids = raw_arxiv if isinstance(raw_arxiv, list) else [raw_arxiv]
            
            clean_arxiv = []
            for a_id in arxiv_ids:
                if a_id:
                    a_id = str(a_id).split('/')[-1].strip()
                    if len(a_id) < 20: clean_arxiv.append(a_id)

            # Costruiamo il dizionario per il batch
            batch.append({
                "author": author,
                "model_id": model_id,
                "downloads": downloads,
                "trending_downloads": trending_downloads,
                "task": task,
                "license": str(license_type).strip(),
                "base_model": base_model,
                "datasets": datasets,
                "arxiv_ids": clean_arxiv
            })
            
            count += 1
            
            # Scrittura a blocchi
            if len(batch) >= 500:
                session.run(query, batch=batch)
                batch = []
                print(f"⏳ Inseriti {count}/10000 modelli nel database...")

        # Ultimo invio per i rimanenti
        if batch:
            session.run(query, batch=batch)

    driver.close()
    total_time = time.time() - start_time
    print(f"🎉 Spettacolo! {count} modelli importati rispettando lo schema centripeto in {total_time:.2f} secondi!")

if __name__ == "__main__":
    main()