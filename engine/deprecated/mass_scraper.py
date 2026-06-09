import time
from huggingface_hub import HfApi
from neo4j import GraphDatabase

# Configurazione Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "y+8B0fxIcrist"  # <--- METTI LA TUA PASSWORD

def main():
    print("🚀 Avvio dello Scraper Massivo (Target: 10.000 modelli in Trend)...")
    start_time = time.time()
    
    # 1. Connessione a Hugging Face
    api = HfApi()
    print("📥 Scaricamento dati da Hugging Face (richiesta massiva)...")
    
    # Usiamo sort="downloads" (questo è blindato). 
    # Avendo impostato full=True, ci porteremo comunque a casa i trend di ciascuno!
    models_chunk = api.list_models(
        sort="downloads", 
        limit=10000,
        full=True, 
        fetch_config=False
    )
    
    hf_time = time.time() - start_time
    print(f"✅ Dati Hugging Face scaricati in {hf_time:.2f} secondi. Inizio inserimento in Neo4j...")

    # 2. Connessione a Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    query = """
    UNWIND $batch AS item
    // 1. Gestione Autore e Modello
    MERGE (a:Author {name: item.author})
    MERGE (m:Model {name: item.model_id})
    ON CREATE SET m.created_at = timestamp()
    SET m.downloads = item.downloads,
        m.trending_downloads = item.trending_downloads,
        m.license = item.license
    MERGE (a)-[:CREATED]->(m)
    
    // 2. Gestione Task (se presente)
    FOREACH (t IN CASE WHEN item.task IS NOT NULL THEN [item.task] ELSE [] END |
        MERGE (pipeline:Task {name: t})
        MERGE (m)-[:PERFORMS]->(pipeline)
    )
    
    // 3. Gestione Dataset Collegati
    FOREACH (d_name IN item.datasets |
        MERGE (d:Dataset {name: d_name})
        MERGE (m)-[:USED_DATASET]->(d)
    )
    
    // 4. Gestione Paper ArXiv Collegati
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
            
            card_data = model.cardData if model.cardData else {}
            license_type = card_data.get("license", "unknown")
            if isinstance(license_type, list): 
                license_type = license_type[0] if license_type else "unknown"
            
            datasets = card_data.get("datasets", [])
            if isinstance(datasets, str): datasets = [datasets]
            elif datasets is None: datasets = []
            datasets = [str(d)[:50] for d in datasets if d] 
            
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

            # Accumuliamo i dati nel dizionario del batch
            batch.append({
                "author": author,
                "model_id": model_id,
                "downloads": downloads,
                "trending_downloads": trending_downloads,
                "task": task,
                "license": str(license_type),
                "datasets": datasets,
                "arxiv_ids": clean_arxiv
            })
            
            count += 1
            
            # Inviamo a Neo4j a blocchi di 500 per massimizzare la velocità
            if len(batch) >= 500:
                session.run(query, batch=batch)
                batch = []
                print(f"⏳ Inseriti {count}/10000 modelli nel database...")

        # Inseriamo l'ultimo blocco rimanente
        if batch:
            session.run(query, batch=batch)

    driver.close()
    total_time = time.time() - start_time
    print(f"🎉 Spettacolo! {count} modelli in trend importati con successo in {total_time:.2f} secondi!")

if __name__ == "__main__":
    main()