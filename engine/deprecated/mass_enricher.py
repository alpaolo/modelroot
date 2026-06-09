import time
from huggingface_hub import model_info
from neo4j import GraphDatabase

# Configurazione Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "y+8B0fxIcrist"  # <--- METTI LA TUA PASSWORD

def main():
    print("🔄 Avvio dell'Enricher di Massa...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # 1. Prendiamo i primi 300 modelli più caldi in trend che sono ancora 'unknown'
    # Ottimizziamo le risorse partendo da quelli più rilevanti sul mercato!
    find_query = """
    MATCH (m:Model)
    WHERE m.license = 'unknown'
    RETURN m.name AS name
    ORDER BY m.trending_downloads DESC
    LIMIT 300
    """
    
    with driver.session() as session:
        result = session.run(find_query)
        models_to_enrich = [record["name"] for record in result]
        
    print(f"📋 Trovati {len(models_to_enrich)} modelli prioritari da arricchire.")
    
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
    
    count = 0
    with driver.session() as session:
        for model_id in models_to_enrich:
            try:
                # Chiamata mirata al singolo modello (questa restituisce SEMPRE lo YAML completo)
                info = model_info(model_id)
                card_data = info.card_data if info.card_data else {}
                
                # Licenza
                license_type = card_data.get("license", "unknown")
                if isinstance(license_type, list):
                    license_type = license_type[0] if license_type else "unknown"
                
                # Dataset
                datasets = card_data.get("datasets", [])
                if isinstance(datasets, str): datasets = [datasets]
                elif datasets is None: datasets = []
                datasets = [str(d)[:50] for d in datasets if d]
                
                # Paper ArXiv
                arxiv_ids = []
                if hasattr(info, 'transformers_info') and info.transformers_info:
                    arxiv_ids = info.transformers_info.get("arxiv", [])
                if not arxiv_ids and 'arxiv' in card_data:
                    raw_arxiv = card_data['arxiv']
                    arxiv_ids = raw_arxiv if isinstance(raw_arxiv, list) else [raw_arxiv]
                
                clean_arxiv = []
                for a_id in arxiv_ids:
                    if a_id:
                        a_id = str(a_id).split('/')[-1].strip()
                        if len(a_id) < 20: clean_arxiv.append(a_id)
                
                # Invio dati a Neo4j
                session.run(update_query, 
                            model_id=model_id, 
                            license=str(license_type), 
                            datasets=datasets, 
                            arxiv_ids=clean_arxiv)
                
                count += 1
                if count % 50 == 0:
                    print(f"⏳ Arricchiti {count}/{len(models_to_enrich)} modelli...")
                
                # Micro-pausa protettiva per evitare il rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                # Se un modello è privato o è stato rimosso, gli cambiamo lo stato per non cercarlo più
                session.run("MATCH (m:Model {name: $model_id}) SET m.license = 'unavailable'", model_id=model_id)
                continue

    driver.close()
    print(f"🎉 Fatto! {count} modelli di punta arricchiti con successo!")

if __name__ == "__main__":
    main()