import sys
from pathlib import Path

from huggingface_hub import HfApi
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".env"))
import config as env

URI = env.NEO4J_URI
AUTH = env.NEO4J_AUTH

def scraper_organico():
    api = HfApi()
    print("🔄 Estrazione dei primi 100 modelli più popolari da Hugging Face...")
    
    # Ora è pulito e compatibile con la tua versione di huggingface_hub
    models = api.list_models(sort="downloads", limit=100)
    
    # Connessione a Neo4j (il resto del codice rimane identico)
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            
           # Query aggiornata: usiamo 'name' così Neo4j lo mostra di default!
            query = """
            // 1. Gestione dell'Autore
            MERGE (a:Author {name: $author})
            
            // 2. Gestione del Modello (usiamo name per fregare il browser)
            MERGE (m:Model {name: $model_id})
            ON CREATE SET m.created_at = timestamp()
            SET m.downloads = $downloads,
                m.likes = $likes
            
            // Relazione Autore -> Modello
            MERGE (a)-[:CREATED]->(m)
            
            // 3. Gestione del Task
            FOREACH (t IN CASE WHEN $task IS NOT NULL THEN [$task] ELSE [] END |
                MERGE (pipeline:Task {name: t})
                MERGE (m)-[:PERFORMS]->(pipeline)
            )
            """
            
            contatore_successi = 0
            
            for model in models:
                # Se l'autore manca, lo ricaviamo dallo split del model_id (es. "Qwen/Qwen2.5" -> "Qwen")
                if model.author:
                    author_name = model.author
                elif "/" in model.id:
                    author_name = model.id.split("/")[0]
                else:
                    author_name = "Independent" # Per i modelli senza organizzazione
                
                pipeline_tag = model.pipeline_tag if model.pipeline_tag else "Unspecified"
                downloads = model.downloads if model.downloads else 0
                likes = model.likes if model.likes else 0
                
                # Esecuzione della query
                result = session.run(
                    query, 
                    author=str(author_name), 
                    model_id=str(model.id),
                    task=str(pipeline_tag),
                    downloads=int(downloads),
                    likes=int(likes)
                )
                
                summary = result.consume()
                if summary.counters.nodes_created > 0 or summary.counters.properties_set > 0:
                    contatore_successi += 1
                    
            print(f"🎉 Elaborazione completata! Aggiornati con successo {contatore_successi}/100 modelli nel grafo.")

if __name__ == "__main__":
    scraper_organico()