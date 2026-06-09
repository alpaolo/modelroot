import sys
from pathlib import Path

import requests
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".env"))
import config as env

URI = env.NEO4J_URI
AUTH = env.NEO4J_AUTH

def enrich_models_with_papers():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    # 1. Estraiamo i modelli più importanti che non hanno ancora una relazione scientifica
    query_get_models = """
    MATCH (m:Model)
    WHERE NOT (m)-[:BASED_ON_PAPER]->()
    RETURN m.name AS model_id
    ORDER BY m.downloads DESC
    LIMIT 200
    """
    
    with driver.session() as session:
        result = session.run(query_get_models)
        models = [record["model_id"] for record in result]
    
    print(f"Trovati {len(models)} modelli da analizzare su Hugging Face...")
    
    # 2. Scansione su Hugging Face API
    for model_id in models:
        url = f"https://huggingface.co/api/models/{model_id}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Cerchiamo i paper_ids nel JSON di Hugging Face
                paper_ids = data.get("paper_ids", [])
                
                # Se non sono lì, proviamo a cercarli nei tag generici
                if not paper_ids:
                    tags = data.get("tags", [])
                    paper_ids = [t.split("arxiv:")[1] for t in tags if isinstance(t, str) and t.startswith("arxiv:")]
                
                if paper_ids:
                    print(f"-> Modello {model_id} collegato ai paper: {paper_ids}")
                    
                    # 3. Scrittura chirurgica nel Grafo per ogni paper trovato
                    for p_id in paper_ids:
                        # Estraiamo l'anno in modo pulito dai primi due caratteri dell'ID arXiv (es: 2407.123 -> 2024)
                        try:
                            year = int("20" + p_id.split(".")[0][:2])
                        except:
                            year = None
                            
                        query_write_paper = """
                        MATCH (m:Model {name: $model_id})
                        MERGE (p:Paper {id: $paper_id})
                        ON CREATE SET 
                            p.url = "https://arxiv.org/abs/" + $paper_id,
                            p.year = $year,
                            p.title = "In Arricchimento"
                        MERGE (m)-[:BASED_ON_PAPER]->(p)
                        """
                        with driver.session() as session:
                            session.run(query_write_paper, model_id=model_id, paper_id=p_id, year=year)
            else:
                print(f"Errore API per {model_id}: Stato {response.status_code}")
        except Exception as e:
            print(f"Errore durante l'elaborazione di {model_id}: {e}")
            
    driver.close()
    print("Arricchimento Scientifico completato!")

if __name__ == "__main__":
    enrich_models_with_papers()