import os
from langchain_community.graphs import Neo4jGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

def init_cypher_chain(uri, username, password, google_api_key, model_name="gemini-1.5-flash"):
    """
    Inizializza e restituisce la catena LangChain per Text-to-Cypher ad alta precisione
    utilizzando Google Gemini e Neo4j.
    """
    # Imposta l'API key nell'ambiente (richiesto da langchain_google_genai)
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # 1. Connessione a Neo4j (LangChain estrarrà automaticamente lo schema)
    graph = Neo4jGraph(
        url=uri,
        username=username,
        password=password
    )

    # 2. Inizializzazione del modello LLM (Gemini)
    # Impostiamo temperature=0 per avere risposte deterministiche e logiche
    llm = ChatGoogleGenerativeAI(
        model=model_name, 
        temperature=0,
        max_output_tokens=2048
    )

    # 3. Prompting Ingegnerizzato (Few-Shot)
    # Spieghiamo all'LLM le regole del nostro dominio specifico per massimizzare la precisione.
    CYPHER_GENERATION_TEMPLATE = """Sei un esperto di database a grafo Neo4j e linguaggio Cypher.
Il tuo compito è tradurre una domanda in linguaggio naturale in una query Cypher valida.

Usa ESCLUSIVAMENTE i tipi di relazione e le proprietà forniti nello Schema.
NON inventare nodi o relazioni.

Schema del Database:
{schema}

Regole specifiche per questo database HuggingFace:
1. Il nodo 'Model' ha proprietà come 'name', 'downloads' (long), 'likes' (int), 'pipeline_tag' (string).
2. 'Model' è connesso a 'License' tramite LICENSED_AS. Esempio: (m:Model)-[:LICENSED_AS]->(l:License {{name: "apache-2.0"}})
3. 'Model' è connesso a 'Task' tramite PERFORMS_TASK. Esempio: (m:Model)-[:PERFORMS_TASK]->(t:Task {{name: "text-generation"}})
4. 'Model' è connesso a 'UsageDomain' tramite USED_FOR_DOMAIN.
5. Quando l'utente chiede modelli "Embedding", cerca tramite PERFORMS_TASK i Task chiamati "feature-extraction" o "sentence-similarity".
6. Quando l'utente chiede modelli "LLM", cerca tramite PERFORMS_TASK il Task "text-generation".
7. Quando l'utente fa riferimento a "classifiche" (es. MTEB o Open LLM) o chiede i "migliori", usa la proprietà 'downloads' (o 'likes') in ordine decrescente (ORDER BY m.downloads DESC) come proxy per il ranking, usando LIMIT. "Top 10%" o simili implica prendere i primissimi risultati (es. LIMIT 5 o 10).

Esempi di query:

Domanda: Trova la migliore combinazione possibile: un modello di Embedding (dalla classifica MTEB) e un LLM (dalla classifica Open LLM) che abbiano ENTRAMBI licenza Apache-2.0 e che siano nella top 10% delle rispettive classifiche.
Cypher Query:
// Poiché vogliamo due tipi di modelli diversi in un'unica risposta, eseguiamo due ricerche separate e uniamo i risultati
CALL {{
  MATCH (llm:Model)-[:LICENSED_AS]->(l1:License) WHERE l1.name = "apache-2.0"
  MATCH (llm)-[:PERFORMS_TASK]->(t1:Task) WHERE t1.name = "text-generation"
  RETURN llm.name AS best_llm, llm.downloads AS llm_downloads
  ORDER BY llm.downloads DESC LIMIT 1
}}
CALL {{
  MATCH (emb:Model)-[:LICENSED_AS]->(l2:License) WHERE l2.name = "apache-2.0"
  MATCH (emb)-[:PERFORMS_TASK]->(t2:Task) WHERE t2.name IN ["feature-extraction", "sentence-similarity"]
  RETURN emb.name AS best_emb, emb.downloads AS emb_downloads
  ORDER BY emb.downloads DESC LIMIT 1
}}
RETURN best_llm, llm_downloads, best_emb, emb_downloads

Domanda: {question}
Cypher Query:"""

    cypher_prompt = PromptTemplate(
        input_variables=["schema", "question"], 
        template=CYPHER_GENERATION_TEMPLATE
    )

    # 4. Creazione della Catena
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        cypher_prompt=cypher_prompt,
        validate_cypher=True,           # Tenta di auto-correggere eventuali errori di sintassi Cypher
        return_intermediate_steps=True, # Utile per recuperare la query Cypher generata nell'interfaccia UI
        allow_dangerous_requests=True   # Necessario in versioni recenti di LangChain per abilitare lettura DB
    )
    
    return chain

def query_database(chain, question_text):
    """
    Invia la domanda alla catena e restituisce un dizionario contenente:
    - result: la risposta testuale formulata dall'IA
    - intermediate_steps: lista di step (utile per estrarre la query Cypher generata)
    """
    try:
        response = chain.invoke({"query": question_text})
        return response
    except Exception as e:
        # In caso di query incomprensibile o errori DB
        return {
            "result": f"Mi dispiace, si è verificato un errore durante l'interrogazione: {str(e)}",
            "intermediate_steps": []
        }
