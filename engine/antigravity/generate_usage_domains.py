import json
import csv
import os
import re
from collections import defaultdict

RAW_DATA_DIR = "raw_data"
OUTPUT_DIR = "output"

# Definizione dei domini di utilizzo (Usage Domains) e relative parole chiave
USAGE_DOMAINS = {
    "Medicine / Healthcare": [
        "medical", "clinical", "biomedical", "health", "pathology",
        "radiology", "healthcare", "disease", "diagnosis", "patient",
        "oncology", "cardiology", "dermatology", "ophthalmology",
        "dental", "ehr", "electronic-health", "icd", "snomed",
        "pubmed", "medqa", "medbench", "mri", "x-ray", "ct-scan"
    ],
    "Science / Research": [
        "biology", "chemistry", "physics", "protein", "genomics",
        "genome", "dna", "rna", "molecule", "drug", "materials",
        "scientific", "science", "bioinformatics", "astronomy",
        "climate", "earth-science", "geoscience", "ecology",
        "neuroscience", "proteomics", "metabolomics", "math", "mathematics",
        "theorem", "material-science"
    ],
    "Code / Programming": [
        "code", "programming", "sql", "code-generation", "codegen",
        "coder", "starcoder", "codellama", "deepseek-coder",
        "copilot", "autocomplete", "github", "python", "java", "c++",
        "javascript", "html", "bash", "developer"
    ],
    "Finance / Economics": [
        "finance", "financial", "trading", "stock", "banking",
        "finbert", "investment", "economic", "forex", "crypto",
        "sec", "10-k", "10-q", "accounting"
    ],
    "Legal / Law": [
        "legal", "law", "contract", "court", "regulation",
        "compliance", "legislative", "juridical", "lawyer", "paralegal"
    ],
    "Education / Tutoring": [
        "education", "tutoring", "academic", "student", "exam",
        "quiz", "learning", "textbook", "educational", "teacher",
        "classroom"
    ],
    "Robotics / Embodied AI": [
        "robotics", "robot", "control", "navigation", "manipulation",
        "embodied", "sim2real", "mujoco", "isaac", "ros"
    ],
    "CyberSecurity": [
        "security", "cyber", "cybersecurity", "malware", "phishing",
        "vulnerability", "exploit", "pentest", "hacker"
    ],
    "Art / Design": [
        "art", "design", "drawing", "painting", "fashion", "interior-design",
        "logo", "anime", "manga", "comic", "artwork", "illustration"
    ],
    "Entertainment / Gaming": [
        "game", "gaming", "rpg", "npc", "movie", "entertainment", "music",
        "song", "audio-effects", "storytelling"
    ],
    "Automotive / Autonomous Driving": [
        "autonomous", "driving", "lidar", "automotive", "self-driving",
        "vehicle", "adas"
    ]
}

def make_id_safe(name):
    """Converte un nome in un ID sicuro (minuscolo, senza spazi o caratteri speciali)."""
    s = name.lower()
    s = re.sub(r'[^a-z0-9]', '_', s)
    s = re.sub(r'_+', '_', s)
    return s.strip('_')

def get_usage_domains(model):
    """Classifica il modello nei domini di utilizzo in base a ID e tags."""
    raw_tags = model.get("tags", [])
    model_id_lower = model.get("id", "").lower()

    # Filtra i tag prefissati (come abbiamo fatto per quelli tecnici)
    SKIP_PREFIXES = (
        "dataset:", "arxiv:", "base_model:", "region:", "deploy:",
        "base_model:quantized:", "license:", "doi:",
    )
    filtered_tags = set()
    for t in raw_tags:
        t_lower = t.lower()
        if not any(t_lower.startswith(p) for p in SKIP_PREFIXES):
            filtered_tags.add(t_lower)

    domains = []
    
    for domain_name, keywords in USAGE_DOMAINS.items():
        matched = False
        for kw in keywords:
            # Per parole corte o generiche richiediamo match esatto o bound sui bordi dell'ID
            if len(kw) <= 4 or kw in ("code", "drug", "art", "law"):
                if kw in filtered_tags or re.search(r'(?:^|[\-_./])' + re.escape(kw) + r'(?:$|[\-_./])', model_id_lower):
                    matched = True
                    break
            else:
                if any(kw in tag for tag in filtered_tags) or kw in model_id_lower:
                    matched = True
                    break
        if matched:
            domains.append(domain_name)
            
    # Se nessun dominio di utilizzo è trovato, lo classifichiamo come Generale
    if not domains:
        domains.append("General Purpose / Undefined")
        
    return domains

def main():
    cache_file = os.path.join(RAW_DATA_DIR, "raw_models.json")
    if not os.path.exists(cache_file):
        print(f"Errore: File {cache_file} non trovato. Esegui prima lo script principale.")
        return

    print(">>> Caricamento modelli raw dalla cache...")
    with open(cache_file, "r", encoding="utf-8") as f:
        models = json.load(f)
    print(f"    {len(models):,} modelli caricati.")

    print("\n>>> Estrazione dei domini di utilizzo...")
    
    usage_domains_set = set()
    model_usage_rels = [] # list of (model_id, domain_name)
    
    for model in models:
        model_id = model.get("id")
        if not model_id: continue
        
        domains = get_usage_domains(model)
        for d in domains:
            usage_domains_set.add(d)
            model_usage_rels.append((model_id, d))
            
    print(f"    Trovati {len(usage_domains_set)} domini di utilizzo unici.")
    print(f"    Trovate {len(model_usage_rels):,} relazioni modello-dominio_utilizzo.")

    # Creazione directory
    nodes_dir = os.path.join(OUTPUT_DIR, "nodes")
    rels_dir = os.path.join(OUTPUT_DIR, "relationships")
    os.makedirs(nodes_dir, exist_ok=True)
    os.makedirs(rels_dir, exist_ok=True)
    
    # 1. Scrittura Nodi: Usage Domains
    nodes_file = os.path.join(nodes_dir, "nodes_usage_domains.csv")
    with open(nodes_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["usageDomainId:ID(UsageDomain)", "name:string", ":LABEL"])
        for d in sorted(list(usage_domains_set)):
            writer.writerow([make_id_safe(d), d, "UsageDomain"])
    print(f"\n[OK] Generato: {nodes_file}")

    # 2. Scrittura Relazioni: Model -> Usage Domain
    rels_file = os.path.join(rels_dir, "rels_model_usage_domain.csv")
    with open(rels_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([":START_ID(Model)", ":END_ID(UsageDomain)", ":TYPE"])
        for m_id, d in model_usage_rels:
            writer.writerow([m_id, make_id_safe(d), "USED_FOR_DOMAIN"])
    print(f"[OK] Generato: {rels_file}")

    # Statistiche finali
    print("\n" + "=" * 50)
    print("DISTRIBUZIONE DOMINI DI UTILIZZO")
    print("=" * 50)
    
    counts = defaultdict(int)
    for _, d in model_usage_rels:
        counts[d] += 1
        
    for d, c in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {d:35s} {c:>6,} modelli")
        
if __name__ == "__main__":
    main()
