import sys
from pathlib import Path

import requests
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".env"))
import config as env

URI = env.NEO4J_URI
AUTH = env.NEO4J_AUTH

# Licenze che non devono essere scritte nel grafo (arricchimento fallito)
UNRESOLVED_LICENSES = {"unknown", "other", "unknown / unspecified", ""}

# ==========================================
# MAPPATURA LICENZE -> GRUPPI (ID reali nel grafo Neo4j)
# ==========================================
LICENSE_TO_GROUP = {
    # GREEN: uso commerciale libero
    "mit": "GREEN",
    "apache-2.0": "GREEN",
    "bsd-3-clause": "GREEN",
    "bsd-2-clause": "GREEN",
    "cc0-1.0": "GREEN",
    "cc-by-4.0": "GREEN",
    "cc-by-sa-3.0": "GREEN",
    "cc-by-sa-4.0": "GREEN",
    "cdla-permissive-2.0": "GREEN",
    "afl-3.0": "GREEN",
    "apple-amlr": "GREEN",

    # YELLOW: licenze commerciali condizionate / brand-specific
    "llama-3-community": "YELLOW",
    "llama-3.1-community": "YELLOW",
    "llama-3.2-community": "YELLOW",
    "llama2": "YELLOW",
    "llama3": "YELLOW",
    "llama3.1": "YELLOW",
    "llama3.2": "YELLOW",
    "llama3.3": "YELLOW",
    "llama-derived-license": "YELLOW",
    "gemma": "YELLOW",
    "gemma-terms-of-use": "YELLOW",
    "qwen-license": "YELLOW",
    "qwen-commercial-license": "YELLOW",
    "flux-1-dev-license": "YELLOW",
    "microsoft-open-source-license": "YELLOW",
    "unsloth-derived-license": "YELLOW",
    "wan-2-license": "YELLOW",
    "tencent-hunyuan-license": "YELLOW",
    "deepseek-license": "YELLOW",
    "mistral-community-license": "YELLOW",
    "01-ai-yi-license": "YELLOW",
    "modified-mit": "GREEN",
    "nvidia-nemotron-open-model-license": "YELLOW",
    "nvidia-open-model-license": "YELLOW",

    # ORANGE: OpenRAIL e licenze con restrizioni d'uso specifiche
    "creativeml-openrail-m": "ORANGE",
    "openrail": "ORANGE",
    "openrail++": "ORANGE",
    "bigcode-openrail-m": "ORANGE",
    "bigscience-bloom-rail-1.0": "ORANGE",

    # RED_COPYLEFT: copyleft forte
    "agpl-3.0": "RED_COPYLEFT",
    "gpl-3.0": "RED_COPYLEFT",

    # RED_RESTRICTED: divieto uso commerciale / non specificato
    "cc-by-nc-2.0": "RED_RESTRICTED",
    "cc-by-nc-3.0": "RED_RESTRICTED",
    "cc-by-nc-4.0": "RED_RESTRICTED",
    "cc-by-nc-sa-4.0": "RED_RESTRICTED",
    "ai-by-nc-1.0": "RED_RESTRICTED",
    "research-only-license": "RED_RESTRICTED",
    "unknown": "RED_RESTRICTED",
    "other": "RED_RESTRICTED",
    "unknown / unspecified": "RED_RESTRICTED",
}

VALID_LICENSE_GROUP_IDS = {"GREEN", "YELLOW", "ORANGE", "RED_COPYLEFT", "RED_RESTRICTED"}


def _normalizza_license_tag(license_tag):
    if isinstance(license_tag, list):
        license_tag = license_tag[0] if license_tag else None
    return str(license_tag).strip().lower() if license_tag else "unknown"


def estrai_licenza_da_api(data):
    """
    Legge la licenza da cardData o top-level API HF.
    Se license è 'other'/'unknown', usa cardData.license_name (licenza reale su HF).
    """
    card_data = data.get("cardData") or {}
    license_tag = card_data.get("license") or data.get("license")
    license_name = _normalizza_license_tag(license_tag)

    if license_name in UNRESOLVED_LICENSES:
        license_name_from_card = _normalizza_license_tag(card_data.get("license_name"))
        if license_name_from_card not in UNRESOLVED_LICENSES:
            return license_name_from_card

    return license_name


def estrai_base_model_id(model_data):
    """Estrae il modello base da cardData.base_model o dal tag HF base_model:."""
    card_data = model_data.get("cardData") or {}
    base_model = card_data.get("base_model")
    if isinstance(base_model, list):
        base_model = base_model[0] if base_model else None
    if base_model:
        return str(base_model).strip()

    for tag in model_data.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("base_model:"):
            return tag.split(":", 1)[1].strip()

    return None


def scarica_e_analizza_file_license(model_id):
    """
    Ispezione profonda del repository: scarica il file LICENSE raw
    quando cardData/API restituiscono 'other' o 'unknown'.
    """
    branches = ["main", "master"]
    extensions = ["LICENSE", "LICENSE.txt", "license", "license.txt"]

    for branch in branches:
        url_base_raw = f"https://huggingface.co/{model_id}/raw/{branch}/"
        for extension in extensions:
            try:
                response = requests.get(url_base_raw + extension, timeout=5)
                if response.status_code != 200:
                    continue

                license_text = response.text.lower()
                print(f"   [FILE] Trovato {extension} ({branch}) per {model_id}")

                if "apache license" in license_text and "version 2.0" in license_text:
                    return "apache-2.0"
                if "mit license" in license_text:
                    return "mit"
                if "qwen research and commercial license" in license_text:
                    return "qwen-commercial-license"
                if "llama 3 community license" in license_text:
                    return "llama-3-community"
                if "llama 3.1 community license" in license_text:
                    return "llama-3.1-community"
                if "gemma terms of use" in license_text or "gemma-license" in license_text:
                    return "gemma-terms-of-use"
                if "creative commons attribution non commercial" in license_text or "cc-by-nc" in license_text:
                    return "cc-by-nc-4.0"
                if "flux-1-dev-local-license" in license_text:
                    return "flux-1-dev-license"

                clean_name = model_id.replace("/", "-").lower()
                return f"custom-{clean_name}-license"

            except requests.RequestException:
                continue

    return "unknown"


def risolvi_licenza_da_base_model(base_model_id):
    """Risale al modello base (es. export Xenova) per ereditare la licenza."""
    try:
        response = requests.get(f"https://huggingface.co/api/models/{base_model_id}", timeout=10)
        if response.status_code != 200:
            return "unknown"

        base_data = response.json()
        base_license = estrai_licenza_da_api(base_data)
        if base_license not in UNRESOLVED_LICENSES:
            return base_license

        return scarica_e_analizza_file_license(base_model_id)
    except requests.RequestException:
        return "unknown"


def determina_licenza_reale(model_id, api_license_tag, model_data=None):
    """
    Risolve la licenza incrociando tag API/cardData, file LICENSE nel repo
    e, come ultimo fallback, la licenza del modello base.
    """
    api_license = str(api_license_tag).lower().strip() if api_license_tag else "unknown"

    if "openai/clip-" in model_id.lower():
        return "mit"

    if api_license not in UNRESOLVED_LICENSES:
        return api_license

    print(f"[WARN] Modello '{model_id}' marcato come '{api_license}'. Controllo file LICENSE...")
    license_from_file = scarica_e_analizza_file_license(model_id)
    if license_from_file not in UNRESOLVED_LICENSES:
        return license_from_file

    if model_data:
        base_model_id = estrai_base_model_id(model_data)
        if base_model_id and base_model_id.lower() != model_id.lower():
            print(f"   [BASE] Risalgo a '{base_model_id}'...")
            license_from_base = risolvi_licenza_da_base_model(base_model_id)
            if license_from_base not in UNRESOLVED_LICENSES:
                return license_from_base

    return "unknown"


def risolvi_gruppo_rischio(license_name):
    """Mappa una licenza al LicenseGroup.id presente nel grafo."""
    if license_name in LICENSE_TO_GROUP:
        return LICENSE_TO_GROUP[license_name]
    if license_name.startswith("custom-"):
        return "YELLOW"

    license_lower = license_name.lower()
    if "nc" in license_lower or "non-commercial" in license_lower or "research-only" in license_lower:
        return "RED_RESTRICTED"
    if "agpl" in license_lower or "gpl" in license_lower:
        return "RED_COPYLEFT"
    if "openrail" in license_lower or "rail" in license_lower:
        return "ORANGE"
    if any(token in license_lower for token in ("llama", "gemma", "qwen", "flux", "mistral", "deepseek", "hunyuan", "nvidia", "minimax")):
        return "YELLOW"
    if any(token in license_lower for token in ("mit", "apache", "bsd", "cc0", "cdla")):
        return "GREEN"

    return "YELLOW"


def aggiorna_licenza_sul_modello(session, model_id, license_name):
    """Collega il modello alla licenza risolta. Ritorna True se il modello esiste."""
    result = session.run(
        """
        MATCH (m:Model {name: $model_id})
        OPTIONAL MATCH (m)-[r:UNDER_LICENSE]->()
        DELETE r
        WITH m
        MERGE (l:License {name: $license_name})
        MERGE (m)-[:UNDER_LICENSE]->(l)
        RETURN l.name AS license_name
        """,
        model_id=model_id,
        license_name=license_name,
    )
    return result.single() is not None


def collega_licenza_al_gruppo(session, license_name, group_id):
    """Collega License a LicenseGroup. Ritorna True solo se entrambi i nodi esistono."""
    if group_id not in VALID_LICENSE_GROUP_IDS:
        raise ValueError(f"Gruppo licenza non valido: {group_id}")

    result = session.run(
        """
        MATCH (l:License {name: $license_name})
        MATCH (g:LicenseGroup {id: $group_id})
        MERGE (l)-[:BELONGS_TO]->(g)
        RETURN g.id AS group_id
        """,
        license_name=license_name,
        group_id=group_id,
    )
    return result.single() is not None


def esegui_arricchimento_massivo():
    driver = GraphDatabase.driver(URI, auth=AUTH)

    query_seleziona_modelli = """
    MATCH (m:Model)-[:UNDER_LICENSE]->(l:License)
    WHERE l.name IN ["Unknown / Unspecified", "other", "unknown"]
    RETURN m.name AS model_id
    ORDER BY m.downloads DESC
    """

    with driver.session() as session:
        result = session.run(query_seleziona_modelli)
        modelli_da_elaborare = [record["model_id"] for record in result]

    print(f"Inizio scansione su {len(modelli_da_elaborare)} modelli con licenza non risolta...\n")

    aggiornati = 0
    saltati = 0

    for model_id in modelli_da_elaborare:
        url_api = f"https://huggingface.co/api/models/{model_id}"

        try:
            response = requests.get(url_api, timeout=10)
            if response.status_code != 200:
                print(f"[ERRORE API] HF non raggiungibile per {model_id} (status: {response.status_code})")
                continue

            data = response.json()
            tag_licenza_api = estrai_licenza_da_api(data)
            licenza_vera = determina_licenza_reale(model_id, tag_licenza_api, model_data=data)

            if licenza_vera in UNRESOLVED_LICENSES:
                saltati += 1
                print(f"[SKIP] {model_id} -> licenza ancora non risolvibile ('{licenza_vera}')")
                continue

            gruppo_rischio = risolvi_gruppo_rischio(licenza_vera)

            with driver.session() as session:
                model_updated = aggiorna_licenza_sul_modello(session, model_id, licenza_vera)
                if not model_updated:
                    print(f"[ERRORE] Modello non trovato nel grafo: {model_id}")
                    continue

                group_linked = collega_licenza_al_gruppo(session, licenza_vera, gruppo_rischio)

            if group_linked:
                print(f"[OK] {model_id} -> {licenza_vera} -> {gruppo_rischio}")
                aggiornati += 1
            else:
                print(f"[WARN] {model_id} -> licenza salvata ({licenza_vera}) ma gruppo {gruppo_rischio} non collegato")

        except Exception as error:
            print(f"[ERRORE] Elaborazione di {model_id}: {error}")

    driver.close()
    print(f"\n[COMPLETATO] Aggiornati: {aggiornati} | Saltati (ancora unknown): {saltati}")


if __name__ == "__main__":
    esegui_arricchimento_massivo()
