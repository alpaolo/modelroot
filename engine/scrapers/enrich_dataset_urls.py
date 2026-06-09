"""
Build dataset name → documentation URL mapping for ModelRoot.

Reads distinct Dataset nodes from Neo4j, resolves URLs via:
1. Curated official/HF overrides (plain names)
2. Hugging Face datasets path for org/name IDs (incl. gated repos)
3. HF Hub search with exact slug match, then best search hit

Output: constants/dataset_urls.json
"""
import json
import sys
import time
from pathlib import Path

import requests
from huggingface_hub import HfApi
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / ".env"))
import config as env

NEO4J_URI = env.NEO4J_URI
NEO4J_AUTH = env.NEO4J_AUTH

HF_DATASETS_BASE_URL = "https://huggingface.co/datasets/"
HF_DATASETS_API_URL = "https://huggingface.co/api/datasets/"
REQUEST_DELAY_SECONDS = 0.3
OUTPUT_JSON_PATH = Path(__file__).resolve().parents[2] / "constants" / "dataset_urls.json"

DATASET_OFFICIAL_URL_OVERRIDES = {
    "agender": "https://github.com/rudinger/possession-english-racial-bias",
    "ami": "https://groups.inf.ed.ac.uk/ami/corpus/",
    "aqua_rat": "https://github.com/google-deepmind/AQuA",
    "bookcorpus": "https://github.com/soskek/bookcorpus",
    "brWaC": "https://huggingface.co/datasets/UFRGS/brwac",
    "c4": "https://huggingface.co/datasets/allenai/c4",
    "cnn_dailymail": "https://huggingface.co/datasets/abisee/cnn_dailymail",
    "code_search_net": "https://github.com/github/CodeSearchNet",
    "common_voice": "https://commonvoice.mozilla.org/datasets",
    "conll2003": "https://huggingface.co/datasets/eriktks/conll2003",
    "dihard": "https://dihardchallenge.github.io/dihard3/",
    "eli5": "https://huggingface.co/datasets/defunct-datasets/eli5",
    "esnli": "https://github.com/OanaMariaCamburu/esnli",
    "glue": "https://gluebenchmark.com/",
    "gooaq": "https://github.com/google-research-datasets/gooaq",
    "gsm8k": "https://huggingface.co/datasets/openai/gsm8k",
    "imagenet-1k": "https://www.image-net.org/",
    "imagenet-21k": "https://www.image-net.org/",
    "indonlu": "https://huggingface.co/datasets/indonlp/indonlu",
    "lambada": "https://huggingface.co/datasets/cimec/lambada",
    "librispeech_asr": "https://www.openslr.org/12",
    "mozillacommonvoice": "https://commonvoice.mozilla.org/datasets",
    "ms_marco": "https://microsoft.github.io/msmarco/",
    "multi_nli": "https://huggingface.co/datasets/nyu-mll/multi_nli",
    "natural_questions": "https://ai.google.com/research/NaturalQuestions/",
    "nsmc": "https://github.com/e9t/nsmc",
    "openwebtext": "https://github.com/jcpeterson/openwebtext",
    "qed": "https://huggingface.co/datasets/google-research-datasets/qed",
    "quasc": "https://github.com/IBM/question-answering-with-synthetic-compositions",
    "s2orc": "https://github.com/allenai/s2orc",
    "search_qa": "https://github.com/nyu-dl/dl4ir-searchQA",
    "snli": "https://nlp.stanford.edu/projects/snli/",
    "sst2": "https://huggingface.co/datasets/stanfordnlp/sst2",
    "taskmaster2": "https://huggingface.co/datasets/google-research-datasets/taskmaster2",
    "timit": "https://catalog.ldc.upenn.edu/LDC93S1",
    "trivia_qa": "https://nlp.cs.washington.edu/triviaqa/",
    "tweet_eval": "https://huggingface.co/datasets/cardiffnlp/tweet_eval",
    "voxceleb": "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/",
    "voxceleb2": "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html",
    "voxconverse": "https://github.com/joonson/voxconverse",
    "wider_face": "http://mmlab.ie.cuhk.edu.hk/projects/WIDER/WIDER_Face/",
    "wikihow": "https://github.com/mahnazkoupaee/WikiHow",
    "wikipedia": "https://dumps.wikimedia.org/",
    "yahoo_answers_topics": "https://webscope.sandbox.yahoo.com/catalog.php?datatype=l",
}


def log(message):
    print(message, flush=True)


def load_dataset_names_from_neo4j(driver):
    with driver.session() as session:
        return [
            record["name"]
            for record in session.run(
                "MATCH (d:Dataset) RETURN d.name AS name ORDER BY name"
            )
        ]


def hugging_face_dataset_page_url(dataset_repo_id):
    return f"{HF_DATASETS_BASE_URL}{dataset_repo_id}"


def hugging_face_dataset_api_status(dataset_repo_id):
    response = requests.get(
        f"{HF_DATASETS_API_URL}{dataset_repo_id}",
        timeout=15,
    )
    return response.status_code


def search_hugging_face_dataset_repo_id(dataset_name, hf_api):
    search_results = list(hf_api.list_datasets(search=dataset_name, limit=8))
    normalized_dataset_name = dataset_name.lower()

    for dataset_info in search_results:
        if dataset_info.id.lower() == normalized_dataset_name:
            return dataset_info.id

    for dataset_info in search_results:
        if dataset_info.id.split("/")[-1].lower() == normalized_dataset_name:
            return dataset_info.id

    if search_results:
        return search_results[0].id

    return None


def resolve_dataset_documentation_url(dataset_name, hf_api):
    if dataset_name in DATASET_OFFICIAL_URL_OVERRIDES:
        return DATASET_OFFICIAL_URL_OVERRIDES[dataset_name], "override"

    if "/" in dataset_name:
        hf_page_url = hugging_face_dataset_page_url(dataset_name)
        api_status = hugging_face_dataset_api_status(dataset_name)
        if api_status in {200, 401, 403}:
            return hf_page_url, f"hf_repo_api_{api_status}"
        return hf_page_url, "hf_repo_fallback"

    hf_repo_id = search_hugging_face_dataset_repo_id(dataset_name, hf_api)
    if hf_repo_id:
        if hf_repo_id.split("/")[-1].lower() == dataset_name.lower():
            return hugging_face_dataset_page_url(hf_repo_id), "hf_search_exact"
        return hugging_face_dataset_page_url(hf_repo_id), "hf_search_fuzzy"

    return None, "unresolved"


def build_dataset_url_mapping(dataset_names):
    hf_api = HfApi()
    dataset_urls = {}
    resolution_sources = {}
    unresolved_dataset_names = []

    for index, dataset_name in enumerate(dataset_names, start=1):
        documentation_url, resolution_source = resolve_dataset_documentation_url(
            dataset_name,
            hf_api,
        )
        if documentation_url:
            dataset_urls[dataset_name] = documentation_url
            resolution_sources[dataset_name] = resolution_source
            log(f"[OK {index}/{len(dataset_names)}] {dataset_name} -> {documentation_url} ({resolution_source})")
        else:
            unresolved_dataset_names.append(dataset_name)
            log(f"[MISS {index}/{len(dataset_names)}] {dataset_name}")

        time.sleep(REQUEST_DELAY_SECONDS)

    return dataset_urls, resolution_sources, unresolved_dataset_names


def write_dataset_urls_json(dataset_urls):
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_PATH.write_text(
        json.dumps(dataset_urls, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    log("=== enrich_dataset_urls.py ===")
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    dataset_names = load_dataset_names_from_neo4j(driver)
    driver.close()

    log(f"Datasets in Neo4j: {len(dataset_names)}")
    dataset_urls, resolution_sources, unresolved_dataset_names = build_dataset_url_mapping(
        dataset_names
    )
    write_dataset_urls_json(dataset_urls)

    source_counts = {}
    for source in resolution_sources.values():
        source_counts[source] = source_counts.get(source, 0) + 1

    log("\n=== COMPLETATO ===")
    log(f"Risolti:   {len(dataset_urls)}")
    log(f"Non trovati: {len(unresolved_dataset_names)}")
    log(f"Output:    {OUTPUT_JSON_PATH}")
    log(f"Fonti:     {source_counts}")
    if unresolved_dataset_names:
        log(f"Mancanti:  {unresolved_dataset_names}")


if __name__ == "__main__":
    main()
