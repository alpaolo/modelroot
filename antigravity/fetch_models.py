"""
Fetch 10,000+ top Hugging Face models and generate Neo4j-ready CSV files.

This script:
1. Fetches models from the HF API sorted by downloads
2. Classifies each model by domain, task, license, resource profile
3. Generates node and relationship CSV files for Neo4j import

Usage:
    python fetch_models.py [--target N]  (default: 10500)
"""

import csv
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from collections import defaultdict
from datetime import datetime

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
TARGET_MODELS = 10500  # fetch slightly more than 10k to account for filtering
BATCH_SIZE = 100       # HF API max per request for full=true is ~100
API_BASE = "https://huggingface.co/api/models"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data")
MAX_RETRIES = 5
RETRY_DELAY = 3  # seconds

# ──────────────────────────────────────────────
# Domain Classification Rules
# ──────────────────────────────────────────────
DOMAIN_RULES = {
    "NLP / Language": {
        "pipeline_tags": {
            "text-generation", "text-classification", "token-classification",
            "fill-mask", "translation", "summarization", "question-answering",
            "text2text-generation", "conversational", "chat-completion",
            "zero-shot-classification", "table-question-answering",
            "text-generation-inference",
        },
        "tag_keywords": [
            "nlp", "language-model", "natural-language", "sentiment",
            "ner", "pos", "dependency-parsing",
        ],
    },
    "Computer Vision": {
        "pipeline_tags": {
            "image-classification", "object-detection", "image-segmentation",
            "depth-estimation", "image-to-text", "video-classification",
            "zero-shot-image-classification", "image-feature-extraction",
            "mask-generation", "keypoint-detection",
        },
        "tag_keywords": [
            "computer-vision", "cv", "vision", "image", "yolo", "cnn",
            "resnet", "vit", "convnext", "segmentation",
        ],
    },
    "Audio / Speech": {
        "pipeline_tags": {
            "automatic-speech-recognition", "text-to-speech",
            "audio-classification", "voice-activity-detection",
            "audio-to-audio", "text-to-audio",
        },
        "tag_keywords": [
            "speech", "audio", "voice", "asr", "tts", "whisper",
            "wav2vec", "sound",
        ],
    },
    "Generative AI / Creative": {
        "pipeline_tags": {
            "text-to-image", "text-to-video", "image-to-image",
            "unconditional-image-generation", "image-to-video",
        },
        "tag_keywords": [
            "diffusion", "stable-diffusion", "sdxl", "dalle", "midjourney",
            "lora", "dreambooth", "controlnet", "flux", "generative",
        ],
    },
    "Science / Research": {
        "pipeline_tags": set(),
        "tag_keywords": [
            "biology", "chemistry", "physics", "protein", "genomics",
            "genome", "dna", "rna", "molecule", "drug", "materials",
            "scientific", "science", "bioinformatics", "astronomy",
            "climate", "earth-science", "geoscience", "ecology",
            "neuroscience", "proteomics", "metabolomics",
        ],
    },
    "Medicine / Healthcare": {
        "pipeline_tags": set(),
        "tag_keywords": [
            "medical", "clinical", "biomedical", "health", "pathology",
            "radiology", "healthcare", "disease", "diagnosis", "patient",
            "oncology", "cardiology", "dermatology", "ophthalmology",
            "dental", "ehr", "electronic-health", "icd", "snomed",
            "pubmed", "medqa", "medbench",
        ],
    },
    "Robotics / Embodied AI": {
        "pipeline_tags": {"reinforcement-learning", "robotics"},
        "tag_keywords": [
            "robotics", "robot", "control", "navigation", "manipulation",
            "embodied", "sim2real", "mujoco", "isaac", "ros",
        ],
    },
    "Code / Programming": {
        "pipeline_tags": set(),
        "tag_keywords": [
            "code", "programming", "sql", "code-generation", "codegen",
            "coder", "starcoder", "codellama", "deepseek-coder",
            "copilot", "autocomplete", "github",
        ],
    },
    "Finance": {
        "pipeline_tags": set(),
        "tag_keywords": [
            "finance", "financial", "trading", "stock", "banking",
            "finbert", "investment", "economic", "forex", "crypto",
        ],
    },
    "Legal": {
        "pipeline_tags": set(),
        "tag_keywords": [
            "legal", "law", "contract", "court", "regulation",
            "compliance", "legislative", "juridical",
        ],
    },
    "Education": {
        "pipeline_tags": set(),
        "tag_keywords": [
            "education", "tutoring", "academic", "student", "exam",
            "quiz", "learning", "textbook", "educational",
        ],
    },
    "Multimodal": {
        "pipeline_tags": {
            "visual-question-answering", "document-question-answering",
            "video-text-to-text", "image-text-to-text", "any-to-any",
        },
        "tag_keywords": [
            "multimodal", "multi-modal", "vlm", "vision-language",
            "clip", "blip", "llava", "fuyu",
        ],
    },
    "Embeddings / Retrieval": {
        "pipeline_tags": {
            "feature-extraction", "sentence-similarity", "text-ranking",
        },
        "tag_keywords": [
            "embedding", "embeddings", "retrieval", "reranker", "reranking",
            "sentence-transformer", "dense-retrieval", "rag", "vector",
            "semantic-search", "text-embeddings",
        ],
    },
}

# ──────────────────────────────────────────────
# License Classification
# ──────────────────────────────────────────────
LICENSE_TYPES = {
    # Permissive
    "apache-2.0": ("Permissive", True),
    "mit": ("Permissive", True),
    "bsd-2-clause": ("Permissive", True),
    "bsd-3-clause": ("Permissive", True),
    "isc": ("Permissive", True),
    "unlicense": ("Permissive", True),
    "wtfpl": ("Permissive", True),
    "artistic-2.0": ("Permissive", True),
    "zlib": ("Permissive", True),
    "bsl-1.0": ("Permissive", True),
    "ecl-2.0": ("Permissive", True),
    "postgresql": ("Permissive", True),
    "ncsa": ("Permissive", True),
    "ms-pl": ("Permissive", True),
    # Copyleft
    "gpl-2.0": ("Copyleft", True),
    "gpl-3.0": ("Copyleft", True),
    "agpl-3.0": ("Copyleft", True),
    "lgpl-2.1": ("Copyleft", True),
    "lgpl-3.0": ("Copyleft", True),
    "mpl-2.0": ("Copyleft", True),
    "eupl-1.1": ("Copyleft", True),
    "osl-3.0": ("Copyleft", True),
    # Creative Commons
    "cc-by-4.0": ("Creative Commons", True),
    "cc-by-3.0": ("Creative Commons", True),
    "cc-by-sa-4.0": ("Creative Commons", True),
    "cc-by-sa-3.0": ("Creative Commons", True),
    "cc-by-nc-4.0": ("Creative Commons - NC", False),
    "cc-by-nc-3.0": ("Creative Commons - NC", False),
    "cc-by-nc-sa-4.0": ("Creative Commons - NC", False),
    "cc-by-nc-sa-3.0": ("Creative Commons - NC", False),
    "cc-by-nd-4.0": ("Creative Commons - ND", True),
    "cc-by-nc-nd-4.0": ("Creative Commons - NC-ND", False),
    "cc-by-nc-nd-3.0": ("Creative Commons - NC-ND", False),
    "cc0-1.0": ("Creative Commons", True),
    "pddl": ("Creative Commons", True),
    # Responsible AI
    "openrail": ("Responsible AI", True),
    "openrail++": ("Responsible AI", True),
    "bigscience-openrail-m": ("Responsible AI", True),
    "creativeml-openrail-m": ("Responsible AI", True),
    "bigcode-openrail-m": ("Responsible AI", True),
    "bigscience-bloom-rail-1.0": ("Responsible AI", True),
    # Custom/Restricted
    "llama2": ("Custom/Restricted", True),
    "llama3": ("Custom/Restricted", True),
    "llama3.1": ("Custom/Restricted", True),
    "llama3.2": ("Custom/Restricted", True),
    "llama3.3": ("Custom/Restricted", True),
    "llama4": ("Custom/Restricted", True),
    "gemma": ("Custom/Restricted", True),
    "deepseek": ("Custom/Restricted", True),
    "qwen": ("Permissive", True),
    "tongyi-qianwen": ("Permissive", True),
    "yi-license": ("Custom/Restricted", True),
    "other": ("Custom/Restricted", None),
}

# ──────────────────────────────────────────────
# Resource Profile Classification
# ──────────────────────────────────────────────
RESOURCE_PROFILES = [
    ("Nano",    0,          100_000_000,     0.2),
    ("Micro",   100_000_000, 500_000_000,   1.0),
    ("Small",   500_000_000, 1_000_000_000, 2.0),
    ("Medium",  1_000_000_000, 7_000_000_000, 14.0),
    ("Large",   7_000_000_000, 30_000_000_000, 60.0),
    ("XLarge",  30_000_000_000, 70_000_000_000, 140.0),
    ("XXLarge", 70_000_000_000, float("inf"), 280.0),
]


def get_resource_profile(num_params):
    """Classify model into a resource profile based on parameter count."""
    if num_params is None or num_params <= 0:
        return "Unknown"
    for name, min_p, max_p, _ in RESOURCE_PROFILES:
        if min_p <= num_params < max_p:
            return name
    return "Unknown"


def estimate_params_from_name(model_id):
    """Try to estimate parameter count from model name patterns like '7b', '13b', '70b', '1.5b', '350m'."""
    name_lower = model_id.lower()
    # Match patterns like 7b, 7B, 1.5b, 1.5B, 0.5b
    match = re.search(r'[\-_.](\d+\.?\d*)b(?:[\-_.\s]|$)', name_lower)
    if match:
        return int(float(match.group(1)) * 1_000_000_000)
    # Match patterns like 350m, 125m
    match = re.search(r'[\-_.](\d+\.?\d*)m(?:[\-_.\s]|$)', name_lower)
    if match:
        return int(float(match.group(1)) * 1_000_000)
    # Match patterns like 1.5t (trillion)
    match = re.search(r'[\-_.](\d+\.?\d*)t(?:[\-_.\s]|$)', name_lower)
    if match:
        return int(float(match.group(1)) * 1_000_000_000_000)
    return None


def classify_domains(model):
    """Classify a model into one or more domains."""
    pipeline_tag = model.get("pipeline_tag", "") or ""
    raw_tags = model.get("tags", [])
    model_id_lower = model.get("id", "").lower()

    # Filter out prefixed tags that cause false positives
    # Only keep meaningful tags for domain classification
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

    for domain_name, rules in DOMAIN_RULES.items():
        matched = False
        # Check pipeline_tag
        if pipeline_tag in rules["pipeline_tags"]:
            matched = True
        # Check tags (only filtered tags, not dataset/arxiv/base_model refs)
        if not matched:
            for kw in rules["tag_keywords"]:
                # For short/generic keywords, require exact tag match or word boundary in model ID
                if len(kw) <= 3 or kw in ("code", "control", "drug", "learning"):
                    # Exact match in tags or word-boundary match in model ID
                    if kw in filtered_tags or re.search(r'(?:^|[\-_./])' + re.escape(kw) + r'(?:$|[\-_./])', model_id_lower):
                        matched = True
                        break
                else:
                    if any(kw in tag for tag in filtered_tags) or kw in model_id_lower:
                        matched = True
                        break
        if matched:
            domains.append(domain_name)

    if not domains:
        domains.append("Other")

    return domains


def extract_license(tags):
    """Extract license string from model tags."""
    for tag in tags:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1].strip()
    return "unknown"


def extract_base_models(tags):
    """Extract base model IDs from tags."""
    base_models = []
    for tag in tags:
        if tag.startswith("base_model:") and not tag.startswith("base_model:quantized:"):
            bm = tag.split(":", 1)[1].strip()
            if bm:
                base_models.append(bm)
    return base_models


def extract_language(tags):
    """Extract language codes from tags."""
    langs = []
    known_lang_codes = {
        "en", "fr", "de", "es", "it", "pt", "nl", "ru", "zh", "ja", "ko",
        "ar", "hi", "tr", "pl", "sv", "da", "no", "fi", "cs", "hu", "ro",
        "bg", "uk", "vi", "th", "id", "ms", "tl", "he", "fa", "ur",
        "bn", "ta", "te", "kn", "ml", "si", "my", "ka", "mk", "sq",
        "et", "lv", "lt", "sk", "sl", "hr", "sr", "bs", "mt", "ga",
        "cy", "eu", "ca", "gl", "ast", "oc", "af", "sw", "ha", "yo",
        "ig", "zu", "xh", "rw", "mg", "sn", "so", "am", "or", "ne",
        "lo", "km", "mn", "bo", "ug", "kk", "ky", "uz", "tg", "ps",
        "sd", "gu", "mr", "pa", "as", "sa", "ks", "bh", "doi", "kok",
        "sat", "mai", "mni", "lus", "multilingual",
    }
    for tag in tags:
        tag_clean = tag.strip().lower()
        if tag_clean in known_lang_codes:
            langs.append(tag_clean)
    return langs


def extract_num_params(model):
    """Try to extract num_parameters from safetensors metadata or model name."""
    # Try safetensors metadata
    safetensors = model.get("safetensors")
    if safetensors and isinstance(safetensors, dict):
        params = safetensors.get("total")
        if params and isinstance(params, (int, float)) and params > 0:
            return int(params)
        # Try parameters field
        parameters = safetensors.get("parameters")
        if parameters and isinstance(parameters, dict):
            total = sum(v for v in parameters.values() if isinstance(v, (int, float)))
            if total > 0:
                return int(total)

    # Try cardData
    card_data = model.get("cardData")
    if card_data and isinstance(card_data, dict):
        params = card_data.get("num_parameters") or card_data.get("parameters")
        if params and isinstance(params, (int, float)) and params > 0:
            return int(params)

    # Try model config
    config = model.get("config")
    if config and isinstance(config, dict):
        # Some models have num_parameters in config
        params = config.get("num_parameters")
        if params and isinstance(params, (int, float)):
            return int(params)

    # Fallback: estimate from name
    return estimate_params_from_name(model.get("id", ""))


def fetch_models_batch(skip, limit, use_full=False, pipeline_tag=None):
    """Fetch a batch of models from the HF API.
    
    Returns list of models, or empty list on 400 error (pagination limit).
    """
    params = {
        "sort": "downloads",
        "direction": "-1",
        "limit": str(limit),
        "skip": str(skip),
    }
    if use_full:
        params["full"] = "true"
        params["config"] = "true"
    if pipeline_tag:
        params["pipeline_tag"] = pipeline_tag

    url = f"{API_BASE}?{urllib.parse.urlencode(params)}"

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "HuggingFace-Neo4j-Catalog/1.0")
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_time = RETRY_DELAY * (2 ** attempt)
                print(f"    [!] Rate limited (429). Waiting {wait_time}s...")
                time.sleep(wait_time)
            elif e.code == 400:
                # Pagination limit reached
                return []
            elif e.code >= 500:
                wait_time = RETRY_DELAY * (attempt + 1)
                print(f"    [!] Server error ({e.code}). Retry {attempt+1}/{MAX_RETRIES} in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    [X] HTTP error {e.code}: {e.reason}")
                if attempt == MAX_RETRIES - 1:
                    return []
                time.sleep(RETRY_DELAY)
        except (urllib.error.URLError, TimeoutError) as e:
            wait_time = RETRY_DELAY * (attempt + 1)
            print(f"    [!] Connection error: {e}. Retry {attempt+1}/{MAX_RETRIES} in {wait_time}s...")
            time.sleep(wait_time)

    print(f"    [!] Failed after {MAX_RETRIES} retries at skip={skip}")
    return []


def save_partial(models, filepath):
    """Save partial results to JSON file for recovery."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(models, f, ensure_ascii=False)


def fetch_by_pipeline_tag(tag, max_per_tag=3000, use_full=False):
    """Fetch models for a specific pipeline_tag, respecting API skip limit."""
    models = []
    skip = 0
    while len(models) < max_per_tag:
        batch = fetch_models_batch(skip, BATCH_SIZE, use_full=use_full, pipeline_tag=tag)
        if not batch:
            break
        models.extend(batch)
        skip += BATCH_SIZE
        time.sleep(0.2)
    return models


def fetch_all_models(target=TARGET_MODELS):
    """Fetch 10,000+ models from HF API using a multi-strategy approach.

    Strategy:
    1. Phase 1: Fetch top ~3000 models globally with full=true (rich metadata).
    2. Phase 2: For each pipeline_tag, fetch additional models (up to 2000 per tag).
    3. Deduplicate by model ID, preserving the richest version of each model.
    
    This bypasses the API's hard limit on skip (~3000) by using pipeline_tag
    filters, each of which resets the skip counter to 0.
    """
    # All pipeline_tags to query
    PIPELINE_TAGS = [
        # NLP
        "text-generation", "text-classification", "token-classification",
        "fill-mask", "translation", "summarization", "question-answering",
        "text2text-generation", "conversational", "zero-shot-classification",
        "table-question-answering", "sentence-similarity", "feature-extraction",
        "text-ranking",
        # Computer Vision
        "image-classification", "object-detection", "image-segmentation",
        "depth-estimation", "image-to-text", "video-classification",
        "zero-shot-image-classification", "image-feature-extraction",
        "mask-generation", "keypoint-detection",
        # Generative AI
        "text-to-image", "text-to-video", "image-to-image",
        "unconditional-image-generation", "image-to-video",
        # Audio
        "automatic-speech-recognition", "text-to-speech",
        "audio-classification", "voice-activity-detection",
        "audio-to-audio", "text-to-audio",
        # Multimodal
        "visual-question-answering", "document-question-answering",
        "video-text-to-text", "image-text-to-text",
        # Other
        "reinforcement-learning", "robotics",
        "any-to-any", "graph-ml",
    ]

    models_by_id = {}  # model_id -> model dict (deduplicated)
    partial_file = os.path.join(RAW_DATA_DIR, "partial_models.json")

    print(f">>> Fetching {target}+ models from Hugging Face API...")
    print(f"    Strategy: global top + per-pipeline-tag fetching")
    print(f"    Batch size: {BATCH_SIZE}")
    print()

    # ── Phase 1: Global top models with full metadata ──
    print("  === Phase 1: Global top models (full metadata) ===")
    skip = 0
    phase1_count = 0
    while skip < 3000:
        batch = fetch_models_batch(skip, BATCH_SIZE, use_full=True)
        if not batch:
            break
        for m in batch:
            mid = m.get("id", "")
            if mid and mid not in models_by_id:
                models_by_id[mid] = m
                phase1_count += 1
        skip += BATCH_SIZE
        print(f"    [OK] Global: {phase1_count:,} unique models [skip={skip}]")
        time.sleep(0.2)

    print(f"  Phase 1 complete: {phase1_count:,} models with full metadata")
    save_partial(list(models_by_id.values()), partial_file)
    print(f"  Checkpoint saved.\n")

    # ── Phase 2: Per-pipeline_tag fetching ──
    print("  === Phase 2: Fetching by pipeline_tag ===")
    max_per_tag = 2000  # Up to 2000 models per tag (20 batches)

    for tag_idx, tag in enumerate(PIPELINE_TAGS):
        if len(models_by_id) >= target:
            print(f"    Target reached ({len(models_by_id):,} models). Stopping.")
            break

        tag_models = fetch_by_pipeline_tag(tag, max_per_tag=max_per_tag, use_full=False)
        new_count = 0
        for m in tag_models:
            mid = m.get("id", "")
            if mid and mid not in models_by_id:
                models_by_id[mid] = m
                new_count += 1

        total = len(models_by_id)
        print(f"    [{tag_idx+1}/{len(PIPELINE_TAGS)}] {tag:40s} fetched={len(tag_models):>5,}  new={new_count:>5,}  total={total:>6,}")

        # Save checkpoint every 5 tags
        if (tag_idx + 1) % 5 == 0:
            save_partial(list(models_by_id.values()), partial_file)

    # ── Phase 3: If still short, try additional strategies ──
    if len(models_by_id) < target:
        print(f"\n  === Phase 3: Additional fetching (still need {target - len(models_by_id):,} more) ===")
        # Try fetching more per tag with higher limits
        for tag in PIPELINE_TAGS:
            if len(models_by_id) >= target:
                break
            extra = fetch_by_pipeline_tag(tag, max_per_tag=3000, use_full=False)
            new_count = 0
            for m in extra:
                mid = m.get("id", "")
                if mid and mid not in models_by_id:
                    models_by_id[mid] = m
                    new_count += 1
            if new_count > 0:
                print(f"    Extra {tag}: +{new_count:,} new (total: {len(models_by_id):,})")

    # Convert to list sorted by downloads
    all_models = sorted(models_by_id.values(),
                        key=lambda m: m.get("downloads", 0),
                        reverse=True)

    print(f"\n  [DONE] Total unique models fetched: {len(all_models):,}")
    return all_models[:target]


def process_models(raw_models):
    """Process raw model data into structured records."""
    print(f"\n>>> Processing {len(raw_models):,} models...")

    # Collect unique entities
    domains_set = set()
    tasks_set = set()
    licenses_set = set()
    authors_set = defaultdict(int)  # author -> count
    frameworks_set = set()

    processed = []
    domain_rels = []
    task_rels = []
    license_rels = []
    resource_rels = []
    author_rels = []
    framework_rels = []
    base_model_rels = []

    for i, model in enumerate(raw_models):
        model_id = model.get("id", "")
        if not model_id:
            continue

        # Extract fields
        author = model.get("author", "") or model_id.split("/")[0] if "/" in model_id else ""
        pipeline_tag = model.get("pipeline_tag", "") or "unknown"
        tags = model.get("tags", [])
        downloads = model.get("downloads", 0)
        likes = model.get("likes", 0)
        library_name = model.get("library_name", "") or ""
        created_at = model.get("createdAt", "")
        last_modified = model.get("lastModified", "")
        gated = model.get("gated", False)
        private = model.get("private", False)

        # Extract license
        license_str = extract_license(tags)

        # Extract base models
        base_models = extract_base_models(tags)

        # Extract languages
        languages = extract_language(tags)

        # Extract num parameters
        num_params = extract_num_params(model)

        # Classify resource profile
        resource_profile = get_resource_profile(num_params)

        # Classify domains
        model_domains = classify_domains(model)

        # Model size category (human readable)
        if num_params:
            if num_params >= 1_000_000_000_000:
                size_label = f"{num_params / 1_000_000_000_000:.1f}T"
            elif num_params >= 1_000_000_000:
                size_label = f"{num_params / 1_000_000_000:.1f}B"
            elif num_params >= 1_000_000:
                size_label = f"{num_params / 1_000_000:.0f}M"
            else:
                size_label = f"{num_params / 1_000:.0f}K"
        else:
            size_label = ""

        # Build processed record
        record = {
            "modelId": model_id,
            "name": model_id.split("/")[-1] if "/" in model_id else model_id,
            "author": author,
            "pipeline_tag": pipeline_tag,
            "downloads": downloads,
            "likes": likes,
            "library_name": library_name,
            "license": license_str,
            "created_at": created_at,
            "last_modified": last_modified,
            "num_parameters": num_params or "",
            "size_label": size_label,
            "resource_profile": resource_profile,
            "languages": ";".join(languages) if languages else "",
            "gated": str(gated).lower(),
            "private": str(private).lower(),
            "tags_count": len(tags),
        }
        processed.append(record)

        # Collect entities and relationships
        for d in model_domains:
            domains_set.add(d)
            domain_rels.append((model_id, d))

        tasks_set.add(pipeline_tag)
        task_rels.append((model_id, pipeline_tag))

        licenses_set.add(license_str)
        license_rels.append((model_id, license_str))

        resource_rels.append((model_id, resource_profile))

        if author:
            authors_set[author] += 1
            author_rels.append((model_id, author))

        if library_name:
            frameworks_set.add(library_name)
            framework_rels.append((model_id, library_name))

        for bm in base_models:
            base_model_rels.append((model_id, bm))

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1:,} / {len(raw_models):,} models...")

    print(f"  [DONE] Processing complete.")
    print(f"     Domains: {len(domains_set)}")
    print(f"     Tasks: {len(tasks_set)}")
    print(f"     Licenses: {len(licenses_set)}")
    print(f"     Authors: {len(authors_set)}")
    print(f"     Frameworks: {len(frameworks_set)}")

    return {
        "models": processed,
        "domains": sorted(domains_set),
        "tasks": sorted(tasks_set),
        "licenses": sorted(licenses_set),
        "authors": dict(authors_set),
        "frameworks": sorted(frameworks_set),
        "domain_rels": domain_rels,
        "task_rels": task_rels,
        "license_rels": license_rels,
        "resource_rels": resource_rels,
        "author_rels": author_rels,
        "framework_rels": framework_rels,
        "base_model_rels": base_model_rels,
    }


def make_id_safe(s):
    """Create a safe ID from a string."""
    return re.sub(r'[^a-zA-Z0-9._/-]', '_', str(s).strip()).lower()


def generate_csvs(data):
    """Generate all Neo4j-compatible CSV files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nodes_dir = os.path.join(OUTPUT_DIR, "nodes")
    rels_dir = os.path.join(OUTPUT_DIR, "relationships")
    os.makedirs(nodes_dir, exist_ok=True)
    os.makedirs(rels_dir, exist_ok=True)

    print(f"\n>>> Generating CSV files in {OUTPUT_DIR}...")

    # ── Node: Models ──
    filepath = os.path.join(nodes_dir, "nodes_models.csv")
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "modelId:ID(Model)", "name:string", "author:string",
            "pipeline_tag:string", "downloads:long", "likes:int",
            "library_name:string", "license:string",
            "created_at:string", "last_modified:string",
            "num_parameters:long", "size_label:string",
            "resource_profile:string", "languages:string[]",
            "gated:boolean", "private:boolean",
            "tags_count:int", ":LABEL"
        ])
        for m in data["models"]:
            writer.writerow([
                m["modelId"], m["name"], m["author"],
                m["pipeline_tag"], m["downloads"], m["likes"],
                m["library_name"], m["license"],
                m["created_at"], m["last_modified"],
                m["num_parameters"], m["size_label"],
                m["resource_profile"], m["languages"],
                m["gated"], m["private"],
                m["tags_count"], "Model"
            ])
    print(f"  [OK] {filepath} ({len(data['models']):,} records)")

    # ── Node: Domains ──
    filepath = os.path.join(nodes_dir, "nodes_domains.csv")
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["domainId:ID(Domain)", "name:string", "description:string", ":LABEL"])
        domain_descriptions = {
            "NLP / Language": "Natural Language Processing: text generation, classification, translation, and understanding",
            "Computer Vision": "Image and video understanding: classification, detection, segmentation",
            "Audio / Speech": "Speech recognition, text-to-speech, audio classification",
            "Generative AI / Creative": "Image/video/audio generation: diffusion models, GANs",
            "Science / Research": "Scientific computing: biology, chemistry, physics, genomics",
            "Medicine / Healthcare": "Medical AI: clinical NLP, medical imaging, diagnostics",
            "Robotics / Embodied AI": "Robotics: control, navigation, reinforcement learning",
            "Code / Programming": "Code generation, code understanding, programming assistance",
            "Finance": "Financial AI: trading, risk analysis, financial NLP",
            "Legal": "Legal AI: contract analysis, legal document processing",
            "Education": "Educational AI: tutoring, assessment, content generation",
            "Multimodal": "Multi-modal models: vision-language, document understanding",
            "Embeddings / Retrieval": "Embedding models: semantic search, retrieval, reranking",
            "Other": "Models not fitting other domain categories",
        }
        for d in data["domains"]:
            writer.writerow([
                make_id_safe(d), d,
                domain_descriptions.get(d, ""),
                "Domain"
            ])
    print(f"  [OK] {filepath} ({len(data['domains'])} records)")

    # ── Node: Tasks ──
    filepath = os.path.join(nodes_dir, "nodes_tasks.csv")
    task_categories = {
        "text-generation": "NLP", "text-classification": "NLP",
        "token-classification": "NLP", "fill-mask": "NLP",
        "translation": "NLP", "summarization": "NLP",
        "question-answering": "NLP", "text2text-generation": "NLP",
        "conversational": "NLP", "chat-completion": "NLP",
        "zero-shot-classification": "NLP", "table-question-answering": "NLP",
        "sentence-similarity": "NLP", "feature-extraction": "NLP",
        "text-ranking": "NLP",
        "image-classification": "Computer Vision",
        "object-detection": "Computer Vision",
        "image-segmentation": "Computer Vision",
        "depth-estimation": "Computer Vision",
        "image-to-text": "Computer Vision",
        "video-classification": "Computer Vision",
        "zero-shot-image-classification": "Computer Vision",
        "image-feature-extraction": "Computer Vision",
        "mask-generation": "Computer Vision",
        "keypoint-detection": "Computer Vision",
        "text-to-image": "Generative AI",
        "text-to-video": "Generative AI",
        "image-to-image": "Generative AI",
        "unconditional-image-generation": "Generative AI",
        "image-to-video": "Generative AI",
        "automatic-speech-recognition": "Audio",
        "text-to-speech": "Audio",
        "audio-classification": "Audio",
        "voice-activity-detection": "Audio",
        "audio-to-audio": "Audio",
        "text-to-audio": "Audio",
        "visual-question-answering": "Multimodal",
        "document-question-answering": "Multimodal",
        "video-text-to-text": "Multimodal",
        "image-text-to-text": "Multimodal",
        "reinforcement-learning": "RL",
        "robotics": "Robotics",
    }
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["taskId:ID(Task)", "name:string", "category:string", ":LABEL"])
        for t in data["tasks"]:
            writer.writerow([
                make_id_safe(t), t,
                task_categories.get(t, "Other"),
                "Task"
            ])
    print(f"  [OK] {filepath} ({len(data['tasks'])} records)")

    # ── Node: Licenses ──
    filepath = os.path.join(nodes_dir, "nodes_licenses.csv")
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "licenseId:ID(License)", "name:string", "type:string",
            "commercial_use:string", ":LABEL"
        ])
        for lic in data["licenses"]:
            lic_info = LICENSE_TYPES.get(lic, ("Unknown", None))
            commercial = ""
            if lic_info[1] is True:
                commercial = "true"
            elif lic_info[1] is False:
                commercial = "false"
            else:
                commercial = "unknown"
            writer.writerow([
                make_id_safe(lic), lic,
                lic_info[0],
                commercial,
                "License"
            ])
    print(f"  [OK] {filepath} ({len(data['licenses'])} records)")

    # ── Node: Resource Profiles ──
    filepath = os.path.join(nodes_dir, "nodes_resource_profiles.csv")
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "profileId:ID(ResourceProfile)", "name:string",
            "min_params:long", "max_params:string",
            "estimated_vram_gb:float", ":LABEL"
        ])
        for name, min_p, max_p, vram in RESOURCE_PROFILES:
            max_str = str(int(max_p)) if max_p != float("inf") else "unlimited"
            writer.writerow([
                make_id_safe(name), name, min_p, max_str, vram, "ResourceProfile"
            ])
        writer.writerow([
            "unknown", "Unknown", 0, "unknown", 0, "ResourceProfile"
        ])
    print(f"  [OK] {filepath} ({len(RESOURCE_PROFILES) + 1} records)")

    # ── Node: Authors ──
    filepath = os.path.join(nodes_dir, "nodes_authors.csv")
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "authorId:ID(Author)", "name:string", "model_count:int", ":LABEL"
        ])
        for author, count in sorted(data["authors"].items(), key=lambda x: -x[1]):
            writer.writerow([
                author, author, count, "Author"
            ])
    print(f"  [OK] {filepath} ({len(data['authors']):,} records)")

    # ── Node: Frameworks ──
    filepath = os.path.join(nodes_dir, "nodes_frameworks.csv")
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frameworkId:ID(Framework)", "name:string", ":LABEL"
        ])
        for fw in data["frameworks"]:
            writer.writerow([make_id_safe(fw), fw, "Framework"])
    print(f"  [OK] {filepath} ({len(data['frameworks'])} records)")

    # ── Relationships ──
    def write_rel_csv(filename, start_id_space, end_id_space, rel_type, rels):
        filepath = os.path.join(rels_dir, filename)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                f":START_ID({start_id_space})",
                f":END_ID({end_id_space})",
                ":TYPE"
            ])
            for start, end in rels:
                end_id = make_id_safe(end) if end_id_space != "Model" else end
                writer.writerow([start, end_id, rel_type])
        print(f"  [OK] {filepath} ({len(rels):,} records)")

    write_rel_csv("rels_model_domain.csv", "Model", "Domain", "BELONGS_TO_DOMAIN", data["domain_rels"])
    write_rel_csv("rels_model_task.csv", "Model", "Task", "PERFORMS_TASK", data["task_rels"])
    write_rel_csv("rels_model_license.csv", "Model", "License", "LICENSED_AS", data["license_rels"])

    # Resource rels - special because ID is direct name
    filepath = os.path.join(rels_dir, "rels_model_resource.csv")
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([":START_ID(Model)", ":END_ID(ResourceProfile)", ":TYPE"])
        for model_id, profile in data["resource_rels"]:
            writer.writerow([model_id, make_id_safe(profile), "REQUIRES_RESOURCES"])
    print(f"  [OK] {filepath} ({len(data['resource_rels']):,} records)")

    # Author rels - author ID is the author name
    filepath = os.path.join(rels_dir, "rels_model_author.csv")
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([":START_ID(Model)", ":END_ID(Author)", ":TYPE"])
        for model_id, author in data["author_rels"]:
            writer.writerow([model_id, author, "CREATED_BY"])
    print(f"  [OK] {filepath} ({len(data['author_rels']):,} records)")

    # Framework rels
    write_rel_csv("rels_model_framework.csv", "Model", "Framework", "USES_FRAMEWORK", data["framework_rels"])

    # Base model rels (Model -> Model)
    filepath = os.path.join(rels_dir, "rels_model_base_model.csv")
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([":START_ID(Model)", ":END_ID(Model)", ":TYPE"])
        # Only include base models that exist in our dataset
        known_models = {m["modelId"] for m in data["models"]}
        valid_count = 0
        for model_id, base_id in data["base_model_rels"]:
            writer.writerow([model_id, base_id, "BASED_ON"])
            valid_count += 1
    print(f"  [OK] {filepath} ({valid_count:,} records, including refs to models outside dataset)")

    print(f"\n[DONE] All CSV files generated in {OUTPUT_DIR}")


def generate_stats(data):
    """Print and save statistics about the dataset."""
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    total = len(data["models"])
    print(f"\n  Total models: {total:,}")

    # Domain distribution
    print(f"\n  {'─' * 50}")
    print(f"  --- DOMAIN DISTRIBUTION ---")
    print(f"  {'─' * 50}")
    domain_counts = defaultdict(int)
    for _, d in data["domain_rels"]:
        domain_counts[d] += 1
    for d, c in sorted(domain_counts.items(), key=lambda x: -x[1]):
        bar = "█" * min(40, c * 40 // max(domain_counts.values()))
        print(f"    {d:30s} {c:>6,} {bar}")

    # Task distribution (top 20)
    print(f"\n  {'─' * 50}")
    print(f"  --- TASK DISTRIBUTION (Top 20) ---")
    print(f"  {'─' * 50}")
    task_counts = defaultdict(int)
    for _, t in data["task_rels"]:
        task_counts[t] += 1
    for t, c in sorted(task_counts.items(), key=lambda x: -x[1])[:20]:
        bar = "█" * min(40, c * 40 // max(task_counts.values()))
        print(f"    {t:40s} {c:>6,} {bar}")

    # License distribution
    print(f"\n  {'─' * 50}")
    print(f"  --- LICENSE DISTRIBUTION (Top 15) ---")
    print(f"  {'─' * 50}")
    license_counts = defaultdict(int)
    for _, l in data["license_rels"]:
        license_counts[l] += 1
    for l, c in sorted(license_counts.items(), key=lambda x: -x[1])[:15]:
        bar = "█" * min(40, c * 40 // max(license_counts.values()))
        print(f"    {l:30s} {c:>6,} {bar}")

    # Resource profile distribution
    print(f"\n  {'─' * 50}")
    print(f"  --- RESOURCE PROFILE DISTRIBUTION ---")
    print(f"  {'─' * 50}")
    resource_counts = defaultdict(int)
    for _, r in data["resource_rels"]:
        resource_counts[r] += 1
    for r, c in sorted(resource_counts.items(), key=lambda x: -x[1]):
        bar = "█" * min(40, c * 40 // max(resource_counts.values()))
        print(f"    {r:20s} {c:>6,} {bar}")

    # Framework distribution (top 10)
    print(f"\n  {'─' * 50}")
    print(f"  --- FRAMEWORK DISTRIBUTION (Top 10) ---")
    print(f"  {'─' * 50}")
    framework_counts = defaultdict(int)
    for _, fw in data["framework_rels"]:
        framework_counts[fw] += 1
    for fw, c in sorted(framework_counts.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * min(40, c * 40 // max(framework_counts.values()))
        print(f"    {fw:30s} {c:>6,} {bar}")

    # Top 10 authors by model count
    print(f"\n  {'─' * 50}")
    print(f"  --- TOP 10 AUTHORS BY MODEL COUNT ---")
    print(f"  {'─' * 50}")
    for author, count in sorted(data["authors"].items(), key=lambda x: -x[1])[:10]:
        print(f"    {author:40s} {count:>6,} models")

    # Base model relationships
    print(f"\n  {'─' * 50}")
    print(f"  --- RELATIONSHIPS SUMMARY ---")
    print(f"  {'─' * 50}")
    print(f"    Model-Domain:    {len(data['domain_rels']):>8,}")
    print(f"    Model-Task:      {len(data['task_rels']):>8,}")
    print(f"    Model-License:   {len(data['license_rels']):>8,}")
    print(f"    Model-Resource:  {len(data['resource_rels']):>8,}")
    print(f"    Model-Author:    {len(data['author_rels']):>8,}")
    print(f"    Model-Framework: {len(data['framework_rels']):>8,}")
    print(f"    Model-BaseModel: {len(data['base_model_rels']):>8,}")

    print(f"\n{'=' * 70}")

    # Save stats to file
    stats_file = os.path.join(OUTPUT_DIR, "dataset_stats.json")
    stats = {
        "total_models": total,
        "total_domains": len(data["domains"]),
        "total_tasks": len(data["tasks"]),
        "total_licenses": len(data["licenses"]),
        "total_authors": len(data["authors"]),
        "total_frameworks": len(data["frameworks"]),
        "domain_distribution": dict(sorted(domain_counts.items(), key=lambda x: -x[1])),
        "task_distribution": dict(sorted(task_counts.items(), key=lambda x: -x[1])),
        "license_distribution": dict(sorted(license_counts.items(), key=lambda x: -x[1])),
        "resource_distribution": dict(sorted(resource_counts.items(), key=lambda x: -x[1])),
        "framework_distribution": dict(sorted(framework_counts.items(), key=lambda x: -x[1])),
        "relationships": {
            "model_domain": len(data["domain_rels"]),
            "model_task": len(data["task_rels"]),
            "model_license": len(data["license_rels"]),
            "model_resource": len(data["resource_rels"]),
            "model_author": len(data["author_rels"]),
            "model_framework": len(data["framework_rels"]),
            "model_base_model": len(data["base_model_rels"]),
        },
        "generated_at": datetime.now().isoformat(),
    }
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n  [FILE] Stats saved to {stats_file}")


def generate_neo4j_guide(data):
    """Generate a Neo4j import guide."""
    guide_path = os.path.join(OUTPUT_DIR, "neo4j_import_guide.md")
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write("""# Neo4j Import Guide - HuggingFace Model Catalog

## Quick Overview

This dataset contains **{total:,}** top Hugging Face models organized into a graph structure with:
- **{domains}** domain categories
- **{tasks}** task types
- **{licenses}** license types
- **{frameworks}** ML frameworks
- **{authors:,}** model authors/organizations
- **7** resource profiles (Nano → XXLarge)

---

## Option 1: neo4j-admin import (Recommended for initial load)

```bash
neo4j-admin database import full huggingface_catalog \\
  --nodes=nodes/nodes_models.csv \\
  --nodes=nodes/nodes_domains.csv \\
  --nodes=nodes/nodes_tasks.csv \\
  --nodes=nodes/nodes_licenses.csv \\
  --nodes=nodes/nodes_resource_profiles.csv \\
  --nodes=nodes/nodes_authors.csv \\
  --nodes=nodes/nodes_frameworks.csv \\
  --relationships=relationships/rels_model_domain.csv \\
  --relationships=relationships/rels_model_task.csv \\
  --relationships=relationships/rels_model_license.csv \\
  --relationships=relationships/rels_model_resource.csv \\
  --relationships=relationships/rels_model_author.csv \\
  --relationships=relationships/rels_model_framework.csv \\
  --relationships=relationships/rels_model_base_model.csv \\
  --trim-strings=true \\
  --skip-bad-relationships=true \\
  --array-delimiter=";" \\
  --overwrite-destination=true
```

## Option 2: LOAD CSV (Cypher) - For incremental loads

### Step 1: Create Constraints

```cypher
CREATE CONSTRAINT model_id FOR (m:Model) REQUIRE m.modelId IS UNIQUE;
CREATE CONSTRAINT domain_id FOR (d:Domain) REQUIRE d.domainId IS UNIQUE;
CREATE CONSTRAINT task_id FOR (t:Task) REQUIRE t.taskId IS UNIQUE;
CREATE CONSTRAINT license_id FOR (l:License) REQUIRE l.licenseId IS UNIQUE;
CREATE CONSTRAINT resource_id FOR (r:ResourceProfile) REQUIRE r.profileId IS UNIQUE;
CREATE CONSTRAINT author_id FOR (a:Author) REQUIRE a.authorId IS UNIQUE;
CREATE CONSTRAINT framework_id FOR (f:Framework) REQUIRE f.frameworkId IS UNIQUE;
```

### Step 2: Load Nodes

```cypher
// Load Models
LOAD CSV WITH HEADERS FROM 'file:///nodes/nodes_models.csv' AS row
CREATE (m:Model {{
  modelId: row['modelId:ID(Model)'],
  name: row['name:string'],
  author: row['author:string'],
  pipeline_tag: row['pipeline_tag:string'],
  downloads: toInteger(row['downloads:long']),
  likes: toInteger(row['likes:int']),
  library_name: row['library_name:string'],
  license: row['license:string'],
  created_at: row['created_at:string'],
  last_modified: row['last_modified:string'],
  num_parameters: CASE WHEN row['num_parameters:long'] <> '' THEN toInteger(row['num_parameters:long']) ELSE null END,
  size_label: row['size_label:string'],
  resource_profile: row['resource_profile:string'],
  languages: split(row['languages:string[]'], ';'),
  gated: toBoolean(row['gated:boolean']),
  private: toBoolean(row['private:boolean']),
  tags_count: toInteger(row['tags_count:int'])
}});

// Load Domains
LOAD CSV WITH HEADERS FROM 'file:///nodes/nodes_domains.csv' AS row
CREATE (d:Domain {{
  domainId: row['domainId:ID(Domain)'],
  name: row['name:string'],
  description: row['description:string']
}});

// Load Tasks
LOAD CSV WITH HEADERS FROM 'file:///nodes/nodes_tasks.csv' AS row
CREATE (t:Task {{
  taskId: row['taskId:ID(Task)'],
  name: row['name:string'],
  category: row['category:string']
}});

// Load Licenses
LOAD CSV WITH HEADERS FROM 'file:///nodes/nodes_licenses.csv' AS row
CREATE (l:License {{
  licenseId: row['licenseId:ID(License)'],
  name: row['name:string'],
  type: row['type:string'],
  commercial_use: row['commercial_use:string']
}});

// Load Resource Profiles
LOAD CSV WITH HEADERS FROM 'file:///nodes/nodes_resource_profiles.csv' AS row
CREATE (r:ResourceProfile {{
  profileId: row['profileId:ID(ResourceProfile)'],
  name: row['name:string'],
  min_params: toInteger(row['min_params:long']),
  max_params: row['max_params:string'],
  estimated_vram_gb: toFloat(row['estimated_vram_gb:float'])
}});

// Load Authors
LOAD CSV WITH HEADERS FROM 'file:///nodes/nodes_authors.csv' AS row
CREATE (a:Author {{
  authorId: row['authorId:ID(Author)'],
  name: row['name:string'],
  model_count: toInteger(row['model_count:int'])
}});

// Load Frameworks
LOAD CSV WITH HEADERS FROM 'file:///nodes/nodes_frameworks.csv' AS row
CREATE (f:Framework {{
  frameworkId: row['frameworkId:ID(Framework)'],
  name: row['name:string']
}});
```

### Step 3: Load Relationships

```cypher
// Model -> Domain
LOAD CSV WITH HEADERS FROM 'file:///relationships/rels_model_domain.csv' AS row
MATCH (m:Model {{modelId: row[':START_ID(Model)']}})
MATCH (d:Domain {{domainId: row[':END_ID(Domain)']}})
CREATE (m)-[:BELONGS_TO_DOMAIN]->(d);

// Model -> Task
LOAD CSV WITH HEADERS FROM 'file:///relationships/rels_model_task.csv' AS row
MATCH (m:Model {{modelId: row[':START_ID(Model)']}})
MATCH (t:Task {{taskId: row[':END_ID(Task)']}})
CREATE (m)-[:PERFORMS_TASK]->(t);

// Model -> License
LOAD CSV WITH HEADERS FROM 'file:///relationships/rels_model_license.csv' AS row
MATCH (m:Model {{modelId: row[':START_ID(Model)']}})
MATCH (l:License {{licenseId: row[':END_ID(License)']}})
CREATE (m)-[:LICENSED_AS]->(l);

// Model -> Resource Profile
LOAD CSV WITH HEADERS FROM 'file:///relationships/rels_model_resource.csv' AS row
MATCH (m:Model {{modelId: row[':START_ID(Model)']}})
MATCH (r:ResourceProfile {{profileId: row[':END_ID(ResourceProfile)']}})
CREATE (m)-[:REQUIRES_RESOURCES]->(r);

// Model -> Author
LOAD CSV WITH HEADERS FROM 'file:///relationships/rels_model_author.csv' AS row
MATCH (m:Model {{modelId: row[':START_ID(Model)']}})
MATCH (a:Author {{authorId: row[':END_ID(Author)']}})
CREATE (m)-[:CREATED_BY]->(a);

// Model -> Framework
LOAD CSV WITH HEADERS FROM 'file:///relationships/rels_model_framework.csv' AS row
MATCH (m:Model {{modelId: row[':START_ID(Model)']}})
MATCH (f:Framework {{frameworkId: row[':END_ID(Framework)']}})
CREATE (m)-[:USES_FRAMEWORK]->(f);

// Model -> Base Model
LOAD CSV WITH HEADERS FROM 'file:///relationships/rels_model_base_model.csv' AS row
MATCH (m:Model {{modelId: row[':START_ID(Model)']}})
MATCH (bm:Model {{modelId: row[':END_ID(Model)']}})
CREATE (m)-[:BASED_ON]->(bm);
```

---

## Example Queries

### Find all models in a domain
```cypher
MATCH (m:Model)-[:BELONGS_TO_DOMAIN]->(d:Domain {{name: 'Medicine / Healthcare'}})
RETURN m.modelId, m.downloads, m.likes
ORDER BY m.downloads DESC LIMIT 20;
```

### Find models by task and license type
```cypher
MATCH (m:Model)-[:PERFORMS_TASK]->(t:Task {{name: 'text-generation'}})
MATCH (m)-[:LICENSED_AS]->(l:License {{type: 'Permissive'}})
RETURN m.modelId, m.downloads, l.name
ORDER BY m.downloads DESC LIMIT 50;
```

### Find the most popular model families (base models)
```cypher
MATCH (derived:Model)-[:BASED_ON]->(base:Model)
RETURN base.modelId, count(derived) AS derivative_count
ORDER BY derivative_count DESC LIMIT 20;
```

### Resource analysis: how many models per resource tier
```cypher
MATCH (m:Model)-[:REQUIRES_RESOURCES]->(r:ResourceProfile)
RETURN r.name, count(m) AS model_count, avg(m.downloads) AS avg_downloads
ORDER BY model_count DESC;
```

### Cross-domain analysis
```cypher
MATCH (m:Model)-[:BELONGS_TO_DOMAIN]->(d1:Domain)
MATCH (m)-[:BELONGS_TO_DOMAIN]->(d2:Domain)
WHERE d1.name < d2.name
RETURN d1.name, d2.name, count(m) AS shared_models
ORDER BY shared_models DESC LIMIT 20;
```

### Author productivity
```cypher
MATCH (a:Author)<-[:CREATED_BY]-(m:Model)
RETURN a.name, count(m) AS models, sum(m.downloads) AS total_downloads
ORDER BY total_downloads DESC LIMIT 20;
```

---

## Graph Schema Summary

```
(Model) -[:BELONGS_TO_DOMAIN]-> (Domain)
(Model) -[:PERFORMS_TASK]-> (Task)
(Model) -[:LICENSED_AS]-> (License)
(Model) -[:REQUIRES_RESOURCES]-> (ResourceProfile)
(Model) -[:CREATED_BY]-> (Author)
(Model) -[:USES_FRAMEWORK]-> (Framework)
(Model) -[:BASED_ON]-> (Model)
```

""".format(
            total=len(data["models"]),
            domains=len(data["domains"]),
            tasks=len(data["tasks"]),
            licenses=len(data["licenses"]),
            frameworks=len(data["frameworks"]),
            authors=len(data["authors"]),
        ))
    print(f"  📄 Import guide saved to {guide_path}")


def main():
    """Main execution flow."""
    target = TARGET_MODELS
    if len(sys.argv) > 2 and sys.argv[1] == "--target":
        target = int(sys.argv[2])

    start_time = time.time()

    # Step 1: Fetch models
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    cache_file = os.path.join(RAW_DATA_DIR, "raw_models.json")

    if os.path.exists(cache_file):
        print(f">>> Loading cached data from {cache_file}...")
        with open(cache_file, "r", encoding="utf-8") as f:
            raw_models = json.load(f)
        print(f"   Loaded {len(raw_models):,} models from cache")
        if len(raw_models) < target:
            print(f"   Cache has fewer than {target} models, re-fetching...")
            raw_models = fetch_all_models(target)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(raw_models, f, ensure_ascii=False)
            print(f"   Saved to cache: {cache_file}")
    else:
        raw_models = fetch_all_models(target)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(raw_models, f, ensure_ascii=False)
        print(f"   Saved to cache: {cache_file}")

    # Step 2: Process and classify
    data = process_models(raw_models)

    # Step 3: Generate CSVs
    generate_csvs(data)

    # Step 4: Generate import guide
    generate_neo4j_guide(data)

    # Step 5: Statistics
    generate_stats(data)

    elapsed = time.time() - start_time
    print(f"\n[DONE] Finished! Total time: {elapsed:.1f}s")
    print(f"   Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
