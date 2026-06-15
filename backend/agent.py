"""Evidence-based agentic dataset discovery."""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterator

from backend.search import inspect_dataset, search_datasets

MODEL = os.getenv("WEAVER_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct")
MAX_TASK_LENGTH = 2000

_local_model = None
_local_tokenizer = None
_model_load_failed = False
_model_lock = threading.Lock()
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "your", "have",
    "need", "want", "like", "build", "using", "data", "dataset", "model",
    "small", "find", "looking", "project", "create", "make", "into",
}
_LANGUAGES = {
    "english": "en", "arabic": "ar", "french": "fr", "german": "de",
    "spanish": "es", "italian": "it", "portuguese": "pt", "chinese": "zh",
    "japanese": "ja", "korean": "ko", "hindi": "hi", "multilingual": "multilingual",
}
_MODALITIES = {
    "audio": ("audio", "speech", "voice", "asr"),
    "image": ("image", "vision", "photo", "ocr"),
    "video": ("video",),
    "tabular": ("tabular", "table", "csv", "classification"),
    "text": ("text", "document", "summarization", "translation", "intent", "chat"),
}
_LABEL_TERMS = {"label", "labels", "class", "classes", "intent", "category", "target"}
_TASK_TYPES = {
    "intent classification", "classification", "summarization", "translation",
    "question answering", "retrieval", "fine-tuning", "pretraining", "dataset discovery",
}


def _llm(system: str, user: str, max_tokens: int = 350, temperature: float = 0.2) -> str | None:
    """Run the planning step locally with a genuinely small model."""
    global _local_model, _local_tokenizer, _model_load_failed
    if _model_load_failed:
        return None
    if _local_model is None or _local_tokenizer is None:
        try:
            with _model_lock:
                if _local_model is None or _local_tokenizer is None:
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    _local_tokenizer = AutoTokenizer.from_pretrained(MODEL)
                    _local_model = AutoModelForCausalLM.from_pretrained(
                        MODEL,
                        low_cpu_mem_usage=True,
                    )
                    _local_model.eval()
        except Exception:
            _model_load_failed = True
            return None
    try:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = _local_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = _local_tokenizer(prompt, return_tensors="pt")
        generation_args = {
            "max_new_tokens": min(max_tokens, 220),
            "do_sample": temperature > 0,
            "pad_token_id": _local_tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_args["temperature"] = temperature
        with _model_lock:
            output = _local_model.generate(
                **inputs,
                **generation_args,
            )
        generated = output[0, inputs["input_ids"].shape[1]:]
        return _local_tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )
    except Exception:
        return None


def _extract_json(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(cleaned[start:end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
    return {}


def _keywords(text: str) -> list[str]:
    return [word.lower() for word in _WORD_RE.findall(text) if word.lower() not in _STOPWORDS]


def parse_task(task: str, use_llm: bool = True) -> tuple[dict[str, Any], bool]:
    """Turn a free-form project description into explicit search requirements."""
    lower = task.lower()
    languages = [code for name, code in _LANGUAGES.items() if name in lower]
    modalities = [
        modality
        for modality, terms in _MODALITIES.items()
        if any(term in lower for term in terms)
    ]
    if not modalities:
        modalities = ["text"]
    required_fields = []
    if any(term in lower for term in _LABEL_TERMS):
        required_fields.append("label")
    for field in ("question", "answer", "instruction", "response", "summary", "translation"):
        if field in lower:
            required_fields.append(field)
    profile: dict[str, Any] = {
        "languages": languages,
        "modalities": modalities,
        "task_type": "classification" if "classifier" in lower else next(
            (
                name for name in (
                    "intent classification", "classification", "summarization", "translation",
                    "question answering", "retrieval", "fine-tuning", "pretraining",
                )
                if name in lower
            ),
            "dataset discovery",
        ),
        "required_fields": list(dict.fromkeys(required_fields)),
        "license": "commercial-friendly" if any(
            term in lower for term in ("commercial", "production", "apache", "mit")
        ) else "",
        "size_preference": "small" if any(
            term in lower for term in ("small", "tiny", "prototype", "quick")
        ) else "",
        "domain_keywords": _keywords(task)[:12],
        "intended_use": task[:280],
    }

    llm_used = False
    if use_llm:
        prompt = (
            f"Request: {task}\n"
            "Return exactly one compact JSON object. Use at most 5 domain_keywords. "
            "Use an empty string or empty list when a requirement is not explicit. "
            'Schema: {"languages":[],"modalities":[],"task_type":"","required_fields":[],'
            '"license":"","size_preference":"","domain_keywords":[]}'
        )
        parsed = _extract_json(_llm(
            "Extract only requirements explicitly stated. No prose. No repetition.",
            prompt,
            max_tokens=120,
            temperature=0,
        ))
        if parsed:
            llm_used = True
            proposed_languages = [
                str(item).lower() for item in parsed.get("languages", [])
                if isinstance(item, str)
                and (len(item.strip()) in {2, 3} or item.lower() == "multilingual")
            ]
            proposed_modalities = [
                str(item).lower() for item in parsed.get("modalities", [])
                if str(item).lower() in _MODALITIES
            ]
            proposed_fields = [
                str(item).lower() for item in parsed.get("required_fields", [])
                if isinstance(item, str) and len(item) <= 30
            ]
            proposed_keywords = [
                str(item).lower() for item in parsed.get("domain_keywords", [])
                if isinstance(item, str) and 2 < len(item) <= 30
            ]
            profile["languages"] = list(dict.fromkeys(profile["languages"] + proposed_languages))
            profile["modalities"] = list(dict.fromkeys(profile["modalities"] + proposed_modalities))
            profile["required_fields"] = list(
                dict.fromkeys(profile["required_fields"] + proposed_fields)
            )
            profile["domain_keywords"] = list(
                dict.fromkeys(profile["domain_keywords"] + proposed_keywords)
            )[:12]
            proposed_task = str(parsed.get("task_type") or "").lower()
            if profile["task_type"] == "dataset discovery" and proposed_task in _TASK_TYPES:
                profile["task_type"] = proposed_task
            proposed_license = str(parsed.get("license") or "").lower()
            if profile["license"] == "" and proposed_license and proposed_license in lower:
                profile["license"] = proposed_license
            proposed_size = str(parsed.get("size_preference") or "").lower()
            if profile["size_preference"] == "" and proposed_size in {"small", "medium", "large"}:
                if proposed_size in lower:
                    profile["size_preference"] = proposed_size
    return profile, llm_used


def generate_queries(task: str, profile: dict[str, Any]) -> list[str]:
    ignored = {
        *profile["languages"],
        *_LANGUAGES.keys(),
        "labels", "label", "classifier", "classification", "compact",
        "corpus", "examples", "records",
    }
    terms = [term for term in profile["domain_keywords"] if term not in ignored]
    task_type = profile["task_type"]
    queries = [
        " ".join(terms[:3]),
        " ".join(terms[:2]),
        " ".join(terms[-1:] + [task_type]),
        " ".join(terms[1:2] + profile["required_fields"][:1]),
    ]
    cleaned = []
    for query in queries:
        normalized = re.sub(r"\s+", " ", query).strip()
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned[:4] or [" ".join(_keywords(task)[:4])]


def _text_blob(dataset: dict[str, Any]) -> str:
    values = [
        dataset.get("id", ""),
        dataset.get("description", ""),
        " ".join(dataset.get("tags", [])),
        " ".join(dataset.get("features", [])),
        " ".join(dataset.get("task_categories", [])),
    ]
    return " ".join(values).lower()


def _pre_score(profile: dict[str, Any], dataset: dict[str, Any]) -> float:
    blob = _text_blob(dataset)
    keywords = set(profile["domain_keywords"])
    overlap = sum(1 for word in keywords if word in blob)
    modality_matches = sum(1 for value in profile["modalities"] if value in dataset.get("modalities", []))
    language_matches = sum(1 for value in profile["languages"] if value in dataset.get("languages", []))
    popularity = math.log10(1 + dataset.get("downloads", 0) + dataset.get("likes", 0) * 10)
    return overlap * 8 + modality_matches * 12 + language_matches * 8 + popularity


def _contains_any(values: list[str], expected: list[str]) -> bool:
    lowered = {str(value).lower() for value in values}
    return any(item.lower() in lowered for item in expected)


def score_dataset(profile: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    """Compute a transparent score entirely from collected evidence."""
    blob = _text_blob(dataset)
    fields = [field.lower() for field in dataset.get("features", [])]
    sample_fields = {
        str(key).lower()
        for row in dataset.get("sample_rows", [])
        if isinstance(row, dict)
        for key in row
    }
    available_fields = set(fields) | sample_fields
    requested = set(profile["domain_keywords"])
    matched_keywords = sorted(word for word in requested if word in blob)

    relevance = min(35, round(35 * len(matched_keywords) / max(3, len(requested))))
    modality_values = dataset.get("modalities", [])
    modality = 15 if _contains_any(modality_values, profile["modalities"]) else 0
    if not modality_values:
        modality = 7
    language_values = dataset.get("languages", [])
    language = 10 if not profile["languages"] or _contains_any(
        language_values, profile["languages"]
    ) else 0
    if profile["languages"] and not language_values:
        language = 4

    required_fields = profile["required_fields"]
    matched_fields = [
        requirement
        for requirement in required_fields
        if any(requirement in field for field in available_fields)
        or requirement in blob
    ]
    schema = 15 if not required_fields else round(15 * len(matched_fields) / len(required_fields))
    if required_fields and not available_fields:
        schema = min(schema, 5)

    license_value = dataset.get("license", "")
    permissive = {"apache-2.0", "mit", "cc-by-4.0", "cc0-1.0", "odc-by"}
    license_score = 10 if license_value in permissive else 5 if license_value else 0
    documentation = 5 if dataset.get("card_complete") else 2 if dataset.get("description") else 0
    popularity = min(
        5,
        round(math.log10(1 + dataset.get("downloads", 0) + dataset.get("likes", 0) * 20)),
    )
    accessibility = 5 if dataset.get("accessible") and not dataset.get("gated") else 0
    total = relevance + modality + language + schema + license_score + documentation + popularity + accessibility

    checks = {
        "modality": "pass" if modality == 15 else "unknown" if not modality_values else "fail",
        "language": "pass" if language == 10 else "unknown" if not language_values else "fail",
        "required_fields": "pass" if not required_fields or len(matched_fields) == len(required_fields)
        else "unknown" if not available_fields else "fail",
        "license": "pass" if license_score == 10 else "unknown" if not license_value else "review",
        "accessible": "pass" if accessibility == 5 else "fail",
    }
    rejection_reasons = []
    if checks["accessible"] == "fail":
        rejection_reasons.append("Dataset could not be inspected or is gated/private.")
    if checks["modality"] == "fail":
        rejection_reasons.append(
            f"Modality {modality_values} does not match requested {profile['modalities']}."
        )
    if checks["required_fields"] == "fail":
        rejection_reasons.append(
            f"Required fields {required_fields} were not found in the inspected schema."
        )
    if checks["language"] == "fail":
        rejection_reasons.append(
            f"Languages {language_values} do not match requested {profile['languages']}."
        )
    status = "rejected" if rejection_reasons else "recommended" if total >= 72 else "conditional"
    evidence = [
        f"Matched project terms: {', '.join(matched_keywords) or 'none verified'}",
        f"Modalities: {', '.join(modality_values) or 'not declared'}",
        f"Languages: {', '.join(language_values) or 'not declared'}",
        f"Features: {', '.join(sorted(available_fields)[:10]) or 'viewer schema unavailable'}",
        f"License: {license_value or 'not declared'}",
    ]
    strength = (
        f"Verified {len(matched_keywords)} project terms"
        + (f" and fields {', '.join(matched_fields)}" if matched_fields else "")
        + "."
    )
    weakness = rejection_reasons[0] if rejection_reasons else next(
        (
            label for key, label in (
                ("required_fields", "Required schema fields need manual confirmation."),
                ("license", "License needs manual review."),
                ("language", "Language coverage is not declared."),
            )
            if checks[key] in {"unknown", "review"}
        ),
        "No major metadata gap detected.",
    )
    recommendation = {
        "recommended": "Strong candidate for a first experiment; inspect sample rows before training.",
        "conditional": "Promising candidate, but resolve the highlighted evidence gaps first.",
        "rejected": "Do not use for this request unless the project requirements change.",
    }[status]
    quality = round(
        (schema + license_score + documentation + popularity + accessibility) / 40 * 100
    )
    return {
        **dataset,
        "score": total,
        "relevance": round((relevance + modality + language) / 60 * 100),
        "quality": min(100, quality),
        "status": status,
        "score_breakdown": {
            "project_match": relevance,
            "modality": modality,
            "language": language,
            "schema": schema,
            "license": license_score,
            "documentation": documentation,
            "adoption": popularity,
            "accessibility": accessibility,
        },
        "checks": checks,
        "evidence": evidence,
        "rejection_reasons": rejection_reasons,
        "strength": strength,
        "weakness": weakness,
        "recommendation": recommendation,
    }


def _cross_reference(datasets: list[dict[str, Any]]) -> list[dict[str, str]]:
    candidates = [dataset for dataset in datasets if dataset["status"] != "rejected"][:6]
    pairs = []
    for index, first in enumerate(candidates):
        for second in candidates[index + 1:]:
            first_terms = set(first.get("modalities", []) + first.get("languages", []))
            second_terms = set(second.get("modalities", []) + second.get("languages", []))
            if first_terms != second_terms or set(first.get("features", [])) != set(second.get("features", [])):
                pairs.append({
                    "from": first["id"],
                    "to": second["id"],
                    "label": "complementary coverage",
                })
            if len(pairs) >= 6:
                return pairs
    return pairs


def weave_events(task: str, max_datasets: int = 8) -> Iterator[dict[str, Any]]:
    task = task.strip()
    if not task:
        raise ValueError("Task description is required.")
    if len(task) > MAX_TASK_LENGTH:
        raise ValueError(f"Task description must be {MAX_TASK_LENGTH} characters or fewer.")

    started = time.time()
    yield {"type": "started", "task": task, "message": "Research session started."}
    profile, llm_used = parse_task(task)
    queries = generate_queries(task, profile)
    yield {
        "type": "plan",
        "profile": profile,
        "queries": queries,
        "model_used": MODEL if llm_used else None,
        "fallback_used": not llm_used,
        "message": "Converted the request into explicit dataset requirements.",
    }

    collected: dict[str, dict[str, Any]] = {}
    for query in queries:
        found = search_datasets(query, limit=12)
        for dataset in found:
            current = collected.get(dataset["id"])
            if current is None or _pre_score(profile, dataset) > _pre_score(profile, current):
                collected[dataset["id"]] = dataset
        yield {
            "type": "search",
            "query": query,
            "found": len(found),
            "unique": len(collected),
            "message": f"Searched “{query}” and found {len(found)} candidates.",
        }

    pre_ranked = sorted(
        collected.values(),
        key=lambda dataset: _pre_score(profile, dataset),
        reverse=True,
    )[:max_datasets]
    inspected: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(pre_ranked)))) as pool:
        futures = {
            pool.submit(inspect_dataset, dataset["id"], dataset): dataset["id"]
            for dataset in pre_ranked
        }
        for future in as_completed(futures):
            dataset_id = futures[future]
            try:
                evidence = future.result()
            except Exception as exc:
                evidence = {
                    **next(item for item in pre_ranked if item["id"] == dataset_id),
                    "accessible": False,
                    "inspection_error": str(exc),
                    "features": [],
                    "sample_rows": [],
                    "configs": [],
                    "splits": [],
                }
            scored = score_dataset(profile, evidence)
            inspected.append(scored)
            yield {
                "type": "inspect",
                "dataset_id": dataset_id,
                "status": scored["status"],
                "score": scored["score"],
                "checks": scored["checks"],
                "message": f"Inspected {dataset_id}: {scored['status']} ({scored['score']}/100).",
            }
            yield {"type": "candidate", "dataset": _public_dataset(scored)}

    ranked = sorted(
        inspected,
        key=lambda dataset: (
            dataset["status"] == "recommended",
            dataset["status"] == "conditional",
            dataset["score"],
        ),
        reverse=True,
    )
    pairs = _cross_reference(ranked)
    nodes = [
        {
            "id": dataset["id"],
            "score": dataset["score"],
            "status": dataset["status"],
            "downloads": dataset.get("downloads", 0),
        }
        for dataset in ranked
    ]
    result = {
        "task": task,
        "profile": profile,
        "queries": queries,
        "datasets": [_public_dataset(dataset) for dataset in ranked],
        "nodes": nodes,
        "threads": pairs,
        "top_pick": next(
            (dataset["id"] for dataset in ranked if dataset["status"] != "rejected"),
            ranked[0]["id"] if ranked else None,
        ),
        "model_used": MODEL if llm_used else None,
        "fallback_used": not llm_used,
        "elapsed_ms": round((time.time() - started) * 1000),
    }
    yield {
        "type": "ranking",
        "top_pick": result["top_pick"],
        "count": len(ranked),
        "message": f"Ranked {len(ranked)} inspected candidates using verified evidence.",
    }
    yield {"type": "complete", "result": result, "message": "Dataset research complete."}


def _public_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "id", "author", "description", "downloads", "likes", "tags",
        "task_categories", "languages", "license", "size_category", "formats",
        "modalities", "configs", "splits", "features", "sample_rows", "hub_url",
        "accessible", "inspection_error", "card_complete", "score", "relevance",
        "quality", "status", "score_breakdown", "checks", "evidence",
        "rejection_reasons", "strength", "weakness", "recommendation",
    }
    return {key: value for key, value in dataset.items() if key in allowed}


def weave(task: str, max_datasets: int = 8) -> dict[str, Any]:
    result = None
    for event in weave_events(task, max_datasets=max_datasets):
        if event["type"] == "complete":
            result = event["result"]
    return result or {
        "task": task,
        "datasets": [],
        "nodes": [],
        "threads": [],
        "top_pick": None,
        "fallback_used": True,
    }
