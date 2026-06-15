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
    "about", "pairs", "examples", "records", "documents", "corpus",
    "evaluation", "evaluate", "permissive", "license", "compact",
    "abstractive", "extractive", "recordings", "transcripts",
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
    "tabular": ("tabular", "table", "csv", "structured"),
    "text": ("text", "document", "summarization", "translation", "intent", "chat"),
}
_LABEL_TERMS = {"label", "labels", "class", "classes", "intent", "category"}
_DIRECT_LABEL_FIELDS = {
    "label", "labels", "intent", "category", "class", "target",
    "detected_intent", "intent_label", "class_label",
}
_PROXY_LABEL_FIELDS = {"type", "queue", "topic", "department", "route", "routing"}
_TASK_TYPES = {
    "intent classification", "classification", "summarization", "translation",
    "question answering", "retrieval", "automatic speech recognition",
    "fine-tuning", "pretraining", "dataset discovery",
}
_TASK_ALIASES = {
    "intent classification": ("intent classification", "intent"),
    "classification": ("classification",),
    "summarization": ("summarization", "summary"),
    "translation": ("translation",),
    "question answering": ("question answering", "qa"),
    "retrieval": ("retrieval", "search"),
    "automatic speech recognition": ("speech recognition", "asr", "transcription"),
    "fine-tuning": ("instruction", "fine tuning"),
    "pretraining": ("pretraining",),
}
_FIELD_ALIASES = {
    "text": {
        "text", "sentence", "content", "document", "article", "body", "query",
        "utterance", "input_text", "text_input",
    },
    "label": _DIRECT_LABEL_FIELDS,
    "document": {
        "document", "article", "text", "content", "body", "source",
        "judgement", "judgment", "case_text", "legal_text",
    },
    "summary": {
        "summary", "highlights", "abstract", "target", "headline",
        "summarizer", "processed_text",
    },
    "question": {"question", "query", "prompt", "question_text", "instruction"},
    "answer": {"answer", "answers", "response", "context", "answer_text"},
    "source": {"source", "src", "text", "sentence", "input", "source_text", "input_text"},
    "target": {
        "target", "tgt", "translation", "translated_text", "output",
        "target_text", "output_text",
    },
    "audio": {"audio", "speech", "file", "path", "audio_path", "audio_file"},
    "transcript": {
        "transcript", "transcription", "sentence", "text",
        "transcript_text", "transcription_text", "label",
    },
    "instruction": {"instruction", "prompt", "input"},
    "response": {"response", "output", "completion", "answer"},
}


def _domain_terms(profile: dict[str, Any]) -> list[str]:
    task_words = {
        word
        for alias in _TASK_ALIASES.get(profile["task_type"], (profile["task_type"],))
        for word in alias.split()
    }
    ignored = {
        *profile["languages"],
        *_LANGUAGES.keys(),
        *task_words,
        *profile["required_fields"],
        "labels", "label", "classifier", "classification",
        "summarization", "summary", "translation", "retrieval", "search",
        "question", "answer", "speech", "recognition", "asr", "transcript",
        "text", "audio", "image", "video",
    }
    return [term for term in profile["domain_keywords"] if term not in ignored]


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
    normalized = text.replace("-", " ")
    return [word.lower() for word in _WORD_RE.findall(normalized) if word.lower() not in _STOPWORDS]


def _task_type(lower: str) -> str:
    if "intent" in lower and any(term in lower for term in ("classif", "label", "dataset", "data")):
        return "intent classification"
    if "summar" in lower:
        return "summarization"
    if "translat" in lower:
        return "translation"
    if "retrieval" in lower or "search evaluation" in lower:
        return "retrieval"
    if "question answer" in lower or re.search(r"\bqa\b", lower):
        return "question answering"
    if re.search(r"\basr\b", lower) or "speech recognition" in lower or "speech to text" in lower:
        return "automatic speech recognition"
    if "classif" in lower or "classifier" in lower:
        return "classification"
    if "fine-tun" in lower or "finetun" in lower:
        return "fine-tuning"
    if "pretrain" in lower:
        return "pretraining"
    return "dataset discovery"


def _default_required_fields(task_type: str) -> list[str]:
    return {
        "intent classification": ["text", "label"],
        "classification": ["text", "label"],
        "summarization": ["document", "summary"],
        "translation": ["source", "target"],
        "question answering": ["question", "answer"],
        "automatic speech recognition": ["audio", "transcript"],
    }.get(task_type, [])


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
    task_type = _task_type(lower)
    required_fields = _default_required_fields(task_type)
    word_set = set(_keywords(task))
    if task_type not in {"translation", "summarization", "question answering"}:
        if word_set & _LABEL_TERMS:
            required_fields.append("label")
    for field in ("question", "answer", "instruction", "response", "summary", "transcript"):
        if field in word_set:
            required_fields.append(field)
    profile: dict[str, Any] = {
        "languages": languages,
        "modalities": modalities,
        "task_type": task_type,
        "required_fields": list(dict.fromkeys(required_fields)),
        "license": "permissive" if any(
            term in lower for term in ("commercial", "production", "permissive", "apache", "mit")
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
    task_type = profile["task_type"]
    task_aliases = _TASK_ALIASES.get(task_type, (task_type,))
    terms = _domain_terms(profile)
    primary_task = task_aliases[0]
    compact_task = task_aliases[-1]
    language_names = [
        name for name, code in _LANGUAGES.items()
        if code in profile["languages"] and name != "multilingual"
    ]
    domain = terms[:2]
    field_terms = [field for field in profile["required_fields"] if field not in {"text", "document", "source", "target"}]
    queries = []
    if domain:
        task_for_domain = primary_task if len(domain) + len(primary_task.split()) <= 3 else compact_task
        queries.append(" ".join(domain + [task_for_domain]))
        queries.append(" ".join([domain[0], compact_task]))
        queries.append(" ".join(domain + ["dataset"]))
    if language_names:
        queries.append(" ".join([language_names[0], compact_task]))
    if language_names and domain:
        queries.append(" ".join([language_names[0], domain[0]]))
        queries.append(" ".join([language_names[0], domain[0], "dataset"]))
    elif len(language_names) >= 2:
        queries.append(" ".join(language_names[:2] + [compact_task]))
    if domain and {"question", "answer"}.issubset(profile["required_fields"]):
        queries.append(f"{domain[0]} question")
        queries.append(f"{domain[0]} qa")
        queries.append(" ".join(domain + ["qa"]))
    if domain and field_terms:
        queries.append(" ".join(domain[:1] + field_terms[:2]))
    if task_type == "automatic speech recognition":
        queries.append("speech transcription")
        queries.append("librispeech")
    if task_type == "intent classification":
        queries.append("intent dataset")
        if domain:
            queries.append(f"{domain[0]} intent")
    if domain:
        queries.append(" ".join(domain))
    queries.append(primary_task)
    if len(task_aliases) > 1:
        queries.append(task_aliases[-1])
    cleaned = []
    for query in queries:
        normalized = re.sub(r"\s+", " ", query).strip()
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned[:9] or [" ".join(_keywords(task)[:4])]


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
    keywords = set(_domain_terms(profile) or profile["domain_keywords"])
    overlap = sum(1 for word in keywords if word in blob)
    modality_matches = sum(1 for value in profile["modalities"] if value in dataset.get("modalities", []))
    language_matches = sum(1 for value in profile["languages"] if value in dataset.get("languages", []))
    task_terms = _TASK_ALIASES.get(profile["task_type"], (profile["task_type"],))
    task_match = sum(1 for term in task_terms if term in blob)
    schema_hint = sum(1 for field in profile["required_fields"] if field in blob)
    popularity = min(1.5, math.log10(1 + dataset.get("downloads", 0) + dataset.get("likes", 0) * 10) / 2)
    return overlap * 10 + task_match * 10 + schema_hint * 4 + modality_matches * 8 + language_matches * 8 + popularity


def _contains_any(values: list[str], expected: list[str]) -> bool:
    lowered = {str(value).lower() for value in values}
    return any(item.lower() in lowered for item in expected)


def _field_names(dataset: dict[str, Any]) -> set[str]:
    names = {str(field).lower() for field in dataset.get("features", [])}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                names.add(str(key).lower())
                visit(nested)
        elif isinstance(value, list):
            for nested in value[:5]:
                visit(nested)

    for row in dataset.get("sample_rows", []):
        visit(row)
    return names


def _sample_text(dataset: dict[str, Any]) -> str:
    values: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value[:5]:
                visit(nested)
        elif isinstance(value, str):
            values.append(value)

    for row in dataset.get("sample_rows", []):
        visit(row)
    return " ".join(values).lower()


def _infer_script_languages(text: str) -> list[str]:
    if not text:
        return []
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return []
    ranges = {
        "ar": lambda char: "\u0600" <= char <= "\u06ff",
        "zh": lambda char: "\u4e00" <= char <= "\u9fff",
        "ja": lambda char: "\u3040" <= char <= "\u30ff",
        "ko": lambda char: "\uac00" <= char <= "\ud7af",
    }
    return [
        language
        for language, matcher in ranges.items()
        if sum(1 for char in letters if matcher(char)) / len(letters) >= 0.05
    ]


def _matches_field(requirement: str, field: str) -> bool:
    aliases = _FIELD_ALIASES.get(requirement, {requirement})
    normalized = field.replace("-", "_").lower()
    return normalized in aliases


def score_dataset(profile: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    """Compute a transparent score entirely from collected evidence."""
    blob = _text_blob(dataset)
    available_fields = _field_names(dataset)
    sample_text = _sample_text(dataset)
    evidence_blob = f"{blob} {sample_text}"
    requested = {
        word for word in _domain_terms(profile)
        if word not in {"english", "labels", "label", "compact", "classifier", "dataset", "data"}
    }
    matched_keywords = sorted(word for word in requested if word in evidence_blob)
    domain_check = "pass" if not requested or matched_keywords else "fail"

    lexical_match = min(22, round(22 * len(matched_keywords) / max(2, len(requested))))
    task_type = profile["task_type"]
    task_terms = _TASK_ALIASES.get(task_type, (task_type,))
    task_match = 13 if any(term in evidence_blob for term in task_terms) else 0
    relevance = lexical_match + task_match
    modality_values = dataset.get("modalities", [])
    modality = 15 if _contains_any(modality_values, profile["modalities"]) else 0
    if not modality_values:
        modality = 7
    language_values = [str(value).lower() for value in dataset.get("languages", [])]
    inferred_languages = _infer_script_languages(sample_text) if not language_values else []
    language_evidence = language_values or inferred_languages
    requested_languages = {
        value for value in profile["languages"] if value != "multilingual"
    }
    declared_languages = set(language_evidence)
    if not requested_languages:
        language = 10
        language_check = "pass"
    elif not language_evidence:
        language = 4
        language_check = "unknown"
    elif requested_languages.issubset(declared_languages) or "multilingual" in declared_languages:
        language = 10
        language_check = "pass"
    elif requested_languages & declared_languages:
        language = 5
        language_check = "review"
    else:
        language = 0
        language_check = "fail"

    required_fields = profile["required_fields"]
    proxy_label_fields = sorted(field for field in available_fields if field in _PROXY_LABEL_FIELDS)
    embedded_label = bool(
        required_fields
        and ("output:" in sample_text or "intent categories" in sample_text)
    )
    matched_fields = []
    matched_requirements = {
        requirement: sorted(field for field in available_fields if _matches_field(requirement, field))
        for requirement in required_fields
    }
    if (
        {"source", "target"}.issubset(required_fields)
        and "translation" in available_fields
        and len(set(profile["languages"]) & available_fields) >= 2
    ):
        matched_requirements["source"] = ["translation"]
        matched_requirements["target"] = ["translation"]
    schema_evidence = "not-required"
    if not required_fields:
        schema = 15
    elif all(matched_requirements.values()):
        schema = 15
        matched_fields = sorted({
            field for fields_for_requirement in matched_requirements.values()
            for field in fields_for_requirement
        })
        schema_evidence = "direct"
    elif "label" in required_fields and proxy_label_fields and all(
        matched_requirements[requirement] for requirement in required_fields if requirement != "label"
    ):
        schema = 8
        matched_fields = proxy_label_fields
        schema_evidence = "proxy"
    elif "label" in required_fields and embedded_label and all(
        matched_requirements[requirement] for requirement in required_fields if requirement != "label"
    ):
        schema = 5
        matched_fields = ["embedded instruction output"]
        schema_evidence = "embedded"
    else:
        matched_count = sum(bool(fields_for_requirement) for fields_for_requirement in matched_requirements.values())
        schema = round(10 * matched_count / len(required_fields))
        matched_fields = sorted({
            field for fields_for_requirement in matched_requirements.values()
            for field in fields_for_requirement
        })
        schema_evidence = "missing" if available_fields else "unknown"

    license_value = dataset.get("license", "")
    permissive = {"apache-2.0", "mit", "cc-by-4.0", "cc0-1.0", "odc-by", "bsd-3-clause"}
    license_score = 10 if license_value in permissive else 5 if license_value else 0
    if profile["license"]:
        license_check = "pass" if license_value in permissive else "unknown" if not license_value else "fail"
    else:
        license_check = "pass" if license_value in permissive else "unknown" if not license_value else "review"
    documentation = 5 if dataset.get("card_complete") else 2 if dataset.get("description") else 0
    num_examples = int(dataset.get("num_examples") or 0)
    if num_examples >= 10_000:
        popularity = 5
    elif num_examples >= 1_000:
        popularity = 4
    elif num_examples >= 100:
        popularity = 3
    elif num_examples > 0:
        popularity = 1
    else:
        popularity = min(
            3,
            round(math.log10(1 + dataset.get("downloads", 0) + dataset.get("likes", 0) * 20)),
        )
    sample_size_adjustment = 0
    sample_size_check = "pass"
    if "classification" in profile["task_type"] and num_examples:
        if num_examples < 100:
            sample_size_adjustment = -12
            sample_size_check = "review"
        elif num_examples < 500:
            sample_size_adjustment = -4
            sample_size_check = "review"
    domain_penalty = -18 if requested and not matched_keywords else 0
    accessibility = 5 if dataset.get("accessible") and not dataset.get("gated") else 0
    total = max(
        0,
        relevance + modality + language + schema + license_score
        + documentation + popularity + accessibility + sample_size_adjustment + domain_penalty,
    )

    checks = {
        "modality": "pass" if modality == 15 else "unknown" if not modality_values else "fail",
        "domain": domain_check,
        "language": language_check,
        "required_fields": "pass" if schema_evidence in {"not-required", "direct"}
        else "review" if schema_evidence in {"proxy", "embedded"}
        else "unknown" if schema_evidence == "unknown" else "fail",
        "license": license_check,
        "sample_size": sample_size_check if num_examples else "unknown",
        "accessible": "pass" if accessibility == 5 else "fail",
    }
    rejection_reasons = []
    if checks["accessible"] == "fail":
        rejection_reasons.append("Dataset could not be inspected or is gated/private.")
    if checks["modality"] == "fail":
        rejection_reasons.append(
            f"Modality {modality_values} does not match requested {profile['modalities']}."
        )
    if checks["domain"] == "fail":
        rejection_reasons.append(
            "No inspected card, schema, or sample evidence matched the requested subject terms "
            f"({', '.join(sorted(requested))})."
        )
    if checks["required_fields"] == "fail":
        rejection_reasons.append(
            f"Required fields {required_fields} were not found in the inspected schema."
        )
    if checks["language"] == "fail":
        rejection_reasons.append(
            f"Languages {language_values} do not match requested {profile['languages']}."
        )
    if checks["license"] == "fail":
        rejection_reasons.append(
            f"License {license_value} does not meet the requested permissive/commercial constraint."
        )
    recommendation_checks = ["modality", "domain", "required_fields", "accessible"]
    if profile["languages"]:
        recommendation_checks.append("language")
    if profile["license"]:
        recommendation_checks.append("license")
    status = "rejected" if rejection_reasons else "recommended" if (
        total >= 70
        and schema_evidence in {"not-required", "direct"}
        and sample_size_check == "pass"
        and all(checks[key] == "pass" for key in recommendation_checks)
    ) else "conditional"
    evidence = [
        f"Matched project terms: {', '.join(matched_keywords) or 'none verified'}",
        f"Modalities: {', '.join(modality_values) or 'not declared'}",
        f"Languages: {', '.join(language_evidence) or 'not declared'}"
        + (" (inferred from sample script)" if inferred_languages else ""),
        f"Features: {', '.join(sorted(available_fields)[:10]) or 'viewer schema unavailable'}",
        f"Target evidence: {schema_evidence}"
        + (f" ({', '.join(matched_fields)})" if matched_fields else ""),
        f"Examples: {num_examples or 'not reported'}",
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
                ("domain", "The inspected metadata does not verify the requested subject domain."),
                ("required_fields", "Required schema fields need manual confirmation."),
                ("sample_size", "The inspected dataset is too small for reliable classifier training."),
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
            "sample_size_adjustment": sample_size_adjustment,
            "domain_penalty": domain_penalty,
            "accessibility": accessibility,
        },
        "checks": checks,
        "schema_evidence": schema_evidence,
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


def _rank_key(profile: dict[str, Any], dataset: dict[str, Any]) -> tuple[int, int, int]:
    checks = dataset["checks"]
    evidence_fit = (
        (2 if checks["required_fields"] == "pass" else 0)
        + (1 if checks["domain"] == "pass" else 0)
        + (2 if profile["languages"] and checks["language"] == "pass" else 0)
        + (2 if profile["license"] and checks["license"] == "pass" else 0)
    )
    status_rank = {"recommended": 2, "conditional": 1, "rejected": 0}[dataset["status"]]
    return status_rank, evidence_fit, dataset["score"]


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
    search_batches: list[list[str]] = []
    for query in queries:
        found = search_datasets(query, limit=35)
        search_batches.append([dataset["id"] for dataset in found])
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

    inspection_limit = max(max_datasets * 4, 28)
    global_ranked = sorted(
        collected.values(),
        key=lambda dataset: _pre_score(profile, dataset),
        reverse=True,
    )
    diversified_ids: list[str] = []
    diversified_seen: set[str] = set()
    for position in range(8):
        for batch in search_batches:
            if position >= len(batch):
                continue
            dataset_id = batch[position]
            if dataset_id not in diversified_seen:
                diversified_seen.add(dataset_id)
                diversified_ids.append(dataset_id)
    for dataset in global_ranked:
        if dataset["id"] not in diversified_seen:
            diversified_ids.append(dataset["id"])
    pre_ranked = [
        collected[dataset_id]
        for dataset_id in diversified_ids[:inspection_limit]
        if dataset_id in collected
    ]
    yield {
        "type": "search",
        "query": "deep candidate pool",
        "found": len(pre_ranked),
        "unique": len(collected),
        "message": f"Prepared {len(pre_ranked)} diverse candidates for evidence inspection.",
    }
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
        key=lambda dataset: _rank_key(profile, dataset),
        reverse=True,
    )
    ranked = ranked[:max_datasets]
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
        "accessible", "inspection_error", "card_complete", "num_examples", "score", "relevance",
        "quality", "status", "score_breakdown", "checks", "evidence",
        "schema_evidence", "rejection_reasons", "strength", "weakness", "recommendation",
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
