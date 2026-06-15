"""Hugging Face dataset search and evidence collection."""
from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from huggingface_hub import HfApi

_api = None
_VIEWER_BASE = "https://datasets-server.huggingface.co"


def _get_api() -> HfApi:
    global _api
    if _api is None:
        token = os.getenv("HF_TOKEN")
        _api = HfApi(token=token if token else None)
    return _api


def _tag_values(tags: list[str], prefix: str) -> list[str]:
    marker = f"{prefix}:"
    return [tag[len(marker):] for tag in tags if tag.startswith(marker)]


def _http_json(path: str, params: dict[str, str], timeout: float = 8.0) -> dict[str, Any]:
    url = f"{_VIEWER_BASE}{path}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "hf-agentic-search/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _normalize_dataset(ds: Any) -> dict[str, Any]:
    tags = list(getattr(ds, "tags", []) or [])
    dataset_id = ds.id
    return {
        "id": dataset_id,
        "author": (getattr(ds, "author", "") or "")
        or (dataset_id.split("/")[0] if "/" in dataset_id else ""),
        "tags": tags,
        "downloads": getattr(ds, "downloads", 0) or 0,
        "likes": getattr(ds, "likes", 0) or 0,
        "task_categories": list(getattr(ds, "task_categories", []) or [])
        or _tag_values(tags, "task_categories"),
        "languages": [str(value) for value in (getattr(ds, "languages", []) or [])]
        or _tag_values(tags, "language"),
        "license": str(getattr(ds, "license", "") or "")
        or (_tag_values(tags, "license") or [""])[0],
        "size_category": str(getattr(ds, "size_category", "") or "")
        or (_tag_values(tags, "size_categories") or [""])[0],
        "formats": _tag_values(tags, "format"),
        "modalities": _tag_values(tags, "modality"),
        "description": str(getattr(ds, "description", "") or "")[:1200],
        "created_at": str(getattr(ds, "created_at", "") or ""),
        "updated_at": str(getattr(ds, "updated_at", "") or ""),
        "private": bool(getattr(ds, "private", False)),
        "gated": getattr(ds, "gated", False) or False,
        "hub_url": f"https://huggingface.co/datasets/{dataset_id}",
    }


def search_datasets(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search the Hub and normalize metadata encoded in dataset tags."""
    try:
        results = list(_get_api().list_datasets(search=query, limit=limit))
    except Exception:
        return []
    datasets = []
    for dataset in results:
        try:
            datasets.append(_normalize_dataset(dataset))
        except Exception:
            continue
    return datasets


def inspect_dataset(dataset_id: str, base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect card, config, split, feature, and sample evidence for one dataset."""
    evidence = dict(base or {"id": dataset_id})
    evidence.update(
        {
            "accessible": False,
            "inspection_error": "",
            "configs": [],
            "splits": [],
            "features": [],
            "sample_rows": [],
            "files": [],
            "card_complete": False,
        }
    )
    try:
        info = _get_api().dataset_info(dataset_id, files_metadata=True)
        evidence.update(_normalize_dataset(info))
        evidence["accessible"] = not bool(getattr(info, "private", False))
        evidence["files"] = [
            sibling.rfilename
            for sibling in (getattr(info, "siblings", []) or [])
            if getattr(sibling, "rfilename", None)
        ][:40]
        card_data = getattr(info, "card_data", None)
        if card_data:
            card_dict = card_data.to_dict() if hasattr(card_data, "to_dict") else dict(card_data)
            evidence["card_data"] = card_dict
            evidence["card_complete"] = bool(
                evidence.get("description")
                and (card_dict.get("license") or evidence.get("license"))
                and (card_dict.get("language") or evidence.get("languages"))
            )
    except Exception as exc:
        evidence["inspection_error"] = f"Hub metadata unavailable: {type(exc).__name__}"
        return evidence

    try:
        viewer = _http_json("/info", {"dataset": dataset_id})
        dataset_info = viewer.get("dataset_info", {})
        for config_name, config_data in dataset_info.items():
            evidence["configs"].append(config_name)
            for split in config_data.get("splits", []) or []:
                split_name = split.get("name")
                if split_name and split_name not in evidence["splits"]:
                    evidence["splits"].append(split_name)
            features = config_data.get("features") or {}
            if isinstance(features, dict):
                feature_names = list(features)
            elif isinstance(features, list):
                feature_names = [
                    item.get("name") for item in features if isinstance(item, dict) and item.get("name")
                ]
            else:
                feature_names = []
            for feature in feature_names:
                if feature not in evidence["features"]:
                    evidence["features"].append(feature)
    except Exception:
        pass

    if evidence["configs"] and evidence["splits"]:
        try:
            rows = _http_json(
                "/first-rows",
                {
                    "dataset": dataset_id,
                    "config": evidence["configs"][0],
                    "split": evidence["splits"][0],
                },
                timeout=10.0,
            )
            evidence["sample_rows"] = [
                item.get("row", {}) for item in (rows.get("rows", []) or [])[:3]
            ]
            if not evidence["features"] and evidence["sample_rows"]:
                evidence["features"] = list(evidence["sample_rows"][0])
        except Exception:
            pass

    evidence["configs"] = evidence["configs"][:10]
    evidence["splits"] = evidence["splits"][:12]
    evidence["features"] = evidence["features"][:30]
    return evidence


def get_dataset_info(dataset_id: str) -> dict[str, Any] | None:
    """Backward-compatible detailed dataset lookup."""
    result = inspect_dataset(dataset_id)
    return result if result.get("accessible") else None
