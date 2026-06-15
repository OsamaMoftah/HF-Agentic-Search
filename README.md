---
title: HF Agentic Search
emoji: 🧭
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.18.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Evidence-led agentic dataset discovery for Hugging Face
tags:
  - track:backyard
  - sponsor:openai
  - achievement:offbrand
---

# HF Agentic Search

Finding a dataset is easy. Deciding whether it actually fits a project is not.

HF Agentic Search turns a project brief into a research plan, searches the Hugging Face Hub
from several angles, inspects candidate metadata and Dataset Viewer evidence, tests explicit
constraints, and returns a ranked shortlist with visible reasons. It is designed to be honest
about what it could verify and what still requires human review.

- **Submission Space:** https://huggingface.co/spaces/build-small-hackathon/HF-Agentic-Search
- **Working staging Space:** https://huggingface.co/spaces/sammoftah/HF-Agentic-Search
- **Source:** https://github.com/OsamaMoftah/HF-Agentic-Search
- **Demo video:** `TODO: replace with the public demo-video URL before final validation`
- **Social post:** `TODO: replace with the public social-post URL before final validation`

## Why it is agentic

The app performs a multi-step research loop:

1. Parse the brief into language, modality, task, schema, license, size, and intended-use constraints.
2. Plan multiple targeted Hub searches.
3. Deduplicate and pre-rank candidates by request relevance.
4. Inspect dataset cards, tags, configurations, splits, schema fields, and sample rows.
5. Run explicit modality, language, required-field, license, and accessibility checks.
6. Rank candidates from evidence and connect potentially complementary datasets.
7. Stream the trace and explain verified strengths, limitations, and rejection reasons.

`Qwen/Qwen2.5-3B-Instruct` helps interpret the brief when Hugging Face Inference Providers are
available. The model never supplies the numeric score. If inference is unavailable, the complete
workflow continues with deterministic parsing and clearly labels the fallback.

## Evidence-based scoring

Every 100-point score is assembled from inspectable components:

| Signal | Points |
| --- | ---: |
| Project-term match | 35 |
| Modality | 15 |
| Language | 10 |
| Required schema | 15 |
| License | 10 |
| Dataset-card completeness | 5 |
| Adoption signal | 5 |
| Accessibility | 5 |

Missing evidence is shown as `unknown` rather than silently converted into an average score.
Hard modality, language, accessibility, or required-schema mismatches produce explicit rejection
reasons.

## Architecture

- **Agent:** Python, `huggingface_hub`, Hub API, and Dataset Viewer API
- **Model:** Qwen2.5-3B-Instruct, under the Tiny Titan 4B limit
- **Server:** Gradio `gr.Server` with FastAPI-compatible routes
- **Streaming:** newline-delimited JSON events from `/weave/stream`
- **Interface:** React, Vite, and an interactive PixiJS evidence map

The compatibility endpoint `POST /weave` returns one final JSON document. The primary interface
uses `POST /weave/stream` and receives `started`, `plan`, `search`, `inspect`, `candidate`,
`ranking`, `complete`, and `error` events.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd frontend
npm ci
npm run build
cd ..

HF_TOKEN=your_inference_enabled_token python app.py
```

Open http://localhost:7860. `HF_TOKEN` is optional for public Hub search, but required to use the
hosted planning model.

## Test

```bash
python -m unittest discover -s tests -v
cd frontend && npm run build
```

## Limitations

- Dataset Viewer does not expose schema or samples for every dataset.
- Search quality depends on dataset-card metadata supplied by authors.
- A high score is a research recommendation, not permission to use a dataset; users must still
  verify license terms, consent, privacy, bias, and domain suitability.
- Hosted model availability can vary. Deterministic fallback keeps search functional but may
  interpret nuanced briefs less precisely.
- The in-memory session cache is intentionally lightweight and resets when the Space restarts.

## Build Small submission

Submitted to the **Backyard AI** track and targeting **Best Agent**, **Tiny Titan**, and
**Off Brand**. The custom interface is served through Gradio rather than default Gradio
components.

This project was designed, implemented, tested, and prepared for deployment with OpenAI Codex.
Codex-attributed commits are available in the linked public GitHub repository.
