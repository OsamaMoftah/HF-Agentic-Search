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
models:
  - HuggingFaceTB/SmolLM2-360M-Instruct
---

# HF Agentic Search

Finding a dataset is easy. Deciding whether it actually fits an ML project is not.

HF Agentic Search turns a plain-language project brief into a small research loop: it plans
targeted Hugging Face Hub searches, inspects candidate cards and Dataset Viewer evidence,
checks schemas and samples, then returns a ranked shortlist with transparent reasons. The app
is built for engineers who care less about popularity and more about whether the data can
actually support the model or evaluation they are trying to ship.

- **Submission Space:** https://huggingface.co/spaces/build-small-hackathon/HF-Agentic-Search
- **Working staging Space:** https://huggingface.co/spaces/sammoftah/HF-Agentic-Search
- **Source:** https://github.com/OsamaMoftah/HF-Agentic-Search
- **Demo video:** `PENDING: add a public demo-video URL before final validation`
- **Social post:** `PENDING: add the public social-media post URL before final validation`
- **Team usernames:** `sammoftah`

## Why it is agentic

The app performs a multi-step research loop:

1. Parse the brief into language, modality, task, schema, license, size, and intended-use constraints.
2. Plan multiple targeted Hub searches.
3. Deduplicate and pre-rank candidates by request relevance.
4. Inspect dataset cards, tags, configurations, splits, schema fields, and sample rows.
5. Run explicit modality, language, required-field, license, and accessibility checks.
6. Rank candidates from evidence and connect potentially complementary datasets.
7. Stream the trace and explain verified strengths, limitations, and rejection reasons.

`HuggingFaceTB/SmolLM2-360M-Instruct` runs locally inside the Space CPU runtime and helps interpret
the brief. At only 360M parameters, it is comfortably below the Tiny Titan 4B limit. The model
never supplies the numeric score. If local model loading fails, the complete workflow continues
with deterministic parsing and clearly labels the fallback.

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
- **Model:** SmolLM2-360M-Instruct, loaded locally on CPU
- **Server:** Gradio `gr.Server` with FastAPI-compatible routes
- **Streaming:** newline-delimited JSON events from `/weave/stream`
- **Interface:** React, Vite, and a custom decision board for inspected candidates

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

python app.py
```

Open http://localhost:7860. `HF_TOKEN` is optional and is used only for authenticated Hub access;
the planning model runs locally.

## Test

```bash
PYTHONPATH="$PWD" python3 -m pytest -q tests/test_agent.py
cd frontend && npm run build
```

## Limitations

- Dataset Viewer does not expose schema or samples for every dataset.
- Search quality depends on dataset-card metadata supplied by authors.
- A high score is a research recommendation, not permission to use a dataset; users must still
  verify license terms, consent, privacy, bias, and domain suitability.
- The first request may take longer while the 360M planning model is downloaded and loaded.
- Deterministic fallback keeps search functional if local model loading fails, but may interpret
  nuanced briefs less precisely.
- The in-memory session cache is intentionally lightweight and resets when the Space restarts.

## Build Small submission

Submitted to the **Backyard AI** track and positioned for **Best Agent**, **Tiny Titan**, and
**Off Brand**. The app runs as a Gradio Space inside the `build-small-hackathon` organization,
with a custom React interface served through `gr.Server`.

Submission readiness:

- YAML tags are present: `track:backyard`, `sponsor:openai`, `achievement:offbrand`.
- The model is `HuggingFaceTB/SmolLM2-360M-Instruct` at 360M parameters, well below the 32B limit.
- Team username is listed above.
- Demo video and social-post URLs are the remaining fields to replace before the final validator run.

Built, tested, deployed, and prepared for submission with OpenAI Codex. Codex-attributed commits
are available in the linked public GitHub repository.
