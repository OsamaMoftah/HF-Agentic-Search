import unittest
from unittest.mock import patch

from backend.agent import generate_queries, parse_task, score_dataset, weave, weave_events


def dataset(**overrides):
    base = {
        "id": "example/support-intents",
        "author": "example",
        "description": "English customer support intent classification with labeled examples.",
        "downloads": 1200,
        "likes": 20,
        "tags": ["modality:text", "language:en", "license:apache-2.0"],
        "task_categories": ["text-classification"],
        "languages": ["en"],
        "license": "apache-2.0",
        "size_category": "1K<n<10K",
        "formats": ["parquet"],
        "modalities": ["text"],
        "configs": ["default"],
        "splits": ["train"],
        "features": ["text", "label", "intent"],
        "sample_rows": [{"text": "Where is my order?", "label": 2, "intent": "shipping"}],
        "hub_url": "https://huggingface.co/datasets/example/support-intents",
        "accessible": True,
        "gated": False,
        "inspection_error": "",
        "card_complete": True,
        "num_examples": 5000,
    }
    return {**base, **overrides}


class AgentTests(unittest.TestCase):
    def test_parse_task_extracts_explicit_constraints_without_llm(self):
        profile, llm_used = parse_task(
            "I need an English customer support intent dataset with labels for a classifier",
            use_llm=False,
        )
        self.assertFalse(llm_used)
        self.assertIn("en", profile["languages"])
        self.assertIn("text", profile["modalities"])
        self.assertIn("label", profile["required_fields"])
        self.assertIn("classification", profile["task_type"])

    def test_relevant_labeled_text_beats_audio_and_unlabeled_data(self):
        profile, _ = parse_task(
            "English customer support intent data with labels for a classifier",
            use_llm=False,
        )
        relevant = score_dataset(profile, dataset())
        audio = score_dataset(
            profile,
            dataset(
                id="example/support-audio",
                description="English customer support audio recordings.",
                modalities=["audio"],
                features=["audio"],
                sample_rows=[{"audio": "sample.wav"}],
            ),
        )
        unlabeled = score_dataset(
            profile,
            dataset(
                id="example/support-text",
                description="English customer support messages.",
                features=["text"],
                sample_rows=[{"text": "Where is my order?"}],
            ),
        )
        self.assertGreater(relevant["score"], audio["score"])
        self.assertGreater(relevant["score"], unlabeled["score"])
        self.assertEqual(relevant["status"], "recommended")
        self.assertEqual(audio["status"], "rejected")

    def test_description_does_not_fake_a_required_label_column(self):
        profile, _ = parse_task(
            "English customer support intent data with labels for a classifier",
            use_llm=False,
        )
        misleading = score_dataset(
            profile,
            dataset(
                description="A labeled customer support dataset with many labels.",
                features=["text", "response"],
                sample_rows=[{"text": "hello", "response": "hi"}],
            ),
        )
        self.assertEqual(misleading["checks"]["required_fields"], "fail")
        self.assertEqual(misleading["status"], "rejected")

    def test_direct_intent_schema_beats_proxy_ticket_routing(self):
        profile, _ = parse_task(
            "English customer support intent data with labels for a classifier",
            use_llm=False,
        )
        direct = score_dataset(
            profile,
            dataset(
                id="example/customer-support-intent-classification",
                features=["query", "intent"],
                sample_rows=[{"query": "refund please", "intent": "get_refund"}],
                num_examples=800,
            ),
        )
        proxy = score_dataset(
            profile,
            dataset(
                id="example/customer-support-tickets",
                features=["subject", "body", "type", "queue"],
                sample_rows=[{"subject": "help", "body": "refund", "type": "Request", "queue": "Billing"}],
                num_examples=60_000,
            ),
        )
        self.assertEqual(direct["schema_evidence"], "direct")
        self.assertEqual(proxy["schema_evidence"], "proxy")
        self.assertGreater(direct["score"], proxy["score"])

    def test_tiny_classifier_dataset_is_not_the_top_recommendation(self):
        profile, _ = parse_task(
            "English customer support intent data with labels for a classifier",
            use_llm=False,
        )
        tiny = score_dataset(
            profile,
            dataset(
                id="example/customer-support-intent-classification",
                features=["query", "intent"],
                sample_rows=[{"query": "refund please", "intent": "get_refund"}],
                num_examples=41,
            ),
        )
        training_sized = score_dataset(
            profile,
            dataset(
                id="example/customer-support-intent-data",
                features=["instruction", "intent", "response"],
                sample_rows=[{"instruction": "refund please", "intent": "get_refund"}],
                num_examples=26_872,
            ),
        )
        self.assertEqual(tiny["checks"]["sample_size"], "review")
        self.assertEqual(tiny["status"], "conditional")
        self.assertGreater(training_sized["score"], tiny["score"])

    def test_queries_are_short_and_hub_friendly(self):
        profile, _ = parse_task(
            "English customer support intent data with labels for a compact classifier",
            use_llm=False,
        )
        queries = generate_queries("unused", profile)
        self.assertIn("customer support intent", queries)
        self.assertIn("customer support", queries)
        self.assertTrue(all(len(query.split()) <= 3 for query in queries))

    @patch(
        "backend.agent._llm",
        return_value='{"task_type":"compact","license":"gpl","domain_keywords":["support"]}',
    )
    def test_invalid_small_model_fields_do_not_corrupt_explicit_requirements(self, _mock_llm):
        profile, used = parse_task(
            "English customer support intent data with labels for a classifier",
            use_llm=True,
        )
        self.assertTrue(used)
        self.assertEqual(profile["task_type"], "classification")
        self.assertEqual(profile["license"], "")
        self.assertIn("label", profile["required_fields"])

    @patch("backend.agent._llm", return_value=None)
    def test_inference_failure_is_reported_as_fallback(self, _mock_llm):
        with patch("backend.agent.search_datasets", return_value=[]):
            result = weave("English intent labels")
        self.assertTrue(result["fallback_used"])
        self.assertIsNone(result["model_used"])
        self.assertEqual(result["datasets"], [])

    def test_declared_model_is_tiny_titan_eligible(self):
        from backend.agent import MODEL

        self.assertEqual(MODEL, "HuggingFaceTB/SmolLM2-360M-Instruct")

    @patch("backend.agent._llm", return_value=None)
    def test_stream_event_order_and_complete_result(self, _mock_llm):
        candidate = dataset()
        with (
            patch("backend.agent.search_datasets", return_value=[candidate]),
            patch("backend.agent.inspect_dataset", return_value=candidate),
        ):
            events = list(weave_events("English customer support intent labels"))
        event_types = [event["type"] for event in events]
        self.assertEqual(event_types[0], "started")
        self.assertEqual(event_types[1], "plan")
        self.assertIn("search", event_types)
        self.assertIn("inspect", event_types)
        self.assertEqual(event_types[-2:], ["ranking", "complete"])
        self.assertEqual(events[-1]["result"]["top_pick"], candidate["id"])


if __name__ == "__main__":
    unittest.main()
