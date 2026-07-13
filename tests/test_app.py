import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import app as app_module


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app_module.app)

    def test_rejects_empty_and_oversized_tasks(self):
        empty = self.client.post("/weave", json={"task": ""})
        self.assertEqual(empty.status_code, 400)
        oversized = self.client.post("/weave", json={"task": "x" * 2001})
        self.assertEqual(oversized.status_code, 422)

    def test_generates_and_reuses_session_id(self):
        state = self.client.get("/state")
        self.assertEqual(state.status_code, 200)
        self.assertTrue(state.headers["X-Session-Id"].startswith("dw-"))

    def test_weave_runs_off_the_event_loop(self):
        with patch("app.weave", return_value={
            "task": "test", "datasets": [], "nodes": [], "threads": [],
            "top_pick": None, "fallback_used": True,
        }) as mock_weave:
            response = self.client.post("/weave", json={"task": "test"})
        self.assertEqual(response.status_code, 200)
        mock_weave.assert_called_once_with("test")

    def test_session_cache_is_bounded(self):
        app_module._MAX_SESSIONS = 3
        try:
            for index in range(5):
                app_module._store_session(f"sid-{index}", {"task": str(index)})
            self.assertLessEqual(len(app_module._sessions), 3)
            self.assertNotIn("sid-0", app_module._sessions)
            self.assertIn("sid-4", app_module._sessions)
        finally:
            app_module._sessions.clear()
            app_module._MAX_SESSIONS = 200

    def test_stream_is_ndjson_and_saves_complete_state(self):
        result = {
            "task": "test",
            "datasets": [],
            "nodes": [],
            "threads": [],
            "top_pick": None,
            "fallback_used": True,
        }
        events = iter([
            {"type": "started", "message": "started"},
            {"type": "complete", "result": result, "message": "done"},
        ])
        with patch("app.weave_events", return_value=events):
            response = self.client.post(
                "/weave/stream",
                json={"task": "test"},
                headers={"X-Session-Id": "api-test"},
            )
        self.assertEqual(response.status_code, 200)
        parsed = [json.loads(line) for line in response.text.strip().splitlines()]
        self.assertEqual([event["type"] for event in parsed], ["started", "complete"])
        saved = self.client.get("/state", headers={"X-Session-Id": "api-test"}).json()
        self.assertEqual(saved, result)


if __name__ == "__main__":
    unittest.main()
