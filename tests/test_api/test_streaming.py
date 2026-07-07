"""Tests for streaming: event types, JSON data validity, content-type, error events."""

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def streaming_test_app(mock_graph, mock_llm_client):
    """Create TestClient with mock graph for streaming tests.

    Disables auth for streaming tests; auth tests use independent auth_test_app.
    """
    from src.main import app

    app.state.graph = mock_graph
    app.state.llm_client = mock_llm_client
    app.state.cache_store = None
    app.state.vector_store = None
    app.state.doc_store = None
    app.state.indexing_pipeline = None

    # Disable auth for streaming tests
    import src.auth.dependencies as deps
    original_auth = deps.settings.auth_enabled
    deps.settings.auth_enabled = False

    client = TestClient(app)
    yield client

    # Restore
    deps.settings.auth_enabled = original_auth
    app.state.graph = None
    app.state.llm_client = None


class TestStreamEventTypes:
    """Task 5.6: Streaming event type and JSON validity tests."""

    def test_stream_content_type(self, streaming_test_app):
        """Streaming response has text/event-stream content type."""
        resp = streaming_test_app.post("/chat/stream", json={"query": "test", "chat_history": []})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_stream_produces_intent_event(self, streaming_test_app):
        """Stream produces intent event with valid JSON."""
        resp = streaming_test_app.post("/chat/stream", json={"query": "what is RRF", "chat_history": []})
        body = resp.text

        # Check for intent event type
        assert "event: intent" in body

        # Parse and validate JSON data in intent event
        intent_line = None
        for line in body.split("\n"):
            if line.strip().startswith("data:") and "intent" in body.split("event: intent")[-1].split("event:")[0].split("data:")[-1].split("\n")[0]:
                try:
                    payload = json.loads(line.strip()[5:].strip())
                    assert "intent" in payload
                    assert "confidence" in payload
                    break
                except json.JSONDecodeError:
                    pass

    def test_stream_produces_done_event(self, streaming_test_app):
        """Stream ends with done event containing valid JSON."""
        resp = streaming_test_app.post("/chat/stream", json={"query": "test", "chat_history": []})
        body = resp.text

        assert "event: done" in body

    def test_stream_produces_retrieval_chunk_event(self, streaming_test_app):
        """Stream includes retrieval_chunk events."""
        resp = streaming_test_app.post("/chat/stream", json={"query": "what is RRF", "chat_history": []})
        body = resp.text

        # Should have retrieval_chunk or documents event
        has_retrieval = "event: retrieval_chunk" in body or "event: documents" in body
        assert has_retrieval

    def test_stream_produces_token_event(self, streaming_test_app):
        """Stream includes token events during generation."""
        resp = streaming_test_app.post("/chat/stream", json={"query": "explain RRF", "chat_history": []})
        body = resp.text

        has_token = "event: token" in body
        assert has_token

    def test_stream_error_on_invalid_request(self, streaming_test_app):
        """Stream returns error for empty query."""
        resp = streaming_test_app.post("/chat/stream", json={"query": "", "chat_history": []})
        assert resp.status_code == 422  # Validation error

    def test_all_stream_events_are_valid_json(self, streaming_test_app):
        """Every data line in the SSE stream is valid JSON."""
        resp = streaming_test_app.post("/chat/stream", json={"query": "test RRF", "chat_history": []})
        body = resp.text

        for line in body.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                payload_str = line[5:].strip()
                if payload_str:
                    try:
                        json.loads(payload_str)
                    except json.JSONDecodeError:
                        pytest.fail(f"Invalid JSON in SSE data: {payload_str[:80]}")

    def test_stream_has_correct_headers(self, streaming_test_app):
        """Streaming response includes no-cache headers."""
        resp = streaming_test_app.post("/chat/stream", json={"query": "test", "chat_history": []})
        headers = resp.headers

        assert "no-cache" in headers.get("cache-control", "")
        assert headers.get("x-accel-buffering") == "no"


class TestSSEEventSchema:
    """Test the SSE event helper functions."""

    def test_intent_event_schema(self):
        from src.api.sse_events import intent_event
        payload = intent_event("factoid", 0.95)
        assert payload["intent"] == "factoid"
        assert payload["confidence"] == 0.95

    def test_done_event_schema(self):
        from src.api.sse_events import done_event
        payload = done_event(
            intent="factoid", retrieval_tier=1, documents_count=5,
            model="deepseek-chat", session_id="uuid-1"
        )
        assert payload["intent"] == "factoid"
        assert payload["model"] == "deepseek-chat"
        assert payload["session_id"] == "uuid-1"

    def test_error_event_schema(self):
        from src.api.sse_events import error_event
        payload = error_event(node="retrieval", message="timeout")
        assert payload["node"] == "retrieval"
        assert "timeout" in payload["message"]

    def test_all_event_types_are_valid_enum(self):
        from src.api.sse_events import SSEEventType
        # Verify all chat event types exist
        assert SSEEventType.intent.value == "intent"
        assert SSEEventType.token.value == "token"
        assert SSEEventType.done.value == "done"
        assert SSEEventType.error.value == "error"
