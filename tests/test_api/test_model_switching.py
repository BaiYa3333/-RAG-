"""Model switching API tests."""

from src.graph.state import initial_state


class TestModelSwitching:
    def test_models_endpoint(self, test_app):
        resp = test_app.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["default"] == "deepseek-chat"
        names = {m["name"] for m in data["models"]}
        assert "deepseek-chat" in names
        assert "qwen3.6-plus" in names
        first = data["models"][0]
        assert {"name", "provider", "label", "default"}.issubset(first)

    def test_chat_accepts_valid_model(self, test_app):
        resp = test_app.post("/chat", json={"query": "什么是 RRF", "model": "qwen3.6-plus"})
        assert resp.status_code == 200
        assert "model" in resp.json()

    def test_chat_rejects_invalid_model(self, test_app):
        resp = test_app.post("/chat", json={"query": "什么是 RRF", "model": "missing-model"})
        assert resp.status_code == 422
        assert "Unknown model" in resp.json()["detail"]

    def test_initial_state_model_propagation(self):
        state = initial_state("q", model_name="qwen3.6-plus")
        assert state["model_name"] == "qwen3.6-plus"

    def test_ui_sends_selected_model(self):
        with open("src/ui/index.html", encoding="utf-8") as f:
            html = f.read()
        assert "model-select" in html
        assert "loadModels" in html
        assert "model: selectedModel || undefined" in html
        assert "badge-model" in html
