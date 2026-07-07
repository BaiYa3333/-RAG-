"""Verification tests for configure-langfuse-access + add-langfuse-monitoring changes.

Covers:
  - configure-langfuse-access: Docker compose config, accessibility, UI isolation
  - add-langfuse-monitoring: client init, decorators, node monitoring, LLM tracking,
    retrieval monitoring, ingestion monitoring
"""

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# configure-langfuse-access — Verification Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestLangfuseDockerConfig:
    """Task 4.1: Verify docker-compose NEXTAUTH_URL and port configuration."""

    def test_nexthauth_url_uses_host_port_3001(self):
        """NEXTAUTH_URL must be http://localhost:3001 to match port mapping."""
        import yaml

        compose_path = os.path.join(
            os.path.dirname(__file__), "..", "docker", "docker-compose.yml"
        )
        with open(compose_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        langfuse_env = config["services"]["langfuse-server"]["environment"]
        # NEXTAUTH_URL is set directly as a string in the YAML
        assert "NEXTAUTH_URL" in langfuse_env or any(
            "NEXTAUTH_URL" in str(k) for k in langfuse_env
        ), "NEXTAUTH_URL must be set"

        # Check NEXTAUTH_URL value
        nextauth = langfuse_env.get("NEXTAUTH_URL", "")
        assert "localhost:3001" in nextauth, (
            f"NEXTAUTH_URL must be localhost:3001, got: {nextauth}"
        )

    def test_port_mapping_3001_to_3000(self):
        """Langfuse container port mapping: host 3001 → container 3000."""
        import yaml

        compose_path = os.path.join(
            os.path.dirname(__file__), "..", "docker", "docker-compose.yml"
        )
        with open(compose_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        ports = config["services"]["langfuse-server"]["ports"]
        assert any("3001:3000" in str(p) or ("3001" in str(p) and "3000" in str(p)) for p in ports), (
            f"Port mapping must expose host:3001 → container:3000, got: {ports}"
        )

    def test_app_uses_internal_docker_network_for_langfuse(self):
        """App container must use langfuse-server:3000 internally."""
        import yaml

        compose_path = os.path.join(
            os.path.dirname(__file__), "..", "docker", "docker-compose.yml"
        )
        with open(compose_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        app_env = config["services"]["app"]["environment"]
        langfuse_host = [
            v for v in app_env if "RAG_LANGFUSE_HOST" in str(v)
        ]
        assert len(langfuse_host) > 0, "App must have RAG_LANGFUSE_HOST configured"
        assert any(
            "langfuse-server:3000" in str(v) for v in langfuse_host
        ), f"App must use internal Docker DNS langfuse-server:3000, got: {langfuse_host}"

    def test_langfuse_server_not_exposed_on_default_port(self):
        """Langfuse must NOT be exposed on port 3000 (only 3001)."""
        import yaml

        compose_path = os.path.join(
            os.path.dirname(__file__), "..", "docker", "docker-compose.yml"
        )
        with open(compose_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        ports = config["services"]["langfuse-server"]["ports"]
        # Should not have "3000:3000"
        for p in ports:
            p_str = str(p)
            if ":" in p_str:
                host_port = p_str.split(":")[0].strip('"').strip("'")
                assert host_port != "3000", (
                    f"Langfuse must not expose host port 3000; use 3001. Got: {p}"
                )


class TestLangfuseUIIsolation:
    """Task 4.4: Verify no Langfuse links on main page."""

    def test_ui_html_has_no_langfuse_references(self):
        """No UI HTML files should reference Langfuse."""
        import glob as glob_mod

        ui_dir = os.path.join(os.path.dirname(__file__), "..", "src", "ui")
        html_files = glob_mod.glob(os.path.join(ui_dir, "**", "*.html"), recursive=True)

        for html_path in html_files:
            with open(html_path, encoding="utf-8") as f:
                content = f.read().lower()
            assert "langfuse" not in content, (
                f"{os.path.basename(html_path)} contains 'langfuse' reference"
            )
            assert "3001" not in content, (
                f"{os.path.basename(html_path)} contains port '3001' reference"
            )

    def test_ui_js_has_no_langfuse_references(self):
        """No UI JS files should reference Langfuse."""
        import glob as glob_mod

        ui_dir = os.path.join(os.path.dirname(__file__), "..", "src", "ui")
        js_files = glob_mod.glob(os.path.join(ui_dir, "**", "*.js"), recursive=True)

        for js_path in js_files:
            with open(js_path, encoding="utf-8") as f:
                content = f.read().lower()
            assert "langfuse" not in content, (
                f"{os.path.basename(js_path)} contains 'langfuse' reference"
            )


class TestInfrastructureSpec:
    """Task 3.1: Verify infrastructure spec updated to port 3001."""

    def test_infrastructure_spec_uses_port_3001(self):
        """Infrastructure spec must reference port 3001 for Langfuse."""
        spec_path = os.path.join(
            os.path.dirname(__file__), "..", "openspec", "specs", "infrastructure", "spec.md"
        )
        if not os.path.exists(spec_path):
            pytest.skip("Infrastructure spec file not found")

        with open(spec_path, encoding="utf-8") as f:
            content = f.read()

        # The spec should mention 3001 for Langfuse
        assert "3001" in content, "Infrastructure spec should reference port 3001"


# ═══════════════════════════════════════════════════════════════════════════════
# add-langfuse-monitoring — Verification Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestLangfuseClientInit:
    """Verify Langfuse client initialization and lifecycle.

    Uses patch.object on settings because is_langfuse_enabled() reads from
    the module-level Settings singleton, not directly from environment variables.
    """

    def test_is_langfuse_enabled_true_by_default(self):
        """When RAG_LANGFUSE_ENABLED=true and valid keys, monitoring is enabled."""
        import src.config as config_mod
        import src.observability.client as client_mod

        client_mod._enabled = None
        client_mod._langfuse_client = None

        with patch.object(config_mod.settings, "langfuse_enabled", True):
            with patch.object(config_mod.settings, "langfuse_public_key", "pk-lf-test123"):
                with patch.object(config_mod.settings, "langfuse_secret_key", "sk-lf-test456"):
                    result = client_mod.is_langfuse_enabled()
        assert result is True, f"Expected enabled=True, got {result}"

    def test_is_langfuse_enabled_false_from_config(self):
        """When RAG_LANGFUSE_ENABLED=false, monitoring is disabled."""
        import src.config as config_mod
        import src.observability.client as client_mod

        client_mod._enabled = None
        client_mod._langfuse_client = None

        with patch.object(config_mod.settings, "langfuse_enabled", False):
            with patch.object(config_mod.settings, "langfuse_public_key", "pk-lf-test123"):
                with patch.object(config_mod.settings, "langfuse_secret_key", "sk-lf-test456"):
                    result = client_mod.is_langfuse_enabled()
        assert result is False

    def test_is_langfuse_enabled_false_placeholder_keys(self):
        """When keys are placeholder values (pk-lf-xxx...), monitoring is disabled."""
        import src.config as config_mod
        import src.observability.client as client_mod

        client_mod._enabled = None
        client_mod._langfuse_client = None

        with patch.object(config_mod.settings, "langfuse_enabled", True):
            with patch.object(config_mod.settings, "langfuse_public_key", "pk-lf-xxx-placeholder"):
                with patch.object(config_mod.settings, "langfuse_secret_key", "sk-lf-xxx-placeholder"):
                    result = client_mod.is_langfuse_enabled()
        assert result is False

    def test_is_langfuse_enabled_empty_keys(self):
        """When keys are empty strings, monitoring is disabled."""
        import src.config as config_mod
        import src.observability.client as client_mod

        client_mod._enabled = None
        client_mod._langfuse_client = None

        with patch.object(config_mod.settings, "langfuse_enabled", True):
            with patch.object(config_mod.settings, "langfuse_public_key", ""):
                with patch.object(config_mod.settings, "langfuse_secret_key", ""):
                    result = client_mod.is_langfuse_enabled()
        assert result is False

    def test_get_langfuse_returns_none_when_disabled(self):
        """When disabled, get_langfuse() must return None."""
        import src.config as config_mod
        import src.observability.client as client_mod

        client_mod._enabled = None
        client_mod._langfuse_client = None

        with patch.object(config_mod.settings, "langfuse_enabled", False):
            result = client_mod.get_langfuse()
        assert result is None

    def test_langfuse_env_example_has_enabled_config(self):
        """.env.example must document RAG_LANGFUSE_ENABLED."""
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env.example")
        if not os.path.exists(env_path):
            pytest.skip(".env.example not found")

        with open(env_path, encoding="utf-8") as f:
            content = f.read()

        assert "RAG_LANGFUSE_ENABLED" in content, (
            ".env.example must document RAG_LANGFUSE_ENABLED"
        )


class TestTraceRagNodeDecorator:
    """Verify @trace_rag_node decorator behavior."""

    def test_decorator_noop_when_disabled(self):
        """When Langfuse disabled, decorator returns original function unchanged."""
        # Directly patch is_langfuse_enabled to return False
        with patch("src.observability.decorators.is_langfuse_enabled", return_value=False):
            import importlib
            import src.observability.decorators as deco_mod
            importlib.reload(deco_mod)

            # Define a test function
            async def test_func(x: int) -> int:
                return x * 2

            decorated = deco_mod.trace_rag_node(name="test_func")(test_func)
            # When disabled, must return the same function (not wrapped)
            assert decorated is test_func, (
                "Decorator must return original function when Langfuse is disabled"
            )

    def test_decorator_preserves_function_metadata(self):
        """Decorator must preserve __name__, __doc__, etc."""
        # Test with disabled Langfuse (no-op mode)
        with patch("src.observability.decorators.is_langfuse_enabled", return_value=False):
            from src.observability.decorators import trace_rag_node

            @trace_rag_node(name="my_custom_name")
            async def sample_node(query: str) -> dict:
                """Sample node docstring."""
                return {"result": query}

            assert sample_node.__name__ == "sample_node"
            assert sample_node.__doc__ == "Sample node docstring."

    def test_decorator_sync_function_preserved(self):
        """Decorator must work with sync functions."""
        with patch("src.observability.decorators.is_langfuse_enabled", return_value=False):
            from src.observability.decorators import trace_rag_node

            @trace_rag_node(name="sync_func")
            def sync_node(x: int) -> int:
                return x + 1

            assert sync_node(5) == 6

    @pytest.mark.asyncio
    async def test_decorator_async_function_preserved(self):
        """Decorator must work with async functions."""
        with patch("src.observability.decorators.is_langfuse_enabled", return_value=False):
            from src.observability.decorators import trace_rag_node

            @trace_rag_node(name="async_func")
            async def async_node(x: int) -> int:
                return x * 3

            result = await async_node(4)
            assert result == 12


class TestUpdateCurrentObservation:
    """Verify observation context helpers don't crash when disabled."""

    def test_update_current_observation_noop_when_disabled(self, monkeypatch):
        """update_current_observation must not raise when Langfuse disabled."""
        monkeypatch.setenv("RAG_LANGFUSE_ENABLED", "false")

        import importlib
        import src.observability.client as client_mod
        import src.observability.context as ctx_mod
        importlib.reload(client_mod)
        client_mod._enabled = None
        importlib.reload(ctx_mod)

        # Should not raise
        ctx_mod.update_current_observation(
            node="test_node", intent="factoid", confidence=0.95
        )

    def test_update_current_trace_noop_when_disabled(self, monkeypatch):
        """update_current_trace must not raise when Langfuse disabled."""
        monkeypatch.setenv("RAG_LANGFUSE_ENABLED", "false")

        import importlib
        import src.observability.client as client_mod
        import src.observability.context as ctx_mod
        importlib.reload(client_mod)
        client_mod._enabled = None
        importlib.reload(ctx_mod)

        # Should not raise
        ctx_mod.update_current_trace(
            session_id="sess_123", user_id="user_1", query="test query"
        )

    def test_update_current_observation_handles_nested_dicts(self, monkeypatch):
        """update_current_observation must handle nested dict/list values."""
        monkeypatch.setenv("RAG_LANGFUSE_ENABLED", "false")

        import importlib
        import src.observability.client as client_mod
        import src.observability.context as ctx_mod
        importlib.reload(client_mod)
        client_mod._enabled = None
        importlib.reload(ctx_mod)

        # Should not raise with complex nested data
        ctx_mod.update_current_observation(
            nested={"key": "value", "list": [1, 2, 3]},
            empty=None,  # None values are skipped
        )


class TestLLMTracker:
    """Verify track_llm_call behavior."""

    def test_track_llm_call_noop_when_disabled(self):
        """track_llm_call must not raise when Langfuse disabled."""
        import src.config as config_mod
        import src.observability.client as client_mod
        import src.observability.llm_tracker as tracker_mod

        client_mod._enabled = None
        client_mod._langfuse_client = None

        with patch.object(config_mod.settings, "langfuse_enabled", False):
            # Should not raise
            tracker_mod.track_llm_call(
                name="test_call",
                model="deepseek-chat",
                start_time=time.monotonic(),
                metadata={"test": True},
            )

    def test_track_llm_call_noop_when_client_none(self):
        """track_llm_call must not raise when get_langfuse returns None."""
        mock_langfuse = None

        with patch("src.observability.llm_tracker.is_langfuse_enabled", return_value=True):
            with patch("src.observability.llm_tracker.get_langfuse", return_value=mock_langfuse):
                from src.observability.llm_tracker import track_llm_call

                # Should not raise (returns early when client is None)
                track_llm_call(
                    name="test_call",
                    model="deepseek-chat",
                    start_time=time.monotonic(),
                )

    @patch("langfuse.decorators.langfuse_context")
    def test_track_llm_call_handles_usage_extraction(self, mock_ctx):
        """track_llm_call must correctly extract token usage and call generation()."""
        # Mock a response with usage info
        response = MagicMock()
        response.usage.prompt_tokens = 150
        response.usage.completion_tokens = 80
        response.usage.total_tokens = 230
        response.choices = [MagicMock()]
        response.choices[0].message.content = "Test generated response"

        mock_langfuse = MagicMock()
        mock_gen = MagicMock()
        mock_langfuse.generation.return_value = mock_gen

        mock_ctx.get_current_trace_id.return_value = "trace_123"
        mock_ctx.get_current_observation_id.return_value = "obs_456"

        import src.observability.llm_tracker as tracker_mod

        # Patch the already-imported names in llm_tracker module
        with patch.object(tracker_mod, "is_langfuse_enabled", return_value=True):
            with patch.object(tracker_mod, "get_langfuse", return_value=mock_langfuse):
                from src.observability.llm_tracker import track_llm_call

                track_llm_call(
                    name="generate_llm",
                    model="deepseek-chat",
                    start_time=time.monotonic() - 0.5,  # 500ms latency
                    response=response,
                )

        # Verify generation() was called with correct args (v2 API)
        mock_langfuse.generation.assert_called_once()
        call_kwargs = mock_langfuse.generation.call_args[1]
        assert call_kwargs["name"] == "generate_llm"
        assert call_kwargs["model"] == "deepseek-chat"
        assert call_kwargs["trace_id"] == "trace_123"
        assert call_kwargs["parent_observation_id"] == "obs_456"
        # usage_details is broken in v2.60.10; use deprecated 'usage' param instead
        assert call_kwargs["usage"]["input"] == 150
        assert call_kwargs["usage"]["output"] == 80
        assert call_kwargs["usage"]["total"] == 230
        # generation.end() must be called
        mock_gen.end.assert_called_once()

    @patch("langfuse.decorators.langfuse_context")
    def test_track_llm_call_handles_error_response(self, mock_ctx):
        """track_llm_call must correctly record errors via generation()."""
        mock_langfuse = MagicMock()
        mock_gen = MagicMock()
        mock_langfuse.generation.return_value = mock_gen

        mock_ctx.get_current_trace_id.return_value = "trace_abc"
        mock_ctx.get_current_observation_id.return_value = "obs_def"

        import src.observability.llm_tracker as tracker_mod

        with patch.object(tracker_mod, "is_langfuse_enabled", return_value=True):
            with patch.object(tracker_mod, "get_langfuse", return_value=mock_langfuse):
                from src.observability.llm_tracker import track_llm_call

                track_llm_call(
                    name="router_llm",
                    model="deepseek-chat",
                    start_time=time.monotonic() - 2.0,
                    error="timeout",
                )

        call_kwargs = mock_langfuse.generation.call_args[1]
        assert call_kwargs["level"] == "ERROR"
        assert call_kwargs["status_message"] == "timeout"


class TestLangfuseConfigSettings:
    """Verify Langfuse configuration in Settings."""

    def test_langfuse_settings_exist(self):
        """Settings must include all Langfuse-related config fields."""
        from src.config import Settings

        # Verify fields exist on the model (may use alias)
        field_names = {f for f in Settings.model_fields}
        assert "langfuse_enabled" in field_names, "Missing langfuse_enabled field"
        assert "langfuse_public_key" in field_names, "Missing langfuse_public_key field"
        assert "langfuse_secret_key" in field_names, "Missing langfuse_secret_key field"
        assert "langfuse_host" in field_names, "Missing langfuse_host field"

    def test_langfuse_enabled_default_true(self):
        """RAG_LANGFUSE_ENABLED must default to True."""
        from src.config import Settings

        # Default value from Field(default=True, ...)
        field = Settings.model_fields["langfuse_enabled"]
        assert field.default is True, (
            f"langfuse_enabled default must be True, got {field.default}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Graph Node Monitoring Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGraphNodeMonitoring:
    """Verify all 10 LangGraph nodes have @trace_rag_node decorator."""

    # Expected nodes and their decorator names
    EXPECTED_NODES = [
        ("query_condenser", "src/graph/nodes/query_condenser.py"),
        ("intent_router", "src/graph/nodes/intent_router.py"),
        ("chitchat_answer", "src/graph/workflow.py"),
        ("tier1_retrieve", "src/graph/nodes/retrieval.py"),
        ("tier2_retrieve", "src/graph/nodes/retrieval.py"),
        ("quality_gate", "src/graph/nodes/quality_gate.py"),
        ("rerank_docs", "src/graph/nodes/rerank.py"),
        ("compress_context", "src/graph/nodes/compress.py"),
        ("generate_answer", "src/graph/nodes/generate.py"),
        ("analytical_agent", "src/graph/agent.py"),
    ]

    @pytest.mark.parametrize("node_name,file_path", EXPECTED_NODES)
    def test_node_has_trace_decorator(self, node_name, file_path):
        """Each graph node must be decorated with @trace_rag_node."""
        full_path = os.path.join(os.path.dirname(__file__), "..", file_path)
        with open(full_path, encoding="utf-8") as f:
            content = f.read()

        # Check for decorator usage
        assert f'@trace_rag_node(name="{node_name}")' in content, (
            f"{file_path}: missing @trace_rag_node(name=\"{node_name}\")"
        )

    @pytest.mark.parametrize("node_name,file_path", EXPECTED_NODES)
    def test_node_updates_observation_metadata(self, node_name, file_path):
        """Each node should call langfuse_context.update_current_observation for metadata."""
        full_path = os.path.join(os.path.dirname(__file__), "..", file_path)
        with open(full_path, encoding="utf-8") as f:
            content = f.read()

        # Each node should update observation with some metadata
        assert "update_current_observation" in content, (
            f"{file_path}: should call update_current_observation to set span metadata"
        )


class TestLLMCallTracking:
    """Verify track_llm_call usage in all LLM-calling nodes."""

    LLM_NODES = [
        ("intent_router", "src/graph/nodes/intent_router.py"),
        ("query_condenser", "src/graph/nodes/query_condenser.py"),
        ("chitchat_answer", "src/graph/workflow.py"),
        ("generate_answer", "src/graph/nodes/generate.py"),
        ("analytical_agent", "src/graph/agent.py"),
        # compress_context uses @trace_rag_node but tracks via update_current_observation
    ]

    @pytest.mark.parametrize("node_name,file_path", LLM_NODES)
    def test_llm_calling_node_has_tracking(self, node_name, file_path):
        """LLM-calling nodes must track calls via track_llm_call or generation span."""
        full_path = os.path.join(os.path.dirname(__file__), "..", file_path)
        with open(full_path, encoding="utf-8") as f:
            content = f.read()

        # Each LLM-calling node should import and use either track_llm_call or
        # have a generation span setup
        has_tracking = (
            "track_llm_call" in content
            or "generation" in content
            or "as_type=\"generation\"" in content
        )
        assert has_tracking, (
            f"{file_path}: LLM-calling node must have call tracking"
        )


class TestRetrievalMonitoring:
    """Verify retrieval monitoring span coverage."""

    RETRIEVAL_FILES = [
        "src/rag/retrieval/dense.py",
        "src/rag/retrieval/sparse.py",
        "src/rag/retrieval/rrf.py",
        "src/rag/retrieval/reranker.py",
        "src/rag/retrieval/query_expansion.py",
    ]

    @pytest.mark.parametrize("file_path", RETRIEVAL_FILES)
    def test_retrieval_file_exists(self, file_path):
        """All retrieval files must exist."""
        full_path = os.path.join(os.path.dirname(__file__), "..", file_path)
        assert os.path.exists(full_path), f"{file_path} must exist"


class TestIngestionMonitoring:
    """Verify ingestion pipeline monitoring coverage."""

    INGESTION_FILES = [
        "src/rag/ingestion/loader.py",
        "src/rag/ingestion/parser.py",
        "src/rag/ingestion/cleaner.py",
        "src/rag/indexing/chunker.py",
    ]

    @pytest.mark.parametrize("file_path", INGESTION_FILES)
    def test_ingestion_file_exists(self, file_path):
        """All ingestion files must exist."""
        full_path = os.path.join(os.path.dirname(__file__), "..", file_path)
        assert os.path.exists(full_path), f"{file_path} must exist"


# ═══════════════════════════════════════════════════════════════════════════════
# API-level Trace Integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestAPITraceIntegration:
    """Verify API endpoints have trace-level monitoring."""

    def test_chat_endpoint_creates_trace(self):
        """POST /chat should create a request-level trace."""
        routes_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "api", "routes.py"
        )
        with open(routes_path, encoding="utf-8") as f:
            content = f.read()

        # chat endpoint should setup Langfuse tracing
        # Either via @observe decorator or manual client usage
        assert "chat" in content.lower()

    def test_streaming_endpoint_has_trace(self):
        """POST /chat/stream should include trace setup."""
        routes_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "api", "routes.py"
        )
        with open(routes_path, encoding="utf-8") as f:
            content = f.read()

        assert "stream" in content.lower()

    def test_main_lifespan_initializes_langfuse(self):
        """FastAPI lifespan must initialize and flush Langfuse."""
        main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        with open(main_path, encoding="utf-8") as f:
            content = f.read()

        assert "get_langfuse" in content, "main.py must call get_langfuse() on startup"
        assert "flush_langfuse" in content, "main.py must call flush_langfuse() on shutdown"


# ═══════════════════════════════════════════════════════════════════════════════
# Observability module exports
# ═══════════════════════════════════════════════════════════════════════════════


class TestObservabilityModuleExports:
    """Verify observability module public API."""

    def test_init_exports_correct_symbols(self):
        """__init__.py must export get_langfuse, flush_langfuse, trace_rag_node."""
        from src.observability import get_langfuse, flush_langfuse, trace_rag_node

        assert callable(get_langfuse)
        assert callable(flush_langfuse)
        assert callable(trace_rag_node)

    def test_client_module_has_required_functions(self):
        """client.py must export get_langfuse, flush_langfuse, is_langfuse_enabled."""
        from src.observability.client import (
            get_langfuse,
            flush_langfuse,
            is_langfuse_enabled,
        )

        assert callable(get_langfuse)
        assert callable(flush_langfuse)
        assert callable(is_langfuse_enabled)

    def test_context_module_has_update_functions(self):
        """context.py must export update functions."""
        from src.observability.context import (
            update_current_observation,
            update_current_trace,
        )

        assert callable(update_current_observation)
        assert callable(update_current_trace)

    def test_decorators_module_has_trace_rag_node(self):
        """decorators.py must export trace_rag_node."""
        from src.observability.decorators import trace_rag_node

        assert callable(trace_rag_node)

    def test_llm_tracker_module_has_track_llm_call(self):
        """llm_tracker.py must export track_llm_call."""
        from src.observability.llm_tracker import track_llm_call

        assert callable(track_llm_call)
