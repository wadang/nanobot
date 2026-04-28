import json

import pytest

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.utils import llm_io_log


class LoggingProvider(LLMProvider):
    async def chat(self, *args, **kwargs) -> LLMResponse:
        return LLMResponse(
            content="pong",
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="lookup",
                    arguments={"query": "nanobot"},
                )
            ],
            usage={"prompt_tokens": 3, "completion_tokens": 4},
        )

    def get_default_model(self) -> str:
        return "test-model"


@pytest.mark.asyncio
async def test_chat_with_retry_logs_request_and_response(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "llm_io.jsonl"
    monkeypatch.setenv("NANOBOT_LLM_IO_LOG", "1")
    monkeypatch.setenv("NANOBOT_LLM_IO_LOG_PATH", str(log_path))

    provider = LoggingProvider()

    response = await provider.chat_with_retry(
        messages=[{"role": "user", "content": "ping"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        model="demo-model",
        max_tokens=123,
        temperature=0.2,
    )

    assert response.content == "pong"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [record["phase"] for record in records] == ["request", "response"]
    assert records[0]["interaction_id"] == records[1]["interaction_id"]
    assert records[0]["payload"]["messages"] == [{"role": "user", "content": "ping"}]
    assert records[0]["payload"]["tools"] == [{"type": "function", "function": {"name": "lookup"}}]
    assert records[0]["payload"]["model"] == "demo-model"
    assert records[0]["payload"]["max_tokens"] == 123
    assert records[0]["payload"]["temperature"] == 0.2
    assert records[1]["response"]["content"] == "pong"
    assert records[1]["response"]["tool_calls"][0]["arguments"] == {"query": "nanobot"}
    assert records[1]["response"]["usage"] == {"prompt_tokens": 3, "completion_tokens": 4}


def test_initialize_llm_io_log_creates_session_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NANOBOT_LLM_IO_LOG", "1")
    monkeypatch.delenv("NANOBOT_LLM_IO_LOG_PATH", raising=False)
    monkeypatch.setattr(llm_io_log, "get_logs_dir", lambda: tmp_path)
    monkeypatch.setattr(llm_io_log, "_SESSION_STARTED_AT", "20260428_110000_123456")
    monkeypatch.setattr(llm_io_log, "_SESSION_LOG_PATH", None)
    monkeypatch.setattr(llm_io_log, "_INITIALIZED", False)

    path = llm_io_log.initialize_llm_io_log()
    second_path = llm_io_log.initialize_llm_io_log()

    assert path == second_path
    assert path is not None
    assert path.parent == tmp_path
    assert path.name.startswith("llm_io_20260428_110000_123456_pid")
    assert path.suffix == ".jsonl"
    assert path.exists()
