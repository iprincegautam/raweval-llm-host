"""Shared mock LLM client for examples (no API keys required)."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from packages.llm.client import COST_RATES, LLMClient, LLMMessage, LLMResponse


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost using the same rates as LLMClient."""
    if model in COST_RATES:
        rates = COST_RATES[model]
    else:
        matches = [key for key in COST_RATES if model.startswith(key)]
        if not matches:
            return 0.0
        rates = COST_RATES[max(matches, key=len)]
    return (input_tokens / 1000 * rates["input"]) + (output_tokens / 1000 * rates["output"])


def mock_response(
    model: str,
    content: str,
    *,
    input_tokens: int = 100,
    output_tokens: int = 40,
    latency_ms: float = 5.0,
) -> LLMResponse:
    """Build a deterministic LLMResponse for demos."""
    provider = LLMClient()._detect_provider(model)
    return LLMResponse(
        content=content,
        model=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_estimate=estimate_cost(model, input_tokens, output_tokens),
        latency_ms=latency_ms,
    )


class MockLLMClient(LLMClient):
    """LLMClient that returns canned responses instead of calling real APIs."""

    MOCK_DELAY_SEC: float = 0.0

    def _mock_for_model(self, model: str, messages: list[LLMMessage]) -> LLMResponse:
        user_text = messages[-1].content if messages else ""
        if isinstance(user_text, list):
            user_text = next(
                (b.get("text", "") for b in user_text if isinstance(b, dict) and b.get("type") == "text"),
                "[multimodal]",
            )
        snippets: dict[str, str] = {
            "gpt-4o": f"GPT-4o: Evaluated prompt — {user_text!r}",
            "claude-3-5-sonnet-20241022": f"Claude: Independent judgment on — {user_text!r}",
            "gemini-1.5-pro": f"Gemini: Third opinion on — {user_text!r}",
        }
        content = snippets.get(model, f"Mock response from {model}")
        return mock_response(model, content)

    def call(
        self,
        messages: list[LLMMessage],
        model: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        start = time.perf_counter()
        if self.MOCK_DELAY_SEC:
            time.sleep(self.MOCK_DELAY_SEC)
        response = self._mock_for_model(model, messages)
        response.latency_ms = (time.perf_counter() - start) * 1000
        return response

    async def call_async(
        self,
        messages: list[LLMMessage],
        model: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        start = time.perf_counter()
        if self.MOCK_DELAY_SEC:
            await asyncio.sleep(self.MOCK_DELAY_SEC)
        response = self._mock_for_model(model, messages)
        response.latency_ms = (time.perf_counter() - start) * 1000
        return response
