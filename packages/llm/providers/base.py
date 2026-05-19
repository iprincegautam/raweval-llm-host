"""Abstract base class for LLM provider adapters."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from packages.llm.client import COST_RATES, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """Base interface for provider-specific LLM adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. ``openai``, ``anthropic``)."""

    @property
    @abstractmethod
    def supported_models(self) -> list[str]:
        """Model IDs this provider advertises as supported."""

    def supports_model(self, model: str) -> bool:
        """
        Return whether this provider can handle the given model.

        Default: exact match or prefix match against ``supported_models``.
        """
        if model in self.supported_models:
            return True
        return any(model.startswith(prefix) for prefix in self.supported_models)

    def complete(
        self,
        messages: list[LLMMessage],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Synchronously complete a chat request.

        Runs ``complete_async`` via ``asyncio.run`` (or a thread when a loop
        is already running).
        """
        return self._run_async(self.complete_async(messages, model, **kwargs))

    @abstractmethod
    async def complete_async(
        self,
        messages: list[LLMMessage],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """Asynchronously complete a chat request."""

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate USD cost from token usage and known per-1k rates."""
        rates = self._rates_for_model(model)
        if rates is None:
            logger.warning("No cost rates configured for model: %s", model)
            return 0.0
        return (input_tokens / 1000 * rates["input"]) + (output_tokens / 1000 * rates["output"])

    def _rates_for_model(self, model: str) -> dict[str, float] | None:
        if model in COST_RATES:
            return COST_RATES[model]
        matches = [key for key in COST_RATES if model.startswith(key)]
        if not matches:
            return None
        best = max(matches, key=len)
        return COST_RATES[best]

    def _build_response(
        self,
        *,
        content: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float = 0.0,
    ) -> LLMResponse:
        """Construct a normalized LLMResponse with cost estimate."""
        return LLMResponse(
            content=content,
            model=model,
            provider=self.name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_estimate=self.estimate_cost(model, input_tokens, output_tokens),
            latency_ms=latency_ms,
        )

    @staticmethod
    def _run_async(coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
