"""Unified LLM client abstracting OpenAI, Anthropic, Gemini, Groq, and DeepSeek."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Per 1k tokens (USD)
COST_RATES: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "deepseek-chat": {"input": 0.00014, "output": 0.00028},
}

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


@dataclass
class LLMMessage:
    """A single chat message for LLM providers."""

    role: str
    content: str | list[Any]


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""

    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_estimate: float
    latency_ms: float


class LLMClient:
    """
    Unified client for multiple LLM providers.

    API keys are read from environment variables on construction.
    Provider SDK clients are initialized lazily on first use.
    """

    def __init__(self) -> None:
        """Initialize the client and read API keys from the environment."""
        self._openai_api_key = os.environ.get("OPENAI_API_KEY")
        self._anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self._google_api_key = os.environ.get("GOOGLE_API_KEY")
        self._groq_api_key = os.environ.get("GROQ_API_KEY")
        self._deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")

        self._registry: Any | None = None
        self._groq_client: Any | None = None
        self._groq_async_client: Any | None = None
        self._deepseek_client: Any | None = None
        self._deepseek_async_client: Any | None = None

    def call(
        self,
        messages: list[LLMMessage],
        model: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Synchronously invoke an LLM for the given model and messages.

        Args:
            messages: Conversation history as LLMMessage instances.
            model: Model identifier (used for provider routing).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Normalized LLMResponse with content, usage, cost, and latency.
        """
        provider = self._detect_provider(model)
        logger.info("LLM call: model=%s provider=%s", model, provider)
        start = time.perf_counter()

        if provider in ("openai", "anthropic", "google"):
            adapter = self._get_registry().get_provider(model)
            response = adapter.complete(
                messages, model, max_tokens=max_tokens, temperature=temperature, **kwargs
            )
        elif provider == "groq":
            response = self._call_groq(messages, model, max_tokens, temperature, **kwargs)
        elif provider == "deepseek":
            response = self._call_deepseek(messages, model, max_tokens, temperature, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        latency_ms = (time.perf_counter() - start) * 1000
        response.latency_ms = latency_ms
        logger.info(
            "LLM response: model=%s input_tokens=%d output_tokens=%d cost=%.6f latency_ms=%.2f",
            response.model,
            response.input_tokens,
            response.output_tokens,
            response.cost_estimate,
            response.latency_ms,
        )
        return response

    async def call_async(
        self,
        messages: list[LLMMessage],
        model: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Asynchronously invoke an LLM for the given model and messages.

        Args:
            messages: Conversation history as LLMMessage instances.
            model: Model identifier (used for provider routing).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Normalized LLMResponse with content, usage, cost, and latency.
        """
        provider = self._detect_provider(model)
        logger.info("LLM async call: model=%s provider=%s", model, provider)
        start = time.perf_counter()

        if provider in ("openai", "anthropic", "google"):
            adapter = self._get_registry().get_provider(model)
            response = await adapter.complete_async(
                messages, model, max_tokens=max_tokens, temperature=temperature, **kwargs
            )
        elif provider == "groq":
            response = await self._call_groq_async(
                messages, model, max_tokens, temperature, **kwargs
            )
        elif provider == "deepseek":
            response = await self._call_deepseek_async(
                messages, model, max_tokens, temperature, **kwargs
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        latency_ms = (time.perf_counter() - start) * 1000
        response.latency_ms = latency_ms
        logger.info(
            "LLM async response: model=%s input_tokens=%d output_tokens=%d cost=%.6f latency_ms=%.2f",
            response.model,
            response.input_tokens,
            response.output_tokens,
            response.cost_estimate,
            response.latency_ms,
        )
        return response

    def call_with_vision(
        self,
        image_data: bytes,
        prompt: str,
        model: str = "gpt-4o",
    ) -> LLMResponse:
        """
        Analyze an image with a vision-capable model.

        Args:
            image_data: Raw image bytes (JPEG/PNG/WebP).
            prompt: Text instruction for the model.
            model: Vision model (OpenAI gpt-* or Anthropic claude-*).

        Returns:
            LLMResponse from the vision API call.
        """
        provider = self._detect_provider(model)
        if provider not in ("openai", "anthropic"):
            raise ValueError(
                f"Vision is supported for OpenAI and Anthropic models only, got: {model}"
            )

        b64 = base64.standard_b64encode(image_data).decode("ascii")
        media_type = self._guess_image_media_type(image_data)

        if provider == "openai":
            messages = [
                LLMMessage(
                    role="user",
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{b64}",
                            },
                        },
                    ],
                )
            ]
        else:
            messages = [
                LLMMessage(
                    role="user",
                    content=[
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                )
            ]

        logger.info("Vision call: model=%s provider=%s", model, provider)
        return self.call(messages, model=model)

    def call_parallel(self, calls: list[dict[str, Any]]) -> list[LLMResponse]:
        """
        Run multiple LLM calls concurrently.

        Args:
            calls: List of call specs, each a dict with at least ``messages`` and
                ``model``, plus optional ``max_tokens``, ``temperature``, and kwargs.

        Returns:
            List of LLMResponse objects in the same order as ``calls``.
        """
        return self._run_async(self._call_parallel_async(calls))

    async def _call_parallel_async(self, calls: list[dict[str, Any]]) -> list[LLMResponse]:
        tasks = [self._invoke_call_spec(call) for call in calls]
        return list(await asyncio.gather(*tasks))

    async def _invoke_call_spec(self, call: dict[str, Any]) -> LLMResponse:
        spec = dict(call)
        messages = self._normalize_messages(spec.pop("messages"))
        model = spec.pop("model")
        return await self.call_async(messages, model, **spec)

    def _detect_provider(self, model: str) -> str:
        """Route a model name to its provider identifier."""
        name = model.lower()
        if name.startswith(("gpt-", "o1-")):
            return "openai"
        if name.startswith("claude-"):
            return "anthropic"
        if name.startswith("gemini-"):
            return "google"
        if name.startswith(("mixtral-", "llama-")):
            return "groq"
        if name.startswith("deepseek-"):
            return "deepseek"
        raise ValueError(f"Cannot determine provider for model: {model}")

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
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

    def _normalize_messages(self, messages: list[Any]) -> list[LLMMessage]:
        normalized: list[LLMMessage] = []
        for msg in messages:
            if isinstance(msg, LLMMessage):
                normalized.append(msg)
            elif isinstance(msg, dict):
                normalized.append(LLMMessage(role=msg["role"], content=msg["content"]))
            else:
                raise TypeError(f"Expected LLMMessage or dict, got {type(msg)}")
        return normalized

    def _run_async(self, coro: Any) -> Any:
        """Run a coroutine from sync code, including when an event loop is active."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()

    @staticmethod
    def _guess_image_media_type(image_data: bytes) -> str:
        if image_data[:8] == b"\x89PNG\r\n\x1a\n":
            return "image/png"
        if image_data[:3] == b"\xff\xd8\xff":
            return "image/jpeg"
        if image_data[:4] == b"RIFF" and image_data[8:12] == b"WEBP":
            return "image/webp"
        return "image/jpeg"

    def _get_registry(self) -> Any:
        if self._registry is None:
            from packages.llm.providers.registry import ProviderRegistry

            self._registry = ProviderRegistry()
        return self._registry

    def _get_groq_client(self) -> Any:
        if self._groq_client is None:
            if not self._groq_api_key:
                raise ValueError("GROQ_API_KEY is not set")
            from groq import Groq

            self._groq_client = Groq(api_key=self._groq_api_key)
        return self._groq_client

    def _get_groq_async_client(self) -> Any:
        if self._groq_async_client is None:
            if not self._groq_api_key:
                raise ValueError("GROQ_API_KEY is not set")
            from groq import AsyncGroq

            self._groq_async_client = AsyncGroq(api_key=self._groq_api_key)
        return self._groq_async_client

    def _get_deepseek_client(self) -> Any:
        if self._deepseek_client is None:
            if not self._deepseek_api_key:
                raise ValueError("DEEPSEEK_API_KEY is not set")
            from openai import OpenAI

            self._deepseek_client = OpenAI(
                api_key=self._deepseek_api_key,
                base_url=DEEPSEEK_BASE_URL,
            )
        return self._deepseek_client

    def _get_deepseek_async_client(self) -> Any:
        if self._deepseek_async_client is None:
            if not self._deepseek_api_key:
                raise ValueError("DEEPSEEK_API_KEY is not set")
            from openai import AsyncOpenAI

            self._deepseek_async_client = AsyncOpenAI(
                api_key=self._deepseek_api_key,
                base_url=DEEPSEEK_BASE_URL,
            )
        return self._deepseek_async_client

    def _to_openai_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _call_groq(
        self,
        messages: list[LLMMessage],
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_groq_client()
        result = client.chat.completions.create(
            model=model,
            messages=self._to_openai_messages(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        content = result.choices[0].message.content or ""
        input_tokens = result.usage.prompt_tokens if result.usage else 0
        output_tokens = result.usage.completion_tokens if result.usage else 0
        return LLMResponse(
            content=content,
            model=model,
            provider="groq",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_estimate=self._estimate_cost(model, input_tokens, output_tokens),
            latency_ms=0.0,
        )

    async def _call_groq_async(
        self,
        messages: list[LLMMessage],
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_groq_async_client()
        result = await client.chat.completions.create(
            model=model,
            messages=self._to_openai_messages(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        content = result.choices[0].message.content or ""
        input_tokens = result.usage.prompt_tokens if result.usage else 0
        output_tokens = result.usage.completion_tokens if result.usage else 0
        return LLMResponse(
            content=content,
            model=model,
            provider="groq",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_estimate=self._estimate_cost(model, input_tokens, output_tokens),
            latency_ms=0.0,
        )

    def _call_deepseek(
        self,
        messages: list[LLMMessage],
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_deepseek_client()
        result = client.chat.completions.create(
            model=model,
            messages=self._to_openai_messages(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        content = result.choices[0].message.content or ""
        input_tokens = result.usage.prompt_tokens if result.usage else 0
        output_tokens = result.usage.completion_tokens if result.usage else 0
        return LLMResponse(
            content=content,
            model=model,
            provider="deepseek",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_estimate=self._estimate_cost(model, input_tokens, output_tokens),
            latency_ms=0.0,
        )

    async def _call_deepseek_async(
        self,
        messages: list[LLMMessage],
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_deepseek_async_client()
        result = await client.chat.completions.create(
            model=model,
            messages=self._to_openai_messages(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        content = result.choices[0].message.content or ""
        input_tokens = result.usage.prompt_tokens if result.usage else 0
        output_tokens = result.usage.completion_tokens if result.usage else 0
        return LLMResponse(
            content=content,
            model=model,
            provider="deepseek",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_estimate=self._estimate_cost(model, input_tokens, output_tokens),
            latency_ms=0.0,
        )
