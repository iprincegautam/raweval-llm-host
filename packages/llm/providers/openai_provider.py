"""OpenAI provider adapter."""

from __future__ import annotations

import logging
import os
from typing import Any

from packages.llm.client import LLMMessage, LLMResponse
from packages.llm.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI chat completions (text and multimodal vision)."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client: Any | None = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def supported_models(self) -> list[str]:
        return ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]

    def supports_model(self, model: str) -> bool:
        lower = model.lower()
        if lower.startswith(("gpt-", "o1-")):
            return True
        return super().supports_model(model)

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key)
            logger.debug("Initialized AsyncOpenAI client")
        return self._client

    async def complete_async(
        self,
        messages: list[LLMMessage],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Call OpenAI chat completions API.

        Supports text and multimodal content blocks (vision) on ``LLMMessage.content``.
        """
        max_tokens = kwargs.pop("max_tokens", 1000)
        temperature = kwargs.pop("temperature", 0.7)

        client = self._get_client()
        create_kwargs = self._create_kwargs(model, max_tokens, temperature, kwargs)

        logger.info("OpenAI request: model=%s", model)
        result = await client.chat.completions.create(
            model=model,
            messages=self._to_openai_messages(messages),
            **create_kwargs,
        )

        content = result.choices[0].message.content or ""
        input_tokens = result.usage.prompt_tokens if result.usage else 0
        output_tokens = result.usage.completion_tokens if result.usage else 0

        return self._build_response(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    @staticmethod
    def _to_openai_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """Map LLMMessage list to OpenAI chat messages format."""
        return [{"role": m.role, "content": m.content} for m in messages]

    @staticmethod
    def _create_kwargs(
        model: str,
        max_tokens: int,
        temperature: float,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        create_kwargs = dict(kwargs)
        if model.lower().startswith("o1-"):
            create_kwargs.setdefault("max_completion_tokens", max_tokens)
        else:
            create_kwargs.setdefault("max_tokens", max_tokens)
            create_kwargs.setdefault("temperature", temperature)
        return create_kwargs
