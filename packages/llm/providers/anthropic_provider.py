"""Anthropic provider adapter."""

from __future__ import annotations

import logging
import os
from typing import Any

from packages.llm.client import LLMMessage, LLMResponse
from packages.llm.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """Anthropic Claude messages API."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client: Any | None = None

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def supported_models(self) -> list[str]:
        return ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]

    def supports_model(self, model: str) -> bool:
        if model.lower().startswith("claude-"):
            return True
        return super().supports_model(model)

    def _get_client(self) -> Any:
        if self._client is None:
            if not self._api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self._api_key)
            logger.debug("Initialized AsyncAnthropic client")
        return self._client

    async def complete_async(
        self,
        messages: list[LLMMessage],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call Anthropic messages API with optional system parameter."""
        max_tokens = kwargs.pop("max_tokens", 1000)
        temperature = kwargs.pop("temperature", 0.7)

        system, anthropic_messages = self._split_messages(messages)
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages,
            **kwargs,
        }
        if system:
            request["system"] = system

        logger.info("Anthropic request: model=%s", model)
        client = self._get_client()
        result = await client.messages.create(**request)

        content = "".join(
            block.text for block in result.content if getattr(block, "type", None) == "text"
        )
        return self._build_response(
            content=content,
            model=model,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
        )

    @staticmethod
    def _split_messages(
        messages: list[LLMMessage],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Extract system prompt and convert remaining messages for Anthropic."""
        system: str | None = None
        anthropic_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                if isinstance(msg.content, str):
                    system = msg.content
                else:
                    system = " ".join(
                        block.get("text", "")
                        for block in msg.content
                        if isinstance(block, dict)
                    )
            else:
                anthropic_messages.append({"role": msg.role, "content": msg.content})
        return system, anthropic_messages
