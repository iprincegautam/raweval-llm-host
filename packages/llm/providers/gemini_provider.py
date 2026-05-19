"""Google Gemini provider adapter."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from packages.llm.client import LLMMessage, LLMResponse
from packages.llm.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    """Google Generative AI (Gemini) adapter."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._configured = False

    @property
    def name(self) -> str:
        return "google"

    @property
    def supported_models(self) -> list[str]:
        return ["gemini-1.5-pro", "gemini-1.5-flash"]

    def supports_model(self, model: str) -> bool:
        if model.lower().startswith("gemini-"):
            return True
        return super().supports_model(model)

    def _configure(self) -> None:
        if not self._configured:
            if not self._api_key:
                raise ValueError("GOOGLE_API_KEY is not set")
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            self._configured = True
            logger.debug("Configured google.generativeai")

    async def complete_async(
        self,
        messages: list[LLMMessage],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """Call Gemini generate_content (sync SDK wrapped in a thread)."""
        return await asyncio.to_thread(self._complete_sync, messages, model, **kwargs)

    def _complete_sync(
        self,
        messages: list[LLMMessage],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        max_tokens = kwargs.pop("max_tokens", 1000)
        temperature = kwargs.pop("temperature", 0.7)

        self._configure()
        import google.generativeai as genai

        gemini_model = genai.GenerativeModel(model)
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            **{k: v for k, v in kwargs.items() if k in ("top_p", "top_k", "stop_sequences")},
        )

        logger.info("Gemini request: model=%s", model)
        result = gemini_model.generate_content(
            self._to_gemini_contents(messages),
            generation_config=generation_config,
        )

        content = result.text or ""
        usage = getattr(result, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        return self._build_response(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    @staticmethod
    def _to_gemini_contents(messages: list[LLMMessage]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = "user" if msg.role in ("user", "system") else "model"
            if isinstance(msg.content, str):
                contents.append({"role": role, "parts": [msg.content]})
            else:
                parts: list[Any] = []
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
                contents.append({"role": role, "parts": parts})
        return contents
