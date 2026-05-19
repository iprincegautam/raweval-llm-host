"""Singleton registry for LLM provider adapters."""

from __future__ import annotations

import logging

from packages.llm.providers.anthropic_provider import AnthropicProvider
from packages.llm.providers.base import BaseProvider
from packages.llm.providers.gemini_provider import GeminiProvider
from packages.llm.providers.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Singleton registry mapping models to provider instances.

    All built-in providers are registered automatically on first init.
    """

    _instance: ProviderRegistry | None = None

    def __new__(cls) -> ProviderRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._providers: list[BaseProvider] = []
            cls._instance._ready = False
        return cls._instance

    def __init__(self) -> None:
        if self._ready:
            return
        self._register_defaults()
        self._ready = True
        logger.debug("ProviderRegistry initialized with %d providers", len(self._providers))

    def register(self, provider: BaseProvider) -> None:
        """Register a provider instance."""
        if provider not in self._providers:
            self._providers.append(provider)
            logger.info("Registered provider: %s", provider.name)

    def get_provider(self, model: str) -> BaseProvider:
        """
        Resolve the provider responsible for ``model``.

        Raises:
            ValueError: If no registered provider supports the model.
        """
        for provider in self._providers:
            if provider.supports_model(model):
                return provider
        raise ValueError(f"No provider registered for model: {model}")

    @property
    def providers(self) -> list[BaseProvider]:
        """All registered provider instances."""
        return list(self._providers)

    def _register_defaults(self) -> None:
        for provider in (
            OpenAIProvider(),
            AnthropicProvider(),
            GeminiProvider(),
        ):
            self.register(provider)
