# Demonstrates the unified LLM client — one interface for all providers

"""Run: python examples/basic_call.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.mock_client import MockLLMClient, mock_response
from packages.llm import LLMMessage, LLMResponse


class DemoLLMClient(MockLLMClient):
    """Subclass LLMClient with mocked provider calls (no API keys)."""

    def _mock_for_model(self, model: str, messages: list[LLMMessage]) -> LLMResponse:
        responses = {
            "gpt-4o": (
                "The capital of France is Paris. It is the largest city in France "
                "and a major European cultural center."
            ),
            "claude-3-5-sonnet-20241022": (
                "Paris is the capital and most populous city of France, known for "
                "landmarks such as the Eiffel Tower and the Louvre."
            ),
            "gemini-1.5-pro": (
                "France's capital is Paris, situated on the Seine in northern France."
            ),
        }
        content = responses[model]
        return mock_response(model, content, input_tokens=25, output_tokens=35)


def print_response(response: LLMResponse) -> None:
    print(f"  model:         {response.model}")
    print(f"  provider:      {response.provider}")
    print(f"  content:       {response.content[:72]}{'...' if len(response.content) > 72 else ''}")
    print(f"  cost_estimate: ${response.cost_estimate:.6f}")
    print(f"  latency_ms:    {response.latency_ms:.2f}")
    print(f"  (tokens in={response.input_tokens}, out={response.output_tokens})")


def main() -> None:
    client = DemoLLMClient()
    prompt = LLMMessage(role="user", content="What is the capital of France?")

    models = ["gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-pro"]

    print("Same LLMClient.call() interface across providers:\n")
    for model in models:
        print(f"--- {model} ---")
        response = client.call([prompt], model=model)
        print_response(response)
        print()


if __name__ == "__main__":
    main()
