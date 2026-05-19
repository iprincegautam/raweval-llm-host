# Demonstrates parallel multi-model calls — used in RawEval's 3-judge panel

"""Run: python examples/parallel_calls.py"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.mock_client import MockLLMClient, mock_response
from packages.llm import LLMMessage, LLMResponse


class ParallelDemoClient(MockLLMClient):
    """Mock client with artificial latency to show parallel vs sequential speedup."""

    MOCK_DELAY_SEC = 0.3

    def _mock_for_model(self, model: str, messages: list[LLMMessage]) -> LLMResponse:
        judge_labels = {
            "gpt-4o": "Judge A (GPT-4o): PASS — response meets rubric criteria.",
            "claude-3-5-sonnet-20241022": "Judge B (Claude): FAIL — factual error in step 2.",
            "gemini-1.5-pro": "Judge C (Gemini): PASS — reasoning is sound overall.",
        }
        return mock_response(model, judge_labels[model], input_tokens=200, output_tokens=30)


def print_response(response: LLMResponse, label: str) -> None:
    print(f"  [{label}] {response.model} ({response.provider})")
    print(f"    content:       {response.content}")
    print(f"    cost_estimate: ${response.cost_estimate:.6f}")
    print(f"    latency_ms:    {response.latency_ms:.2f}")


def main() -> None:
    client = ParallelDemoClient()
    prompt = LLMMessage(
        role="user",
        content="Grade this model answer against the rubric: 'Photosynthesis converts light to chemical energy.'",
    )

    calls = [
        {"messages": [prompt], "model": "gpt-4o"},
        {"messages": [prompt], "model": "claude-3-5-sonnet-20241022"},
        {"messages": [prompt], "model": "gemini-1.5-pro"},
    ]

    print("RawEval 3-judge panel — parallel call_parallel():\n")
    parallel_start = time.perf_counter()
    parallel_responses = client.call_parallel(calls)
    parallel_elapsed_ms = (time.perf_counter() - parallel_start) * 1000

    for i, response in enumerate(parallel_responses, start=1):
        print_response(response, f"judge {i}")
    print(f"\n  Total wall time (parallel): {parallel_elapsed_ms:.0f} ms\n")

    print("Same judges run sequentially:\n")
    sequential_start = time.perf_counter()
    sequential_responses: list[LLMResponse] = []
    for call in calls:
        messages = call["messages"]
        model = call["model"]
        sequential_responses.append(client.call(messages, model=model))
    sequential_elapsed_ms = (time.perf_counter() - sequential_start) * 1000

    for i, response in enumerate(sequential_responses, start=1):
        print_response(response, f"judge {i}")
    print(f"\n  Total wall time (sequential): {sequential_elapsed_ms:.0f} ms")
    print(
        f"\n  Speedup: ~{sequential_elapsed_ms / parallel_elapsed_ms:.1f}x faster with call_parallel()"
    )


if __name__ == "__main__":
    main()
