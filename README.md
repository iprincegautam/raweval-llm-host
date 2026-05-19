One interface. Seven models. Every provider abstracted.

raweval-llm-host is the unified LLM client layer that powered RawEval's multi-model evaluation pipeline — the chatbox at chat.raweval.com querying GPT-4, Claude, Gemini, Grok, and DeepSeek in parallel. This is the provider abstraction layer: one call signature regardless of which model you're hitting.

## What it does

**Unified interface** — `LLMClient.call()` accepts `LLMMessage` lists and a model string. Routing to OpenAI, Anthropic, Google, Groq, or DeepSeek is automatic from the model prefix.

**Vision / multimodal** — `call_with_vision()` encodes image bytes as base64 and builds provider-native content blocks (OpenAI `image_url`, Anthropic `image`).

**Parallel calls** — `call_parallel()` runs multiple model requests concurrently via `asyncio.gather`. Same call dict shape as a single call; results return in order.

**Cost tracking** — Every `LLMResponse` includes `input_tokens`, `output_tokens`, and `cost_estimate` computed from hardcoded per-1k token rates.

## Supported models

| Model | Provider | Quality score | Input cost per 1k |
|-------|----------|---------------|-------------------|
| `gpt-4o` | OpenAI | 9.5 | $0.005 |
| `gpt-4o-mini` | OpenAI | 8.0 | $0.00015 |
| `claude-3-5-sonnet-20241022` | Anthropic | 9.3 | $0.003 |
| `claude-3-haiku-20240307` | Anthropic | 7.5 | $0.00025 |
| `gemini-1.5-pro` | Google | 9.0 | $0.00125 |
| `gemini-1.5-flash` | Google | 7.8 | $0.000075 |
| `deepseek-chat` | DeepSeek | 8.5 | $0.00014 |

Groq (`mixtral-*`, `llama-*`) is routed in `LLMClient` but not yet wrapped in a provider adapter. Quality scores are RawEval internal benchmarks (1–10), not third-party leaderboards.

## Quick start

```bash
pip install openai anthropic google-generativeai groq
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export GROQ_API_KEY=...
export DEEPSEEK_API_KEY=...
```

```python
from packages import LLMClient, LLMMessage

client = LLMClient()
response = client.call(
    [LLMMessage(role="user", content="What is the capital of France?")],
    model="gpt-4o",
)
print(response.content, response.cost_estimate)
```

## The three examples

- **`examples/basic_call.py`** — Same `call()` signature against GPT-4o, Claude, and Gemini; prints `LLMResponse` fields. Mocked, no keys.
- **`examples/parallel_calls.py`** — Three-judge `call_parallel()` vs sequential timing. Mocked 300ms latency per judge.
- **`examples/vision_call.py`** — Programmatic 10×10 PNG + `call_with_vision()` multimodal path. Mocked annotation response.

```bash
python examples/basic_call.py
python examples/parallel_calls.py
python examples/vision_call.py
```

## How it fits into RawEval

| RawEval surface | Client API |
|-----------------|------------|
| 3-judge evaluation panel | `call_parallel()` — three models grade the same prompt concurrently |
| chat.raweval.com chatbox | `call()` / `call_async()` — single model per user turn |
| Annotation pipeline | `call_with_vision()` — image + rubric prompt for label QA |

Provider logic lives in `packages/llm/providers/` (`OpenAIProvider`, `AnthropicProvider`, `GeminiProvider`) behind a `ProviderRegistry` singleton. `LLMClient` is the only import most callers need.

## License

MIT
