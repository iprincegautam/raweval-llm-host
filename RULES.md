# RULES — raweval-llm-host

## Must always

1. Route models by prefix/name in `LLMClient._resolve_provider()` — do not scatter provider detection across examples.
2. Return `LLMResponse` with all fields populated: `content`, `model`, `provider`, `input_tokens`, `output_tokens`, `cost_estimate`, `latency_ms`.
3. Read API keys only from environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`).
4. Keep example scripts runnable without keys (mock path or clear skip message).
5. Update `README.md` model table when adding a new supported model or cost rate.

## Must never

1. Commit real API keys or `.env` files — only `.env.example` with placeholders.
2. Import from `packages.api` or other RawEval backend modules — this repo is standalone.
3. Change `agent.yaml` required fields (`name`, `version`, `description`) without updating AgentEval validators.
4. Remove or relocate `agent.yaml`, `SOUL.md`, or `RULES.md` from the repository root.
5. Block the event loop on sync I/O inside `call_async()` or `call_parallel()` without `asyncio.to_thread` or native async SDK calls.

## Code conventions

- Public API exports: `LLMClient`, `LLMMessage`, `LLMResponse` from `packages.llm`.
- New providers: implement `BaseLLMProvider` in `packages/llm/providers/` and register in `ProviderRegistry`.
- Logging via `logging.getLogger(__name__)` — no custom RawEval logger imports.

## Testing expectations

- `python examples/basic_call.py` — prints three mocked provider responses.
- `python examples/parallel_calls.py` — demonstrates parallel vs sequential timing.
- `python examples/vision_call.py` — multimodal path with synthetic PNG.

## AgentEval alignment

This repository is intentionally **GitAgent-compatible** so tools like AgentEval can:

- Parse `agent.yaml` for identity and runtime config
- Evaluate `SOUL.md` for coherent maintainer persona
- Score `RULES.md` for testable must/must-never constraints

When editing agent definitions, keep rules **specific and verifiable** — avoid vague guidance like "write good code."
