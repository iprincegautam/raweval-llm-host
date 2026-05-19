# Demonstrates vision/multimodal support — used in RawEval's annotation pipeline

"""Run: python examples/vision_call.py"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.mock_client import MockLLMClient, mock_response
from packages.llm import LLMMessage, LLMResponse


def make_test_png(width: int = 10, height: int = 10, rgb: tuple[int, int, int] = (220, 50, 50)) -> bytes:
    """Build a minimal 10×10 RGB PNG without external dependencies."""
    r, g, b = rgb
    raw_rows = b"".join(
        b"\x00" + bytes([r, g, b] * width) for _ in range(height)
    )
    compressed = zlib.compress(raw_rows, level=9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


class VisionDemoClient(MockLLMClient):
    """Mock client; overrides call() so call_with_vision() uses mocked vision output."""

    def _mock_for_model(self, model: str, messages: list[LLMMessage]) -> LLMResponse:
        has_image = any(
            isinstance(m.content, list)
            and any(isinstance(b, dict) and b.get("type") in ("image_url", "image") for b in m.content)
            for m in messages
        )
        if has_image:
            content = (
                "Annotation: 10×10 red square detected. "
                "Label suggestion: solid_color_patch. Confidence: 0.97"
            )
            return mock_response(model, content, input_tokens=850, output_tokens=28)
        return super()._mock_for_model(model, messages)


def print_vision_response(response: LLMResponse, image_bytes: int) -> None:
    print("  LLMResponse (vision):")
    print(f"    model:         {response.model}")
    print(f"    provider:      {response.provider}")
    print(f"    content:       {response.content}")
    print(f"    input_tokens:  {response.input_tokens}")
    print(f"    output_tokens: {response.output_tokens}")
    print(f"    cost_estimate: ${response.cost_estimate:.6f}")
    print(f"    latency_ms:    {response.latency_ms:.2f}")
    print(f"    (image payload: {image_bytes} bytes PNG)")


def main() -> None:
    client = VisionDemoClient()
    png_bytes = make_test_png()
    print(f"Generated test image: {len(png_bytes)} byte PNG (10×10)\n")

    print("call_with_vision() — builds multimodal message, then calls provider:\n")
    response = client.call_with_vision(
        image_data=png_bytes,
        prompt="Describe this image for the annotation pipeline.",
        model="gpt-4o",
    )
    print_vision_response(response, len(png_bytes))


if __name__ == "__main__":
    main()
