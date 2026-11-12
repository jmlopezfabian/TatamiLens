"""Backend Anthropic (Claude vía SDK oficial; extra opcional `[anthropic]`)."""

from __future__ import annotations

import json
import logging
import os

from ..video import SampledFrame, encode_jpeg_b64
from .base import (
    SYSTEM_PROMPT,
    BackendStatus,
    BlockResult,
    PositionClassifier,
    align_response,
    context_text,
    failed_frame,
    response_schema,
)

log = logging.getLogger(__name__)
DEFAULT_MODEL = "claude-opus-5-5"


class AnthropicClassifier(PositionClassifier):
    name = "anthropic"
    default_model = DEFAULT_MODEL
    block_size = 48
    image_width = 960

    def __init__(self, model=None, block_size=None, retries: int = 1, client=None, **opts):
        super().__init__(model, **opts)
        if block_size:
            self.block_size = block_size
        self.retries = retries
        if client is None:
            import anthropic  # extra opcional

            client = anthropic.Anthropic()  # lee ANTHROPIC_API_KEY
        self.client = client

    @classmethod
    def status(cls) -> BackendStatus:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return BackendStatus(
                "anthropic", [DEFAULT_MODEL], False, DEFAULT_MODEL,
                "instala el extra: uv sync --extra anthropic",
            )  # fmt: skip
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return BackendStatus(
                "anthropic", [DEFAULT_MODEL], False, DEFAULT_MODEL, "falta ANTHROPIC_API_KEY"
            )
        return BackendStatus("anthropic", [DEFAULT_MODEL], True, DEFAULT_MODEL)

    def _call(self, content: list[dict]) -> dict:
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=32000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": "high", "format": {"type": "json_schema", "schema": response_schema()}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            msg = stream.get_final_message()
        if msg.stop_reason == "refusal":
            raise RuntimeError(f"Claude rechazó la solicitud: {getattr(msg, 'stop_details', '')}")
        if msg.stop_reason == "max_tokens":
            raise RuntimeError("Respuesta truncada (max_tokens)")
        text = next(b.text for b in msg.content if b.type == "text")
        return json.loads(text)

    def classify_block(
        self,
        frames: list[SampledFrame],
        athletes: dict[str, str] | None = None,
        prev_labels: list[dict] | None = None,
    ) -> BlockResult:
        content: list[dict] = []
        ctx = context_text(athletes, prev_labels)
        if ctx:
            content.append({"type": "text", "text": ctx})
        for f in frames:
            content.append({"type": "text", "text": f"t={f.t}s"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": encode_jpeg_b64(f.image, self.image_width),
                },
            })  # fmt: skip
        content.append({"type": "text", "text": f"Label all {len(frames)} frames above."})
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                raw = self._call(content)
                desc = {"A": raw.get("athlete_A", ""), "B": raw.get("athlete_B", "")}
                return BlockResult(align_response(frames, raw), desc)
            except Exception as e:
                last_err = e
                log.warning("Anthropic: bloque t=%s falló (intento %d): %s", frames[0].t, attempt + 1, e)
        log.error("Anthropic: bloque t=%s descartado: %s", frames[0].t, last_err)
        return BlockResult([failed_frame(f.t) for f in frames])
