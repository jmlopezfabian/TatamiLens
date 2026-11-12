"""Backend Ollama (VLM local). Default: qwen3-vl:8b."""

from __future__ import annotations

import json
import logging
import os

from ..video import SampledFrame, encode_jpeg
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
DEFAULT_MODEL = "qwen3-vl:8b"


def _host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class OllamaClassifier(PositionClassifier):
    name = "ollama"
    default_model = DEFAULT_MODEL
    block_size = 6
    image_width = 768

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        block_size: int | None = None,
        retries: int = 2,
        temperature: float = 0.1,
        num_ctx: int = 16384,
        think: bool = False,
        client=None,
        **opts,
    ):
        super().__init__(model, **opts)
        if block_size:
            self.block_size = block_size
        self.retries = retries
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.think = think
        if client is None:
            import ollama

            client = ollama.Client(host=host or _host())
        self.client = client

    @classmethod
    def status(cls) -> BackendStatus:
        try:
            import ollama

            resp = ollama.Client(host=_host(), timeout=3).list()
            models = sorted(m.model for m in resp.models if m.model)
        except Exception as e:  # servidor caído, paquete ausente...
            return BackendStatus("ollama", [], False, DEFAULT_MODEL, f"Ollama no disponible: {e}")
        ok = DEFAULT_MODEL in models
        detail = "" if ok else f"modelo por defecto no instalado: ollama pull {DEFAULT_MODEL}"
        return BackendStatus("ollama", models, ok, DEFAULT_MODEL, detail)

    def _chat(self, messages: list[dict]) -> dict:
        kwargs = dict(
            model=self.model,
            messages=messages,
            format=response_schema(),
            options={"temperature": self.temperature, "num_ctx": self.num_ctx},
        )
        try:
            resp = self.client.chat(think=self.think, **kwargs)
        except Exception as e:
            if "think" not in str(e).lower():
                raise
            resp = self.client.chat(**kwargs)
        return json.loads(
            resp["message"]["content"] if not hasattr(resp, "message") else resp.message.content
        )

    def classify_block(
        self,
        frames: list[SampledFrame],
        athletes: dict[str, str] | None = None,
        prev_labels: list[dict] | None = None,
    ) -> BlockResult:
        listing = "\n".join(f"Image {i + 1}: t={f.t}s" for i, f in enumerate(frames))
        text = (
            context_text(athletes, prev_labels)
            + f"The following {len(frames)} images are consecutive frames of the match, in order.\n"
            + listing
            + f"\nLabel all {len(frames)} frames, one entry per image, in order."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": text,
                "images": [encode_jpeg(f.image, self.image_width) for f in frames],
            },
        ]
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                raw = self._chat(messages)
                labels = align_response(frames, raw)
                if all(lb["confidence"] == 0 for lb in labels):
                    raise ValueError("respuesta sin frames utilizables")
                desc = None
                if raw.get("athlete_A") or raw.get("athlete_B"):
                    desc = {"A": raw.get("athlete_A", ""), "B": raw.get("athlete_B", "")}
                return BlockResult(labels, desc)
            except Exception as e:
                last_err = e
                log.warning("Ollama: bloque t=%s falló (intento %d): %s", frames[0].t, attempt + 1, e)
        log.error("Ollama: bloque t=%s descartado: %s", frames[0].t, last_err)
        return BlockResult([failed_frame(f.t) for f in frames])
