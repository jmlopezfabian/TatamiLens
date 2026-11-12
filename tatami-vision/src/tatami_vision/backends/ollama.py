"""Backend Ollama (VLM local) vía Pydantic AI. Default: qwen3-vl:8b-instruct.

Decisiones de rendimiento (medidas en docs/PERFORMANCE.md):
- Modelo *instruct*: `qwen3-vl:8b` es la variante Thinking e ignora `think=False`; genera miles
  de tokens de razonamiento por bloque (horas por combate).
- Salida compacta: un código `posición:top` por frame (~6 tokens) en lugar de un objeto JSON
  (~44 tokens). Las descripciones de los atletas solo se piden en el primer bloque.
- Mosaico: Qwen3-VL en Ollama reescala cada imagen a una resolución mínima, así que una imagen
  pequeña cuesta lo mismo que una grande; se envían 4 frames por imagen en rejilla 2x2 (`tile=4`).
- API nativa de Ollama (`ollama_native.py`) en lugar del endpoint OpenAI-compatible: 1.7x más
  rápida con imágenes y permite fijar `num_ctx`.
"""

from __future__ import annotations

import logging
import os
from functools import cache
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from ..schema import NEUTRAL_POSITIONS, POSITIONS
from ..video import SampledFrame, encode_jpeg, tile_frames
from .base import (
    SYSTEM_PROMPT,
    BackendStatus,
    BlockResult,
    PositionClassifier,
    failed_frame,
    normalize_frame,
)

log = logging.getLogger(__name__)
DEFAULT_MODEL = "qwen3-vl:8b-instruct"
THINKING_MODELS = ("qwen3-vl:8b",)  # variantes que razonan aunque se pida lo contrario

# Códigos de salida: posiciones neutras solas, el resto con quién está arriba.
CODES: tuple[str, ...] = tuple(
    [p for p in POSITIONS if p in NEUTRAL_POSITIONS]
    + [f"{p}:{t}" for p in POSITIONS if p not in NEUTRAL_POSITIONS for t in ("A", "B")]
)
Code = Literal[CODES]  # type: ignore[valid-type]
CODE_CONFIDENCE = 0.7  # la salida compacta no trae confianza; constante para el suavizado

TOKENS_PER_IMAGE = 1100  # cota superior medida (~500 nativo, ~1040 por /v1) para dimensionar
PROMPT_OVERHEAD_TOKENS = 900  # instrucciones + contexto del bloque anterior
OUTPUT_TOKENS_PER_FRAME = 12
OUTPUT_OVERHEAD_TOKENS = 300  # llaves + descripciones de atletas (primer bloque)

INSTRUCTIONS = SYSTEM_PROMPT.split("Return exactly")[0] + (
    "Output format: for each frame return one code 'position:top', e.g. 'mount:A' or "
    "'half_guard:B'; neutral positions (standing, scramble, separated, not_visible) have no "
    "top, e.g. 'standing'. Return exactly one code per frame, in time order."
)


@cache
def block_output_model(n: int, with_athletes: bool) -> type[BaseModel]:
    """Modelo de salida con exactamente `n` códigos (minItems/maxItems en el JSON Schema, que
    Ollama aplica con decodificación restringida por gramática)."""
    fields: dict[str, Any] = {}
    if with_athletes:
        fields["athlete_A"] = (str, Field(description="Concrete description of athlete A"))
        fields["athlete_B"] = (str, Field(description="Concrete description of athlete B"))
    fields["frames"] = (list[Code], Field(min_length=n, max_length=n))
    return create_model(f"BlockLabels{n}{'WithAthletes' if with_athletes else ''}", **fields)


def code_to_label(t: float, code: str, confidence: float = CODE_CONFIDENCE) -> dict:
    pos, _, top = code.partition(":")
    return normalize_frame(t, pos, top or "none", confidence)


def label_to_code(label: dict) -> str:
    return label["position"] if label["top"] == "none" else f"{label['position']}:{label['top']}"


def _host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def max_block_for_context(context_tokens: int, tile: int) -> int:
    """Frames máximos por petición para que prompt + salida quepan en el contexto."""
    budget = context_tokens - PROMPT_OVERHEAD_TOKENS - OUTPUT_OVERHEAD_TOKENS
    per_image = TOKENS_PER_IMAGE + tile * OUTPUT_TOKENS_PER_FRAME
    return max(1, budget // per_image) * tile


class OllamaClassifier(PositionClassifier):
    name = "ollama"
    default_model = DEFAULT_MODEL
    block_size = 8
    image_width = 768  # solo sin mosaico (tile=1)

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        block_size: int | None = None,
        tile: int = 4,
        context_tokens: int | None = None,
        retries: int = 2,
        temperature: float = 0.1,
        client: Any = None,  # ollama.Client o doble de test
        agent_model: Any = None,  # modelo de Pydantic AI que sustituye a Ollama (tests)
        **opts,
    ):
        super().__init__(model, **opts)
        if tile not in (1, 2, 4):
            raise ValueError("tile debe ser 1, 2 o 4 frames por imagen")
        self.tile = tile
        self.retries = retries
        self.temperature = temperature
        self.context_tokens = context_tokens or int(os.environ.get("TATAMI_OLLAMA_CONTEXT", 8192))
        limit = max_block_for_context(self.context_tokens, tile)
        wanted = block_size or self.block_size
        if wanted > limit:
            log.warning(
                "Ollama: block_size %d no cabe en %d tokens de contexto; uso %d "
                "(sube context_tokens o TATAMI_OLLAMA_CONTEXT para bloques mayores)",
                wanted, self.context_tokens, limit,
            )  # fmt: skip
        self.block_size = min(wanted, limit)
        if self.model in THINKING_MODELS:
            log.warning(
                "Ollama: %s es un modelo con razonamiento (muy lento); usa %s",
                self.model, DEFAULT_MODEL,
            )  # fmt: skip
        self._warned_thinking = False
        if agent_model is None:
            if client is None:
                import ollama

                client = ollama.Client(host=host or _host())
            from .ollama_native import ollama_native_model

            agent_model = ollama_native_model(self.model, client, num_ctx=self.context_tokens)
        self.agent = self._build_agent(agent_model)

    def _build_agent(self, agent_model: Any):
        import pydantic_ai
        from pydantic_ai import Agent, ModelSettings

        pydantic_ai.BANNER_ENABLED = False  # el CLI/API controla su salida
        return Agent(
            agent_model,
            instructions=INSTRUCTIONS,
            model_settings=ModelSettings(temperature=self.temperature),
            retries=1,
        )

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

    def _prompt(
        self, frames: list[SampledFrame], athletes: dict[str, str] | None, prev: list[dict] | None
    ) -> list[Any]:
        from pydantic_ai import BinaryContent

        lines = []
        if athletes:
            lines.append(
                "This is a continuation of the same match. Keep the same identities:\n"
                f"Athlete A: {athletes.get('A', '')}\nAthlete B: {athletes.get('B', '')}"
            )
            if prev:
                lines.append(f"Codes of the previous frames: {[label_to_code(p) for p in prev]}")
        n = len(frames)
        if self.tile == 1:
            lines.append(f"The following {n} images are consecutive frames of the match, in order.")
            lines += [f"Image {i + 1}: t={f.t}s" for i, f in enumerate(frames)]
            images = [encode_jpeg(f.image, self.image_width) for f in frames]
        else:
            groups = [frames[i : i + self.tile] for i in range(0, n, self.tile)]
            lines.append(
                f"{n} consecutive frames of the match. Each image is a grid of up to {self.tile} "
                "frames, read left-to-right then top-to-bottom; each frame is numbered in its "
                "top-left corner."
            )
            lines += [
                f"Image {gi + 1}: " + ", ".join(f"frame {k + 1} t={f.t}s" for k, f in enumerate(g))
                for gi, g in enumerate(groups)
            ]
            images = [encode_jpeg(tile_frames([f.image for f in g]), None) for g in groups]
        lines.append(f"Return exactly {n} codes, one per frame, in time order.")
        return ["\n".join(lines)] + [BinaryContent(b, media_type="image/jpeg") for b in images]

    def _check_thinking(self, result) -> None:
        from pydantic_ai.messages import ThinkingPart

        if self._warned_thinking:
            return
        for msg in result.new_messages():
            if any(isinstance(p, ThinkingPart) for p in getattr(msg, "parts", [])):
                self._warned_thinking = True
                log.warning("Ollama: %s está razonando (lento); usa un modelo -instruct", self.model)
                return

    def classify_block(
        self,
        frames: list[SampledFrame],
        athletes: dict[str, str] | None = None,
        prev_labels: list[dict] | None = None,
    ) -> BlockResult:
        from pydantic_ai import ModelSettings
        from pydantic_ai.output import NativeOutput

        n = len(frames)
        out_model = block_output_model(n, athletes is None)
        prompt = self._prompt(frames, athletes, prev_labels)
        # Tope de salida: si el modelo se desboca (p.ej. razonando) se corta en vez de tardar minutos.
        settings = ModelSettings(max_tokens=OUTPUT_TOKENS_PER_FRAME * n + OUTPUT_OVERHEAD_TOKENS)
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                result = self.agent.run_sync(
                    prompt, output_type=NativeOutput(out_model), model_settings=settings
                )
                self._check_thinking(result)
                out = result.output
                labels = [code_to_label(f.t, c) for f, c in zip(frames, out.frames, strict=True)]
                desc = None
                if getattr(out, "athlete_A", None) or getattr(out, "athlete_B", None):
                    desc = {"A": out.athlete_A, "B": out.athlete_B}
                return BlockResult(labels, desc)
            except Exception as e:
                last_err = e
                log.warning("Ollama: bloque t=%s falló (intento %d): %s", frames[0].t, attempt + 1, e)
        log.error("Ollama: bloque t=%s descartado: %s", frames[0].t, last_err)
        return BlockResult([failed_frame(f.t) for f in frames])
