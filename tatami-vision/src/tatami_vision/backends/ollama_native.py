"""Modelo de Pydantic AI sobre la API nativa de Ollama (`/api/chat`).

`pydantic_ai.models.ollama.OllamaModel` usa el endpoint OpenAI-compatible (`/v1`), que con
Qwen3-VL en Ollama 0.31 (medido en docs/PERFORMANCE.md):
- codifica cada imagen con ~1040 tokens frente a ~500 por la API nativa (bloque 1.7x más lento);
- no acepta `num_ctx`, así que el contexto queda en el valor del servidor (4096).

Este adaptador es un `FunctionModel`: Pydantic AI le pasa los mensajes, las instrucciones, los
ajustes y el JSON Schema de `NativeOutput`, y él hace la llamada nativa con `format=<schema>`
(decodificación restringida por gramática) y `options` (`num_ctx`, `num_predict`, temperatura).
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import BinaryContent, ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.messages import SystemPromptPart, ThinkingPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage


def _user_content(content: Any) -> tuple[str, list[bytes]]:
    if isinstance(content, str):
        return content, []
    texts, images = [], []
    for item in content:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, BinaryContent) and item.is_image:
            images.append(item.data)
        else:
            raise TypeError(f"contenido no soportado por el adaptador de Ollama: {type(item)}")
    return "\n".join(texts), images


def to_ollama_messages(messages: list[ModelMessage], instructions: str | None) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": instructions}] if instructions else []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, SystemPromptPart):
                    out.append({"role": "system", "content": part.content})
                elif isinstance(part, UserPromptPart):
                    text, images = _user_content(part.content)
                    out.append({"role": "user", "content": text, **({"images": images} if images else {})})
                elif hasattr(part, "model_response"):  # reintento (p.ej. salida inválida)
                    out.append({"role": "user", "content": part.model_response()})
        elif isinstance(msg, ModelResponse):
            text = "".join(p.content for p in msg.parts if isinstance(p, TextPart))
            out.append({"role": "assistant", "content": text})
    return out


def ollama_native_model(
    model: str,
    client: Any,
    *,
    num_ctx: int = 8192,
    keep_alive: str = "30m",
    think: bool | None = False,
) -> FunctionModel:
    """`client` es un `ollama.Client` (o un doble de test con el mismo `chat(**kw)`).

    `think=False` se envía siempre (los modelos híbridos lo respetan; las variantes Thinking como
    `qwen3-vl:8b` lo ignoran y el backend lo detecta por `ThinkingPart`)."""

    think_supported = think is not None

    def chat(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        settings: dict = dict(info.model_settings or {})
        params = info.model_request_parameters
        options: dict[str, Any] = {"num_ctx": num_ctx}
        if settings.get("temperature") is not None:
            options["temperature"] = settings["temperature"]
        if settings.get("max_tokens") is not None:
            options["num_predict"] = settings["max_tokens"]
        kwargs = dict(
            model=model,
            messages=to_ollama_messages(messages, info.instructions),
            format=params.output_object.json_schema if params.output_object else None,
            options=options,
            keep_alive=keep_alive,
        )
        nonlocal think_supported
        if think_supported:
            try:
                resp = client.chat(think=think, **kwargs)
            except Exception as e:  # modelos sin soporte de razonamiento rechazan `think`
                if "does not support thinking" not in str(e).lower():
                    raise
                think_supported = False  # no repetir la petición fallida en cada bloque
                resp = client.chat(**kwargs)
        else:
            resp = client.chat(**kwargs)
        message = resp["message"] if isinstance(resp, dict) else resp.message
        get = message.get if isinstance(message, dict) else lambda k: getattr(message, k, None)
        parts: list = []
        if get("thinking"):
            parts.append(ThinkingPart(content=get("thinking")))
        parts.append(TextPart(content=get("content") or ""))
        rget = resp.get if isinstance(resp, dict) else lambda k: getattr(resp, k, None)
        usage = RequestUsage(
            input_tokens=rget("prompt_eval_count") or 0, output_tokens=rget("eval_count") or 0
        )
        return ModelResponse(parts=parts, usage=usage)

    return FunctionModel(chat, model_name=f"ollama:{model}")
