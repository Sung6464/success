"""Thin wrappers around OpenAI SDK and Azure OpenAI SDK."""
from __future__ import annotations

import base64
import time
from pathlib import Path
from openai import AzureOpenAI, OpenAI

from . import config

# Clients
_chat_client: OpenAI | AzureOpenAI | None = None
_embed_client: AzureOpenAI | OpenAI | None = None


def chat_client() -> OpenAI | AzureOpenAI:
    global _chat_client
    if _chat_client is None:
        if config.AZURE_OPENAI_API_KEY and config.AZURE_OPENAI_ENDPOINT:
            _chat_client = AzureOpenAI(
                api_key=config.AZURE_OPENAI_API_KEY,
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                api_version=config.AZURE_OPENAI_API_VERSION,
            )
        else:
            _chat_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _chat_client


def embed_client() -> AzureOpenAI | OpenAI:
    global _embed_client
    if _embed_client is None:
        if config.AZURE_OPENAI_EMBEDDING_API_KEY and config.AZURE_OPENAI_EMBEDDING_ENDPOINT:
            _embed_client = AzureOpenAI(
                api_key=config.AZURE_OPENAI_EMBEDDING_API_KEY,
                azure_endpoint=config.AZURE_OPENAI_EMBEDDING_ENDPOINT,
                api_version=config.AZURE_OPENAI_EMBEDDING_API_VERSION,
            )
        elif config.AZURE_OPENAI_API_KEY and config.AZURE_OPENAI_ENDPOINT:
            _embed_client = AzureOpenAI(
                api_key=config.AZURE_OPENAI_API_KEY,
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
                api_version=config.AZURE_OPENAI_API_VERSION,
            )
        else:
            _embed_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _embed_client


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed a list of texts in batches."""
    out: list[list[float]] = []
    batch = 32
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        for attempt in range(5):
            try:
                model_name = config.AZURE_EMBED_DEPLOYMENT or config.EMBED_MODEL
                resp = embed_client().embeddings.create(
                    model=model_name,
                    input=chunk,
                )
                out.extend([e.embedding for e in resp.data])
                break
            except Exception as e:
                if attempt == 4:
                    raise e
                time.sleep(2 ** attempt)
    return out


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def generate_text(prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = chat_client().chat.completions.create(
        model=config.GEN_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def generate_multimodal(
    prompt: str, image_paths: list[Path], system: str | None = None, temperature: float = 0.2
) -> str:
    """Send text + images to GPT and return the grounded answer."""
    content: list[dict] = [{"type": "text", "text": prompt}]

    for p in image_paths:
        p = Path(p)
        if not p.exists():
            continue

        ext = p.suffix.lower().lstrip(".")
        if ext in ("jpg", "jpeg"):
            mime = "image/jpeg"
            try:
                b64_bytes = p.read_bytes()
            except Exception:
                continue
        elif ext == "gif":
            mime = "image/gif"
            try:
                b64_bytes = p.read_bytes()
            except Exception:
                continue
        elif ext == "webp":
            mime = "image/webp"
            try:
                b64_bytes = p.read_bytes()
            except Exception:
                continue
        elif ext == "png":
            mime = "image/png"
            try:
                b64_bytes = p.read_bytes()
            except Exception:
                continue
        else:
            mime = "image/png"
            try:
                from PIL import Image
                import io
                with Image.open(p) as img:
                    if img.mode not in ("RGB", "RGBA"):
                        img = img.convert("RGBA")
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    b64_bytes = buf.getvalue()
            except Exception as e:
                print(f"Warning: Skipping image {p.name} because conversion to PNG failed: {e}")
                continue

        try:
            b64 = base64.b64encode(b64_bytes).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        except Exception:
            continue

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    resp = chat_client().chat.completions.create(
        model=config.GEN_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()
