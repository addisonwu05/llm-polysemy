"""OpenAI-SDK-backed providers (OpenAI, Grok @ x.ai, Qwen+FLUX @ together.ai).

Generation and judging bodies are ported verbatim from the original eval.py /
classify.py so behavior (and the per-family quirks) is preserved exactly.
"""

import base64
import json
import urllib.request

from ..keys import KeyPool
from ..config import GEN_REQUEST_TIMEOUT


def _client(api_key: str, base_url: str | None = None):
    from openai import OpenAI
    # timeout caps hung requests; max_retries=0 because our runner does retries
    # (the SDK's default 2 internal retries just compound a stall).
    return OpenAI(api_key=api_key, timeout=GEN_REQUEST_TIMEOUT, max_retries=0,
                  **({"base_url": base_url} if base_url else {}))


# ── Text generation (OpenAI / Grok / Qwen share one chat path) ─────────────────

def generate_text(word: str, api_id: str, pool: KeyPool, base_url: str | None,
                  temperature: float, prompt: str) -> str:
    client = _client(pool.next(), base_url)
    resp = client.chat.completions.create(
        model=api_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise ValueError(f"no text returned for '{word}'")
    return text


# ── Image generation (per-family quirks preserved) ─────────────────────────────

def generate_image_openai(word: str, api_id: str, pool: KeyPool) -> bytes:
    client = _client(pool.next())
    resp = client.images.generate(model=api_id, prompt=word, n=1, size="1024x1024")
    item = resp.data[0]
    if getattr(item, "b64_json", None):
        return base64.b64decode(item.b64_json)
    with urllib.request.urlopen(item.url) as r:
        return r.read()


def generate_image_grok(word: str, api_id: str, pool: KeyPool, base_url: str) -> bytes:
    client = _client(pool.next(), base_url)
    resp = client.images.generate(model=api_id, prompt=word, n=1, response_format="b64_json")
    return base64.b64decode(resp.data[0].b64_json)


def generate_image_together(word: str, api_id: str, pool: KeyPool, base_url: str) -> bytes:
    api_key = pool.next()
    client = _client(api_key, base_url)
    resp = client.images.generate(model=api_id, prompt=word, n=1, response_format="b64_json")
    item = resp.data[0]
    if getattr(item, "b64_json", None):
        return base64.b64decode(item.b64_json)
    req = urllib.request.Request(item.url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


# ── Judging (GPT judge) ────────────────────────────────────────────────────────

def judge(api_id: str, prompt: str, schema: dict, pool: KeyPool,
          *, image: bytes | None = None) -> dict:
    client = _client(pool.next())
    if image is not None:
        b64 = base64.b64encode(image).decode()
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
    else:
        content = prompt
    resp = client.chat.completions.create(
        model=api_id,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "out", "strict": True, "schema": schema}},
    )
    return json.loads(resp.choices[0].message.content)
