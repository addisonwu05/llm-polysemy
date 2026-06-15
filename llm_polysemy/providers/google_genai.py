"""Google GenAI provider (Gemini), for both generation and judging.

Bodies ported verbatim from the original eval*.py / classify*.py.
"""

import json

from ..keys import KeyPool
from ..config import GEN_REQUEST_TIMEOUT


def generate_text(word: str, api_id: str, pool: KeyPool, temperature: float, prompt: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=pool.next(),
                          http_options=types.HttpOptions(timeout=GEN_REQUEST_TIMEOUT * 1000))
    resp = client.models.generate_content(
        model=api_id,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    text = (resp.text or "").strip()
    if not text:
        raise ValueError(f"no text returned for '{word}'")
    return text


def generate_image(word: str, api_id: str, pool: KeyPool) -> bytes:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=pool.next(),
                          http_options=types.HttpOptions(timeout=GEN_REQUEST_TIMEOUT * 1000))
    resp = client.models.generate_content(
        model=api_id,
        contents=word,
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )
    for part in resp.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data
    raise ValueError(f"Gemini returned no image for '{word}'")


def judge(api_id: str, prompt: str, schema, pool: KeyPool, *, image: bytes | None = None) -> dict:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=pool.next(),
                          http_options=types.HttpOptions(timeout=GEN_REQUEST_TIMEOUT * 1000))
    if image is not None:
        contents = [types.Part.from_bytes(data=image, mime_type="image/png"), prompt]
    else:
        contents = prompt
    resp = client.models.generate_content(
        model=api_id,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    return json.loads(resp.text)
