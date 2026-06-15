"""Provider dispatch: turn a ModelSpec + key pool into a generator callable, and
a judge name + model into a judge callable. The original `GENERATORS` lambda table
is replaced by registry-driven dispatch here."""

from ..keys import KeyPool
from ..registry import ModelSpec, FAMILY_PROVIDER
from .. import prompts
from . import openai_sdk, google_genai


def build_generator(spec: ModelSpec, pool: KeyPool, temperature: float, gen_prompt=None):
    """Return `word -> bytes|str` for this model, using its family's pool.

    `gen_prompt` (word -> prompt string) is the modality's language-aware prompt
    builder; defaults to the English text prompt for back-compat.
    """
    prov = FAMILY_PROVIDER[spec.family]
    base_url = prov["base_url"]
    text_prompt = gen_prompt or prompts.gen_prompt_text

    if spec.modality == "text":
        if prov["sdk"] == "openai_sdk":
            return lambda w: openai_sdk.generate_text(
                w, spec.api_id, pool, base_url, temperature, text_prompt(w))
        return lambda w: google_genai.generate_text(
            w, spec.api_id, pool, temperature, text_prompt(w))

    # image
    if prov["sdk"] == "google_genai":
        return lambda w: google_genai.generate_image(w, spec.api_id, pool)
    if spec.family == "openai":
        return lambda w: openai_sdk.generate_image_openai(w, spec.api_id, pool)
    if spec.family == "grok":
        return lambda w: openai_sdk.generate_image_grok(w, spec.api_id, pool, base_url)
    return lambda w: openai_sdk.generate_image_together(w, spec.api_id, pool, base_url)


def build_judge(judge_name: str, judge_model: str, pool: KeyPool):
    """Return `(prompt, schema, image=None) -> dict` for a judge."""
    if judge_name == "gpt":
        return lambda prompt, schema, image=None: openai_sdk.judge(
            judge_model, prompt, schema, pool, image=image)
    return lambda prompt, schema, image=None: google_genai.judge(
        judge_model, prompt, schema, pool, image=image)
