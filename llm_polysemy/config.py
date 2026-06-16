"""Modality configuration objects — the single place image-vs-text differences live.

The runner and entrypoints are fully generic over a `GenModality` /
`ClassifyModality`; selecting one of the four instances below switches output dirs,
file extensions, retry tokens, generate-returns-bytes-vs-str, and
judge-input-is-image-vs-text.

Paths use the current on-disk filenames; the data-file migration step updates only
these constants (resume logic is unaffected — see plan).
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import prompts

N_PER_WORD = 30
TEXT_TEMPERATURE = 1.0

# Per-generation-request timeout (seconds) so a hung API call can't squat on a
# key slot, and how many concurrent in-flight requests to allow PER KEY (the
# generation semaphore is len(pool) * this). Both env-overridable for tuning.
GEN_REQUEST_TIMEOUT = int(os.environ.get("GEN_REQUEST_TIMEOUT", "120"))
GEN_PER_KEY_CONCURRENCY = int(os.environ.get("GEN_PER_KEY_CONCURRENCY", "3"))

# Judges — MUST stay identical across modalities for cross-modality comparability.
JUDGES = {
    "gpt":    "gpt-5.4",
    "gemini": "gemini-3.5-flash",
}
# How many concurrent requests per judge key (judge calls are cheap completions).
PER_KEY_CONCURRENCY = 4


@dataclass(frozen=True)
class GenModality:
    name: str                       # "image" | "text"
    output_dir: Path
    results_file: Path
    ext: str                        # "png" | "txt"
    retry_token: str                # extra retryable error substring
    artifact_word: str              # "images" | "sentences" (summary line)
    save: Callable[[Path, object], None]
    temperature: float | None = None
    n_per_word: int = N_PER_WORD
    lang: str = "en"
    gen_prompt: Callable[[str], str] | None = None  # word -> prompt string


@dataclass(frozen=True)
class ClassifyModality:
    name: str                       # "image" | "text"
    output_dir: Path
    class_file: Path
    hist_file: Path
    ext: str
    load_artifact: Callable[[Path], object]      # read_bytes | read_text
    build_prompt: Callable[..., str]             # (meanings, word, artifact) -> str
    build_multiple_prompt: Callable[..., str]
    judge_input_kind: str           # "image" | "text"
    n_per_word: int = N_PER_WORD
    lang: str = "en"
    meanings_file: Path = Path("data/meanings.json")


# ── Instances ──────────────────────────────────────────────────────────────────

DATA = Path("data")

# Per-language file suffix. English keeps the original (unsuffixed) paths so the
# existing corpus/logs are untouched; a new language just gets a "_<lang>" suffix.
LANG_SUFFIX = {"en": "", "tr": "_tr", "fr": "_fr"}


def gen_modality(name: str, lang: str = "en") -> GenModality:
    suf = LANG_SUFFIX[lang]
    if name == "text-imagine":
        # Same text models, but prompted for the image the word evokes. English
        # only (control against the English image-sampling signal); no lang suffix.
        return GenModality(
            name="text-imagine",
            output_dir=DATA / "outputs_text_imagine",
            results_file=DATA / "results_text_imagine.json",
            ext="txt", retry_token="no text", artifact_word="descriptions",
            save=lambda path, data: path.write_text(data), temperature=TEXT_TEMPERATURE,
            gen_prompt=prompts.gen_prompt_imagine,
        )
    if name == "image":
        return GenModality(
            name="image", lang=lang,
            output_dir=DATA / f"outputs_image{suf}",
            results_file=DATA / f"results_image{suf}.json",
            ext="png", retry_token="no image", artifact_word="images",
            save=lambda path, data: path.write_bytes(data),
            gen_prompt=prompts.gen_prompt_image,
        )
    return GenModality(
        name="text", lang=lang,
        output_dir=DATA / f"outputs_text{suf}",
        results_file=DATA / f"results_text{suf}.json",
        ext="txt", retry_token="no text", artifact_word="sentences",
        save=lambda path, data: path.write_text(data), temperature=TEXT_TEMPERATURE,
        gen_prompt=lambda w, _l=lang: prompts.gen_prompt_text(w, _l),
    )


def cls_modality(name: str, lang: str = "en") -> ClassifyModality:
    suf = LANG_SUFFIX[lang]
    meanings_file = DATA / f"meanings{suf}.json"
    if name == "text-imagine":
        return ClassifyModality(
            name="text-imagine",
            output_dir=DATA / "outputs_text_imagine",
            class_file=DATA / "classifications_text_imagine.json",
            hist_file=DATA / "histograms_text_imagine.json",
            ext="txt", load_artifact=lambda p: p.read_text().strip(),
            build_prompt=lambda meanings, word, artifact: prompts.imagine_prompt(meanings, word, artifact),
            build_multiple_prompt=lambda meanings, word, artifact: prompts.imagine_multi(meanings, word, artifact),
            judge_input_kind="text", meanings_file=DATA / "meanings.json",
        )
    if name == "image":
        return ClassifyModality(
            name="image", lang=lang,
            output_dir=DATA / f"outputs_image{suf}",
            class_file=DATA / f"classifications_image{suf}.json",
            hist_file=DATA / f"histograms_image{suf}.json",
            ext="png", load_artifact=lambda p: p.read_bytes(),
            build_prompt=lambda meanings, word, artifact: prompts.image_prompt(meanings, word),
            build_multiple_prompt=lambda meanings, word, artifact: prompts.image_multi(meanings, word),
            judge_input_kind="image", meanings_file=meanings_file,
        )
    return ClassifyModality(
        name="text", lang=lang,
        output_dir=DATA / f"outputs_text{suf}",
        class_file=DATA / f"classifications_text{suf}.json",
        hist_file=DATA / f"histograms_text{suf}.json",
        ext="txt", load_artifact=lambda p: p.read_text().strip(),
        build_prompt=lambda meanings, word, artifact: prompts.text_prompt(meanings, word, artifact),
        build_multiple_prompt=lambda meanings, word, artifact: prompts.text_multi(meanings, word, artifact),
        judge_input_kind="text", meanings_file=meanings_file,
    )


def human_cls_modality(condition: str) -> ClassifyModality:
    """Classify human Prolific responses. condition: "meaning" | "image".

    Reuses the English senses (data/meanings.json — same 100-word panel) and the
    same judges, but with human-framed prompts. The two conditions are the human
    analogs of the LLM text ("meaning") and image ("image") modalities, so their
    histograms compare directly to histograms_text.json / histograms_image.json.
    """
    prompt = prompts.human_meaning_prompt if condition == "meaning" else prompts.human_image_prompt
    multi = prompts.human_meaning_multi if condition == "meaning" else prompts.human_image_multi
    return ClassifyModality(
        name=f"human_{condition}",
        output_dir=DATA / f"outputs_human_{condition}",
        class_file=DATA / f"classifications_human_{condition}.json",
        hist_file=DATA / f"histograms_human_{condition}.json",
        ext="txt", load_artifact=lambda p: p.read_text().strip(),
        build_prompt=lambda meanings, word, artifact, _p=prompt: _p(meanings, word, artifact),
        build_multiple_prompt=lambda meanings, word, artifact, _m=multi: _m(meanings, word, artifact),
        judge_input_kind="text", meanings_file=DATA / "meanings.json",
    )


# English instances kept under the original names + dicts for back-compat.
IMAGE_GEN = gen_modality("image")
TEXT_GEN = gen_modality("text")
IMAGE_CLS = cls_modality("image")
TEXT_CLS = cls_modality("text")

TEXT_IMAGINE_GEN = gen_modality("text-imagine")
TEXT_IMAGINE_CLS = cls_modality("text-imagine")

GEN = {"image": IMAGE_GEN, "text": TEXT_GEN, "text-imagine": TEXT_IMAGINE_GEN}
CLS = {"image": IMAGE_CLS, "text": TEXT_CLS, "text-imagine": TEXT_IMAGINE_CLS}
