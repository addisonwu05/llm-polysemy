"""Modality configuration objects — the single place image-vs-text differences live.

The runner and entrypoints are fully generic over a `GenModality` /
`ClassifyModality`; selecting one of the four instances below switches output dirs,
file extensions, retry tokens, generate-returns-bytes-vs-str, and
judge-input-is-image-vs-text.

Paths use the current on-disk filenames; the data-file migration step updates only
these constants (resume logic is unaffected — see plan).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import prompts

N_PER_WORD = 30
TEXT_TEMPERATURE = 1.0

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


# ── Instances ──────────────────────────────────────────────────────────────────

IMAGE_GEN = GenModality(
    name="image", output_dir=Path("outputs"), results_file=Path("results.json"),
    ext="png", retry_token="no image", artifact_word="images",
    save=lambda path, data: path.write_bytes(data),
)

TEXT_GEN = GenModality(
    name="text", output_dir=Path("outputs_text"), results_file=Path("results_text.json"),
    ext="txt", retry_token="no text", artifact_word="sentences",
    save=lambda path, data: path.write_text(data), temperature=TEXT_TEMPERATURE,
)

IMAGE_CLS = ClassifyModality(
    name="image", output_dir=Path("outputs"),
    class_file=Path("classifications.json"), hist_file=Path("histograms.json"),
    ext="png", load_artifact=lambda p: p.read_bytes(),
    build_prompt=lambda meanings, word, artifact: prompts.image_prompt(meanings, word),
    build_multiple_prompt=lambda meanings, word, artifact: prompts.image_multi(meanings, word),
    judge_input_kind="image",
)

TEXT_CLS = ClassifyModality(
    name="text", output_dir=Path("outputs_text"),
    class_file=Path("classifications_text.json"), hist_file=Path("histograms_text.json"),
    ext="txt", load_artifact=lambda p: p.read_text().strip(),
    build_prompt=lambda meanings, word, artifact: prompts.text_prompt(meanings, word, artifact),
    build_multiple_prompt=lambda meanings, word, artifact: prompts.text_multi(meanings, word, artifact),
    judge_input_kind="text",
)

GEN = {"image": IMAGE_GEN, "text": TEXT_GEN}
CLS = {"image": IMAGE_CLS, "text": TEXT_CLS}
