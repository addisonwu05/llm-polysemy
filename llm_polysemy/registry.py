"""Single source of truth for every model in the benchmark.

One `ModelSpec` per model_key, covering both the image-generation models and the
text models. Everything else (eval/classify dispatch, plot model lists, family
groupings, colors, generation-trend ordering) derives from this module — so adding
a model means adding ONE row here.

Conventions
-----------
- `family` is the lowercase provider/pool id ("openai", "grok", "gemini", "qwen",
  "flux"); it drives key-pool routing, the concurrency semaphore, and display
  grouping. `FAMILY_DISPLAY` maps it to a title-cased label for figures.
- `generation_index` orders models within a family for the generation-trend plot:
  mainline releases get integers (oldest=1), and side-branches (mini/nano/
  non-reasoning/size variants) get `.5`-style offsets so they render beside their
  sibling without displacing the mainline (mirrors the original image
  `openai_mini = 1.5` convention).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str                # registry key == on-disk model_key in outputs/histograms
    api_id: str             # provider API id
    family: str             # "openai" | "grok" | "gemini" | "qwen" | "flux"
    modality: str           # "image" | "text"
    label: str              # human display label (preserved from original plots)
    generation_index: float # ordering within family for the trend plot


# ── Family-level provider config (SDK + endpoint + key-pool prefix) ────────────
# `sdk`: which client to build. `key_prefix`: env-var prefix gathered into the
# family's key pool (kept byte-identical to the original scripts — note the two
# disjoint Together prefixes must NOT be collapsed).
FAMILY_PROVIDER = {
    "openai": {"sdk": "openai_sdk",   "base_url": None,                         "key_prefix": "openai_api_key"},
    "grok":   {"sdk": "openai_sdk",   "base_url": "https://api.x.ai/v1",        "key_prefix": "xai_api_key"},
    "gemini": {"sdk": "google_genai", "base_url": None,                         "key_prefix": "gemini_api_key"},
    "qwen":   {"sdk": "openai_sdk",   "base_url": "https://api.together.xyz/v1", "key_prefix": "together_api_key_qwen"},
    "flux":   {"sdk": "openai_sdk",   "base_url": "https://api.together.xyz/v1", "key_prefix": "together_api_key_bf"},
    # GLM (zai-org) served on Together. Draws from ALL Together keys (the broad
    # `together_api_key` prefix unions the _bf + _qwen pools) — fine because the
    # Qwen/FLUX runs are complete, so the shared pool is uncontended.
    "glm":    {"sdk": "openai_sdk",   "base_url": "https://api.together.xyz/v1", "key_prefix": "together_api_key"},
}

FAMILY_ORDER = ["openai", "grok", "gemini", "qwen", "flux", "glm"]
FAMILY_DISPLAY = {"openai": "OpenAI", "grok": "Grok", "gemini": "Gemini", "qwen": "Qwen", "flux": "FLUX", "glm": "GLM"}
FAMILY_COLOR = {
    "openai": "#DD8452", "grok": "#C44E52", "gemini": "#55A868",
    "qwen": "#8172B3", "flux": "#4C72B0", "glm": "#937860",
}

# Sense colors: listed senses cycle through PALETTE; the extra labels are fixed grays.
SENSE_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]
EXTRA_LABELS = ["multiple", "unclear", "other"]
EXTRA_COLORS = {"multiple": "#BBBBBB", "unclear": "#555555", "other": "#333333"}


# ── The registry ──────────────────────────────────────────────────────────────
# Ordered by FAMILY_ORDER, then by generation_index within each family.
_SPECS = [
    # ════ IMAGE ════════════════════════════════════════════════════════════════
    # OpenAI (gpt-image)
    ModelSpec("openai_1",    "gpt-image-1",                       "openai", "image", "GPT Image 1",      1),
    ModelSpec("openai_mini", "gpt-image-1-mini",                  "openai", "image", "GPT Image 1 Mini", 1.5),
    ModelSpec("openai_15",   "gpt-image-1.5",                     "openai", "image", "GPT Image 1.5",    2),
    ModelSpec("openai",      "gpt-image-2",                       "openai", "image", "GPT Image 2",      3),
    # Grok (grok-imagine-image)
    ModelSpec("grok_base",   "grok-imagine-image",                "grok",   "image", "Grok Base",        1),
    ModelSpec("grok_pro",    "grok-imagine-image-pro",            "grok",   "image", "Grok Pro",         2),
    ModelSpec("grok",        "grok-imagine-image-quality",        "grok",   "image", "Grok Quality",     3),
    # Gemini (image)
    ModelSpec("gemini_25",   "gemini-2.5-flash-image",            "gemini", "image", "Gemini 2.5 Flash", 1),
    ModelSpec("gemini",      "gemini-3.1-flash-image-preview",    "gemini", "image", "Gemini 3.1 Flash", 2),
    ModelSpec("gemini_pro",  "gemini-3-pro-image-preview",        "gemini", "image", "Gemini 3 Pro",     3),
    # Qwen (image)
    ModelSpec("qwen_1",      "Qwen/Qwen-Image",                   "qwen",   "image", "Qwen-Image",          1),
    ModelSpec("qwen_20",     "Qwen/Qwen-Image-2.0",               "qwen",   "image", "Qwen-Image 2.0",         2),
    ModelSpec("qwen",        "Qwen/Qwen-Image-2.0-Pro",           "qwen",   "image", "Qwen-Image 2.0 Pro",     3),
    # FLUX (Black Forest Labs)
    ModelSpec("flux_1pro",   "black-forest-labs/FLUX.1.1-pro",    "flux",   "image", "FLUX.1.1 Pro",     1),
    ModelSpec("flux_flex",   "black-forest-labs/FLUX.2-flex",     "flux",   "image", "FLUX.2 Flex",      2),
    ModelSpec("flux",        "black-forest-labs/FLUX.2-pro",      "flux",   "image", "FLUX.2 Pro",       3),
    ModelSpec("flux_max",    "black-forest-labs/FLUX.2-max",      "flux",   "image", "FLUX.2 Max",       4),

    # ════ TEXT ═════════════════════════════════════════════════════════════════
    # OpenAI (GPT) — mainline 3.5→4o→5→5.5; 5.4 nano/mini are side-branches.
    ModelSpec("openai_35",     "gpt-3.5-turbo",  "openai", "text", "GPT-3.5 Turbo", 1),
    ModelSpec("openai_4o",     "gpt-4o",         "openai", "text", "GPT-4o",        2),
    ModelSpec("openai_5",      "gpt-5",          "openai", "text", "GPT-5",         3),
    ModelSpec("openai_54nano", "gpt-5.4-nano",   "openai", "text", "GPT-5.4 Nano",  3.4),
    ModelSpec("openai_54mini", "gpt-5.4-mini",   "openai", "text", "GPT-5.4 Mini",  3.5),
    ModelSpec("openai_55",     "gpt-5.5",        "openai", "text", "GPT-5.5",       4),
    # Grok — 4.20 checkpoint (reasoning mainline, non-reasoning side) → 4.3.
    ModelSpec("grok_420",    "grok-4.20-0309-reasoning",     "grok", "text", "Grok 4.20",    1),
    ModelSpec("grok_420_nr", "grok-4.20-0309-non-reasoning", "grok", "text", "Grok 4.20 NR", 1.5),
    ModelSpec("grok_43",     "grok-4.3",                     "grok", "text", "Grok 4.3",     2),
    # Gemini (text) — 2.5-flash → 3.5-flash; 3.1-pro is a separate tier (side).
    ModelSpec("gemini_25flash", "gemini-2.5-flash",        "gemini", "text", "Gemini 2.5 Flash", 1),
    ModelSpec("gemini_35flash", "gemini-3.5-flash",        "gemini", "text", "Gemini 3.5 Flash", 2),
    ModelSpec("gemini_31pro",   "gemini-3.1-pro-preview",  "gemini", "text", "Gemini 3.1 Pro",   2.5),
    # Qwen (text) — 2.5-7B → 3.5-9B; 3.5-397B is the large side-branch.
    ModelSpec("qwen_25_7b",  "Qwen/Qwen2.5-7B-Instruct-Turbo", "qwen", "text", "Qwen2.5 7B",   1),
    ModelSpec("qwen_35_9b",  "Qwen/Qwen3.5-9B",                "qwen", "text", "Qwen3.5 9B",    2),
    ModelSpec("qwen_35_397b","Qwen/Qwen3.5-397B-A17B",         "qwen", "text", "Qwen3.5 397B",  2.5),
    # GLM (zai-org, via Together) — text only. 5.1 → 5.2.
    # GLM-4.7 is NOT serverless on Together (needs a dedicated endpoint); disabled
    # until one is created. Re-enable by uncommenting once the endpoint is live.
    # ModelSpec("glm_47",    "zai-org/GLM-4.7",                "glm",  "text", "GLM-4.7",       1),
    ModelSpec("glm_51",      "zai-org/GLM-5.1",                "glm",  "text", "GLM-5.1",       2),
    ModelSpec("glm_52",      "zai-org/GLM-5.2",                "glm",  "text", "GLM-5.2",       3),
]

MODELS: dict[str, ModelSpec] = {s.key: s for s in _SPECS}


# ── Public API ─────────────────────────────────────────────────────────────────

def spec(key: str) -> ModelSpec:
    return MODELS[key]


def models_for(modality: str) -> list[str]:
    """Ordered model keys for a modality (FAMILY_ORDER, then generation_index)."""
    keys = [k for k, s in MODELS.items() if s.modality == modality]
    return sorted(keys, key=lambda k: (FAMILY_ORDER.index(MODELS[k].family),
                                       MODELS[k].generation_index))


def family_of(key: str) -> str:
    return MODELS[key].family


def display_label(key: str) -> str:
    return MODELS[key].label


def family_label(family: str) -> str:
    return FAMILY_DISPLAY[family]


def family_color(family: str) -> str:
    return FAMILY_COLOR[family]


def families(modality: str) -> list[str]:
    """Families present in a modality, in FAMILY_ORDER."""
    present = {MODELS[k].family for k in models_for(modality)}
    return [f for f in FAMILY_ORDER if f in present]


def family_groups(modality: str) -> dict[str, list[str]]:
    """family -> ordered model keys (by generation_index) for that modality."""
    out: dict[str, list[str]] = {f: [] for f in families(modality)}
    for k in models_for(modality):  # already family/gen sorted
        out[MODELS[k].family].append(k)
    return out


def generation_order(modality: str) -> dict[str, list[tuple[str, float, str]]]:
    """family -> [(key, generation_index, label)] oldest→newest. Drives the
    temporal / generation-trend plots."""
    return {
        f: [(k, MODELS[k].generation_index, MODELS[k].label) for k in keys]
        for f, keys in family_groups(modality).items()
    }


def sense_colors(senses: list[str]) -> dict[str, str]:
    """Map listed senses to the palette (cycling) plus the fixed extra-label grays."""
    colors = {s: SENSE_PALETTE[i % len(SENSE_PALETTE)] for i, s in enumerate(senses)}
    colors.update(EXTRA_COLORS)
    return colors


def provider(family: str) -> dict:
    return FAMILY_PROVIDER[family]
