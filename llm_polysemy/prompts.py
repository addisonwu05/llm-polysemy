"""Prompt builders for generation and for the sense-classification judges.

All judge prompts are reproduced verbatim from the original classify.py /
classify_text.py so existing classifications stay comparable.
"""

# ── Generation prompts ─────────────────────────────────────────────────────────

# One per language; the text models are instructed in the word's own language so
# they answer in it. The image prompt is the bare word, which is language-agnostic.
TEXT_PROMPT_TEMPLATES = {
    "en": (
        "Use the following word in a single sentence: {word}\n"
        "Reply with only the sentence and nothing else."
    ),
    "tr": (
        "Aşağıdaki kelimeyi tek bir cümlede kullanın: {word}\n"
        "Yalnızca cümleyi yazın, başka hiçbir şey yazmayın."
    ),
    "fr": (
        "Utilisez le mot suivant dans une seule phrase : {word}\n"
        "Répondez uniquement avec la phrase et rien d'autre."
    ),
}
# Back-compat alias (English) for any importer of the old constant.
TEXT_PROMPT_TEMPLATE = TEXT_PROMPT_TEMPLATES["en"]

# "text-imagine" condition: same text models, but asked what image the word evokes
# (instead of to use it in a sentence). Wording mirrors the human Prolific `image`
# survey verbatim so LLM-vs-human is a clean, framing-matched comparison; only the
# `Reply with only…` tail is added (an LLM preambles, a person typing a box doesn't).
# English only by design — it is a control against the English image-sampling signal.
IMAGINE_PROMPT_TEMPLATES = {
    "en": (
        'What image comes to mind when you think of the word "{word}"? Describe it briefly.\n'
        "Reply with only the description and nothing else."
    ),
}


def gen_prompt_image(word: str) -> str:
    return word  # the prompt sent to an image model is the bare word


def gen_prompt_text(word: str, lang: str = "en") -> str:
    return TEXT_PROMPT_TEMPLATES[lang].format(word=word)


def gen_prompt_imagine(word: str, lang: str = "en") -> str:
    return IMAGINE_PROMPT_TEMPLATES[lang].format(word=word)


# ── Judge prompts ──────────────────────────────────────────────────────────────

def _sense_lines(meanings: dict, word: str) -> str:
    return "\n".join(f'- "{s["sense"]}": {s["meaning"]}' for s in meanings[word])


def image_prompt(meanings: dict, word: str) -> str:
    lines = [f'- "{s["sense"]}": {s["meaning"]}' for s in meanings[word]]
    lines.append('- "multiple": the image clearly shows two or more of the above meanings together')
    lines.append('- "unclear": you cannot tell, or it matches none of the above')
    lines.append('- "other": it clearly shows a different, specific meaning of the word not listed above')
    opts = "\n".join(lines)
    return (
        f'A text-to-image model was given only the single word "{word}" as its prompt, '
        f'with no other context. Look at the image and decide which meaning of "{word}" '
        f'it depicts.\n\nChoose exactly one of these labels:\n{opts}\n\n'
        f'Respond with only the label.'
    )


def image_multi(meanings: dict, word: str) -> str:
    opts = _sense_lines(meanings, word)
    return (
        f'This image (generated from the single word "{word}") clearly shows two or '
        f'more distinct meanings of "{word}" together. Of the meanings below, list '
        f'every one that is actually present in the image:\n{opts}\n\n'
        f'Return the senses that apply (at least two).'
    )


def text_prompt(meanings: dict, word: str, sentence: str) -> str:
    lines = [f'- "{s["sense"]}": {s["meaning"]}' for s in meanings[word]]
    lines.append('- "multiple": the sentence clearly uses two or more of the above meanings together')
    lines.append('- "unclear": you cannot tell, or it matches none of the above')
    lines.append('- "other": it clearly uses a different, specific meaning of the word not listed above')
    opts = "\n".join(lines)
    return (
        f'A language model was given only the single word "{word}" as its prompt, with no '
        f'other context, and asked to use it in one sentence. Here is the sentence it produced:\n\n'
        f'"{sentence}"\n\n'
        f'Decide which meaning of "{word}" the sentence uses.\n\n'
        f'Choose exactly one of these labels:\n{opts}\n\n'
        f'Respond with only the label.'
    )


def text_multi(meanings: dict, word: str, sentence: str) -> str:
    opts = _sense_lines(meanings, word)
    return (
        f'This sentence (generated from the single word "{word}") clearly uses two or '
        f'more distinct meanings of "{word}" together:\n\n"{sentence}"\n\n'
        f'Of the meanings below, list every one that is actually present in the sentence:\n{opts}\n\n'
        f'Return the senses that apply (at least two).'
    )


# "text-imagine" judge prompts: an LLM was asked what image the word evokes and
# described it. Deliberately mirrors human_image_prompt/_multi (swapping "A person"
# -> "A language model") so the label set + framing match the human `image`
# condition, keeping histograms_text_imagine directly comparable to it.

def imagine_prompt(meanings: dict, word: str, description: str) -> str:
    opts = _human_label_lines(meanings, word, "described image")
    return (
        f'A language model was given only the single word "{word}", with no other context, '
        f'and asked to briefly describe the image that comes to mind. Here is its '
        f'description:\n\n"{description}"\n\n'
        f'Decide which meaning of "{word}" the image it describes corresponds to.\n\n'
        f'Choose exactly one of these labels:\n{opts}\n\nRespond with only the label.'
    )


def imagine_multi(meanings: dict, word: str, description: str) -> str:
    opts = _sense_lines(meanings, word)
    return (
        f'This description (of the image a language model pictured for the single word "{word}") '
        f'clearly evokes two or more distinct meanings of "{word}" together:\n\n"{description}"\n\n'
        f'Of the meanings below, list every one that is actually present:\n{opts}\n\n'
        f'Return the senses that apply (at least two).'
    )


# ── Human-response judge prompts ────────────────────────────────────────────────
# Same labels/senses as the LLM judges (so distributions are comparable), but the
# framing matches how the human data was collected: a person, and two conditions
# — "meaning" (use the word in a sentence) and "image" (describe the image that
# comes to mind). Kept separate from the model prompts so neither biases the other.

def _human_label_lines(meanings: dict, word: str, subject: str) -> str:
    lines = [f'- "{s["sense"]}": {s["meaning"]}' for s in meanings[word]]
    lines.append(f'- "multiple": the {subject} clearly uses two or more of the above meanings together')
    lines.append('- "unclear": you cannot tell, or it matches none of the above')
    lines.append('- "other": it clearly uses a different, specific meaning of the word not listed above')
    return "\n".join(lines)


def human_meaning_prompt(meanings: dict, word: str, sentence: str) -> str:
    opts = _human_label_lines(meanings, word, "sentence")
    return (
        f'A person was shown only the single word "{word}", with no other context, and '
        f'asked to use it in one sentence. Here is the sentence they wrote:\n\n"{sentence}"\n\n'
        f'Decide which meaning of "{word}" the sentence uses.\n\n'
        f'Choose exactly one of these labels:\n{opts}\n\nRespond with only the label.'
    )


def human_meaning_multi(meanings: dict, word: str, sentence: str) -> str:
    opts = _sense_lines(meanings, word)
    return (
        f'This sentence (written by a person shown only the single word "{word}") clearly '
        f'uses two or more distinct meanings of "{word}" together:\n\n"{sentence}"\n\n'
        f'Of the meanings below, list every one that is actually present in the sentence:\n{opts}\n\n'
        f'Return the senses that apply (at least two).'
    )


def human_image_prompt(meanings: dict, word: str, description: str) -> str:
    opts = _human_label_lines(meanings, word, "described image")
    return (
        f'A person was shown only the single word "{word}", with no other context, and '
        f'asked to briefly describe the first image that comes to mind. Here is their '
        f'description:\n\n"{description}"\n\n'
        f'Decide which meaning of "{word}" the image they describe corresponds to.\n\n'
        f'Choose exactly one of these labels:\n{opts}\n\nRespond with only the label.'
    )


def human_image_multi(meanings: dict, word: str, description: str) -> str:
    opts = _sense_lines(meanings, word)
    return (
        f'This description (of the image a person pictured for the single word "{word}") '
        f'clearly evokes two or more distinct meanings of "{word}" together:\n\n"{description}"\n\n'
        f'Of the meanings below, list every one that is actually present:\n{opts}\n\n'
        f'Return the senses that apply (at least two).'
    )
