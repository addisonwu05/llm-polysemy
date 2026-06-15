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


def gen_prompt_image(word: str) -> str:
    return word  # the prompt sent to an image model is the bare word


def gen_prompt_text(word: str, lang: str = "en") -> str:
    return TEXT_PROMPT_TEMPLATES[lang].format(word=word)


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
