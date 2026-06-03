"""JSON-schema builders for the two judges. Identical across modalities — the
schemas depend only on the word's label/sense sets, never on image-vs-text.

Each judge call uses one of two shapes:
    single   -> {"label":  <one enum value>}
    multiple -> {"senses": [<two or more sense keys>]}
"""


def single_schema_gpt(labels: list[str]) -> dict:
    return {"type": "object", "additionalProperties": False,
            "properties": {"label": {"type": "string", "enum": labels}},
            "required": ["label"]}


def multi_schema_gpt(senses: list[str]) -> dict:
    return {"type": "object", "additionalProperties": False,
            "properties": {"senses": {"type": "array", "minItems": 2,
                "items": {"type": "string", "enum": senses}}},
            "required": ["senses"]}


def single_schema_gemini(labels: list[str]):
    from google.genai import types
    return types.Schema(type=types.Type.OBJECT, required=["label"],
        properties={"label": types.Schema(type=types.Type.STRING, enum=labels)})


def multi_schema_gemini(senses: list[str]):
    from google.genai import types
    return types.Schema(type=types.Type.OBJECT, required=["senses"],
        properties={"senses": types.Schema(type=types.Type.ARRAY, min_items=2,
            items=types.Schema(type=types.Type.STRING, enum=senses))})
