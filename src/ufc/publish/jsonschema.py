"""JSON Schema of the public artifacts, for the frontend's generated types (M7).

The site reads ``upcoming.json``, ``bouts/{bout_id}.json`` and
``performance.json``. Their pydantic models (``schemas.py``) are exported as one
JSON Schema document with shared definitions, in serialization mode and by
alias, which is how ``gate.py`` writes the files. ``web/`` turns the committed
schema into TypeScript with json-schema-to-typescript, and two tests keep the
chain in step: ``tests/test_publish_jsonschema.py`` (models to schema) and
``web/src/types/artifacts.test.ts`` (schema to TypeScript).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic.json_schema import GenerateJsonSchema, models_json_schema

from ufc.publish.schemas import SCHEMA_VERSION, BoutArtifact, Performance, Upcoming

PUBLIC_MODELS = (Upcoming, BoutArtifact, Performance)
DEFAULT_OUT = Path("web/schema/artifacts.schema.json")


class _NoFieldTitles(GenerateJsonSchema):
    """Field titles ("Favoured Name") would become stray named types in the
    generated TypeScript; model titles are kept."""

    def field_title_should_be_set(self, schema: Any) -> bool:
        return False


def public_schema() -> dict[str, Any]:
    """One document: the three public artifacts under ``$defs``, and a root
    object naming them so every definition is reachable."""
    _, schema = models_json_schema(
        [(m, "serialization") for m in PUBLIC_MODELS],
        by_alias=True,
        title="PublicArtifacts",
        description=(
            "The JSON files the site reads, generated from ufc.publish.schemas "
            f"(schema_version {SCHEMA_VERSION}). Do not edit; run "
            "`uv run python -m ufc.publish schema`."
        ),
        schema_generator=_NoFieldTitles,
    )
    names = {
        "upcoming": "Upcoming",
        "bout": "BoutArtifact",
        "performance": "Performance",
    }
    schema["type"] = "object"
    schema["additionalProperties"] = False
    schema["properties"] = {k: {"$ref": f"#/$defs/{v}"} for k, v in names.items()}
    schema["required"] = list(names)
    return schema


def render(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write(path: Path = DEFAULT_OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(public_schema()), encoding="utf-8")
    return path
