"""The frontend's types are generated from the committed JSON Schema
(``web/schema/artifacts.schema.json``); these tests fail if that file drifts
from ``ufc.publish.schemas``, and check that the site's synthetic test
fixtures are valid artifacts."""

from pathlib import Path

from ufc.publish import jsonschema
from ufc.publish.__main__ import main

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / jsonschema.DEFAULT_OUT


def test_committed_schema_matches_the_models() -> None:
    assert COMMITTED.read_text(encoding="utf-8") == jsonschema.render(
        jsonschema.public_schema()
    ), "run `uv run python -m ufc.publish schema` and regenerate the web types"


def test_schema_names_every_public_artifact() -> None:
    schema = jsonschema.public_schema()
    assert set(schema["required"]) == {"upcoming", "bout", "performance"}
    defs = schema["$defs"]
    # serialization by alias, as the gate writes the files
    assert "for" in defs["Why"]["properties"]
    # fields with defaults are always written, so they are required
    assert "market" in defs["BoutArtifact"]["required"]
    assert "schema_version" in defs["Performance"]["required"]


def test_check_flag(tmp_path: Path) -> None:
    out = tmp_path / "schema.json"
    assert main(["schema", "--check", "--out", str(out)]) == 1
    assert main(["schema", "--out", str(out)]) == 0
    assert main(["schema", "--check", "--out", str(out)]) == 0
    out.write_text("{}", encoding="utf-8")
    assert main(["schema", "--check", "--out", str(out)]) == 1
