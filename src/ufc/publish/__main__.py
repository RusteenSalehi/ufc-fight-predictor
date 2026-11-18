"""Command line for the published artifacts' contract.

uv run python -m ufc.publish schema           write web/schema/artifacts.schema.json
uv run python -m ufc.publish schema --check   exit 1 if the committed file is out of date
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ufc.publish import jsonschema

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ufc.publish")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("schema", help="export the public artifacts' JSON Schema")
    p.add_argument("--out", type=Path, default=jsonschema.DEFAULT_OUT)
    p.add_argument(
        "--check", action="store_true", help="compare with the file instead of writing"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    expected = jsonschema.render(jsonschema.public_schema())
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else None
        if current != expected:
            log.error(
                "%s is out of date; run `uv run python -m ufc.publish schema`", args.out
            )
            return 1
        log.info("%s matches ufc.publish.schemas", args.out)
        return 0
    jsonschema.write(args.out)
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
