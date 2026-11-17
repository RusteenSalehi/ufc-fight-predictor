"""Just enough wikitext parsing for UFC cards: templates, links, sections.

Pure functions on strings. Wikipedia text is untrusted input (D16): nothing
here evaluates it, and anything that does not match an expected shape is
returned as "not found" rather than guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_PAIR = re.compile(r"<ref\b[^>/]*>.*?</ref\s*>", re.DOTALL | re.IGNORECASE)
_REF_SINGLE = re.compile(r"<ref\b[^>]*/>", re.IGNORECASE)
_TAGS = re.compile(r"</?(?:small|span|br|sup|nowiki|abbr)\b[^>]*>", re.IGNORECASE)
_HEADING = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.MULTILINE)


def strip_noise(text: str) -> str:
    """Remove comments and references, which may contain any markup."""
    return _REF_SINGLE.sub("", _REF_PAIR.sub("", _COMMENT.sub("", text)))


@dataclass(frozen=True, slots=True)
class Template:
    """A template call: positional parameters in order, named ones by name.
    Values are raw wikitext, stripped of surrounding whitespace."""

    name: str
    positional: tuple[str, ...]
    named: dict[str, str]


def _split_top(body: str) -> list[str]:
    """Split on '|' that is not inside a nested template or link."""
    parts, depth_t, depth_l, start, i = [], 0, 0, 0, 0
    while i < len(body):
        two = body[i : i + 2]
        if two == "{{":
            depth_t += 1
            i += 2
            continue
        if two == "}}" and depth_t:
            depth_t -= 1
            i += 2
            continue
        if two == "[[":
            depth_l += 1
            i += 2
            continue
        if two == "]]" and depth_l:
            depth_l -= 1
            i += 2
            continue
        if body[i] == "|" and depth_t == 0 and depth_l == 0:
            parts.append(body[start:i])
            start = i + 1
        i += 1
    parts.append(body[start:])
    return parts


def _match_end(text: str, start: int) -> int | None:
    """Index just after the '}}' closing the template opened at ``start``."""
    depth, i = 0, start
    while i < len(text) - 1:
        two = text[i : i + 2]
        if two == "{{":
            depth += 1
            i += 2
        elif two == "}}":
            depth -= 1
            i += 2
            if depth == 0:
                return i
        else:
            i += 1
    return None


def _norm_name(name: str) -> str:
    return " ".join(name.replace("_", " ").split()).casefold()


def templates(text: str, name: str) -> list[Template]:
    """Every call of template ``name`` (case and underscores ignored), in
    order. An unclosed call ends the search."""
    wanted = _norm_name(name)
    found: list[Template] = []
    i = 0
    while True:
        i = text.find("{{", i)
        if i < 0:
            return found
        end = _match_end(text, i)
        if end is None:
            return found
        parts = _split_top(text[i + 2 : end - 2])
        if _norm_name(parts[0]) == wanted:
            positional, named = [], {}
            for part in parts[1:]:
                key, eq, value = part.partition("=")
                if eq and re.fullmatch(r"\s*[\w ]+\s*", key) and "[[" not in key:
                    named[key.strip()] = value.strip()
                else:
                    positional.append(part.strip())
            found.append(Template(parts[0].strip(), tuple(positional), named))
            i = end
        else:
            i += 2


def remove_templates(text: str) -> str:
    """The text with every template call removed (outermost first)."""
    out, i = [], 0
    while True:
        j = text.find("{{", i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        end = _match_end(text, j)
        if end is None:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:j])
        i = end


_LINK = re.compile(r"\[\[([^\[\]|]+)(?:\|([^\[\]]*))?\]\]")


def first_link(text: str) -> tuple[str, str] | None:
    """(target page, displayed text) of the first wikilink in ``text``."""
    match = _LINK.search(text)
    if match is None:
        return None
    target = match.group(1).strip()
    display = (match.group(2) or target).strip()
    return target, display


def plain(text: str) -> str:
    """Displayed text: links reduced to their label, templates and tags removed."""
    text = remove_templates(strip_noise(text))
    text = _LINK.sub(lambda m: m.group(2) or m.group(1), text)
    text = _TAGS.sub("", text).replace("'''", "").replace("''", "")
    return " ".join(text.split())


def section(text: str, heading: str) -> str | None:
    """The body of the first section whose heading is ``heading`` (case
    ignored), up to the next heading of the same or a higher level."""
    headings = list(_HEADING.finditer(text))
    for index, match in enumerate(headings):
        if match.group(2).casefold() != heading.casefold():
            continue
        level = len(match.group(1))
        end = len(text)
        for later in headings[index + 1 :]:
            if len(later.group(1)) <= level:
                end = later.start()
                break
        return text[match.end() : end]
    return None


def table_rows(text: str) -> list[str]:
    """Rows of the first wikitable in ``text``, as raw text between '|-'
    lines. The header row (before the first '|-') is dropped."""
    start = text.find("{|")
    if start < 0:
        return []
    end = text.find("\n|}", start)
    body = text[start : end if end >= 0 else len(text)]
    rows = re.split(r"^\|-.*$", body, flags=re.MULTILINE)
    return [row for row in rows[1:] if row.strip()]
