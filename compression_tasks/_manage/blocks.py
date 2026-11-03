"""Edit one key inside a YAML file without reserialising the file.

`contract.yaml` carries comments and hand-chosen layout,
and a `yaml.safe_load` / `safe_dump` round trip loses both. So a parameter is
written by line: a block is addressed by its key, its exact indentation and
its parent key, only the lines inside it are replaced, and the file is parsed
afterwards (`write_checked`) to confirm the write landed and still loads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def write_checked(path, text: str) -> None:
    """Write `text`, but never leave a structured file unparseable.

    Surgical line editing is what keeps a file's comments and layout, and
    the cost of it is that a bad edit produces a file that still looks like
    YAML: a value that cuts a quoted scalar in half parses as an error far
    below the damage, on the next read, while the write itself reported
    nothing.

    So the check happens BEFORE the bytes land. A `.yaml` that does not
    parse, or a `.json` that does not, is refused with the path and the
    parser's own message, and the file on disk is untouched.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml
        try:
            yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"refusing to write {path}: the result is not valid YAML, so "
                f"the file on disk is unchanged. {exc}") from None
    elif suffix == ".json":
        import json
        try:
            json.loads(text)
        except ValueError as exc:
            raise ValueError(
                f"refusing to write {path}: the result is not valid JSON, so "
                f"the file on disk is unchanged. {exc}") from None
    path.write_text(text, "utf-8")


@dataclass(frozen=True)
class Block:
    """A mapping key and the lines that belong to it.

    `start` is the index of the `key:` line itself; `body` spans the lines
    after it that are indented deeper (a nested mapping, or a `-` list whose
    dashes sit at the key's own indentation, as PyYAML permits).
    """

    key: str
    indent: int
    start: int
    body_start: int
    body_end: int          # exclusive

    @property
    def body(self) -> slice:
        return slice(self.body_start, self.body_end)


def _indent_of(line: str) -> int | None:
    """Indentation of a line, or None if it carries no content."""
    if not line.strip() or line.lstrip().startswith("#"):
        return None
    return len(line) - len(line.lstrip(" "))


def find(lines: list[str], key: str, *, indent: int,
         under: str | None = None) -> Block | None:
    """The block for `key` at exactly `indent`, optionally inside `under`.

    `under` is a key at a shallower indentation that must be the nearest such
    ancestor -- this is what separates `reference.label` from a key of the
    same name nested under another parent.
    """
    want = " " * indent + key + ":"
    for i, line in enumerate(lines):
        if not (line.startswith(want)
                and line[len(want):].strip() in ("", "|", ">", "|-", ">-")):
            continue
        if under is not None and _nearest_ancestor(lines, i, indent) != under:
            continue
        body_start = i + 1
        j = body_start
        while j < len(lines):
            got = _indent_of(lines[j])
            if got is None:                      # blank or comment: provisional
                j += 1
                continue
            if got > indent:
                j += 1
                continue
            # A list whose dashes sit AT the key's indentation still belongs
            # to it: `items:` followed by `- name: x` at the same column.
            if got == indent and lines[j].lstrip(" ").startswith("- "):
                j += 1
                continue
            break
        # Trailing blanks and comments belong to whatever comes next.
        while j > body_start and _indent_of(lines[j - 1]) is None:
            j -= 1
        return Block(key=key, indent=indent, start=i,
                     body_start=body_start, body_end=j)
    return None


def _nearest_ancestor(lines: list[str], at: int, indent: int) -> str | None:
    """The key that encloses line `at`, searching upward for less indentation."""
    for i in range(at - 1, -1, -1):
        got = _indent_of(lines[i])
        if got is None or got >= indent:
            continue
        stripped = lines[i].lstrip(" ")
        if stripped.startswith("- "):
            return None
        if ":" in stripped:
            return stripped.split(":", 1)[0].strip()
    return None


def replace_body(text: str, block: Block, body: list[str]) -> str:
    """`text` with `block`'s body replaced by `body`, nothing else touched."""
    lines = text.splitlines(keepends=True)
    fresh = [line if line.endswith("\n") else line + "\n" for line in body]
    return "".join(lines[:block.body_start] + fresh + lines[block.body_end:])


def scalar_extent(lines: list[str], at: int, indent: int) -> int:
    """One past the last line belonging to the scalar whose key is at `at`.

    A YAML value is not a line. PyYAML emits four shapes and three of them
    can span lines:

        plain:          source: supplied by hand, and the
                          nearest listed sibling is another model
        single-quoted:  note: 'A measured rate.
                          Sources: provider A $0.79/hour, provider B $0.82.'
        double-quoted:  note: "A measured rate ... \"L40S\" ..."
        block:          note: |
                          two paragraphs, kept verbatim

    Replacing only the key's own line leaves the rest as orphaned scalars
    inside a mapping and the file stops parsing.

    An indentation heuristic alone (walk forward while a line is deeper and
    does not look like a `key:`) is wrong on prose: `Sources: provider A
    $0.79/hour, ...` -- a sentence inside a quoted value -- looks exactly like
    a key, and cutting the value there leaves continuation lines behind that
    stop the file parsing.

    A quoted scalar does not need a heuristic: it ends at its closing
    quote, and YAML says exactly where that is. A block scalar ends where
    the deeper indentation ends. Only a PLAIN multi-line scalar needs the
    indent rule, and a plain scalar may not contain ": " at all, so there
    the rule is sound rather than a guess.
    """
    _, _, after = lines[at].partition(":")
    value = after.strip()

    if value[:1] in ("|", ">"):
        # A block scalar owns every deeper line, blank lines included; it
        # ends at the last line that carries content.
        end = last = at + 1
        while end < len(lines):
            got = _indent_of(lines[end])
            if got is None:
                end += 1
                continue
            if got <= indent:
                break
            end += 1
            last = end
        return last

    quote = value[:1]
    if quote in ("'", '"'):
        if _closes(value[1:], quote):
            return at + 1
        end = at + 1
        while end < len(lines):
            if _closes(lines[end], quote):
                return end + 1
            end += 1
        # An unterminated quote means the file was already broken; say so
        # rather than swallowing the rest of it into this value.
        raise ValueError(
            f"{lines[at].strip()[:60]!r} opens a {quote} quoted value that "
            f"never closes; the file is not valid YAML and this module will "
            f"not edit it")

    end = at + 1
    while end < len(lines):
        got = _indent_of(lines[end])
        if got is None:
            # Folded plain scalars may contain blank paragraphs. Look
            # through them: include the blanks only when the next content
            # line is another deeper continuation, and leave ordinary
            # separator whitespace attached to the following key.
            look = end + 1
            while look < len(lines) and _indent_of(lines[look]) is None:
                look += 1
            if look >= len(lines):
                break
            look_indent = _indent_of(lines[look])
            look_text = lines[look].lstrip(" ")
            if (look_indent is not None and look_indent > indent
                    and not _looks_like_a_key(look_text)):
                end = look
                continue
            break
        if got <= indent:
            break
        stripped = lines[end].lstrip(" ")
        # A deeper `key:` is a nested mapping, not this value's tail.
        if _looks_like_a_key(stripped):
            break
        end += 1
    return end


def _closes(fragment: str, quote: str) -> bool:
    """Does `fragment` contain the closing `quote` of an open scalar?

    Escapes are what make this more than `quote in fragment`: YAML doubles
    a single quote inside a single-quoted scalar (`don''t`) and
    backslash-escapes a double quote inside a double-quoted one.
    """
    i, n = 0, len(fragment)
    while i < n:
        char = fragment[i]
        if quote == '"' and char == "\\":
            i += 2
            continue
        if char == quote:
            if quote == "'" and i + 1 < n and fragment[i + 1] == "'":
                i += 2
                continue
            return True
        i += 1
    return False


def set_scalar(text: str, key: str, value: str, *, indent: int,
               under: str | None = None) -> tuple[str, bool]:
    """Set `key: value` in place, replacing a WRAPPED value whole.

    The value's extent comes from :func:`scalar_extent`, which reads the
    YAML shape rather than guessing from indentation -- see the note there
    for the prose sentence that broke a task file by looking like a key.
    """
    lines = text.splitlines(keepends=True)
    want = " " * indent + key + ":"
    for i, line in enumerate(lines):
        if not line.startswith(want):
            continue
        if under is not None and _nearest_ancestor(lines, i, indent) != under:
            continue
        end = scalar_extent(lines, i, indent)
        fresh = f"{' ' * indent}{key}: {value}\n"
        if lines[i:end] == [fresh]:
            return text, False
        return "".join(lines[:i] + [fresh] + lines[end:]), True
    return text, False


def insert_scalar(text: str, key: str, value: str, *, indent: int,
                  under: str) -> tuple[str, bool]:
    """Add `key: value` inside `under`, when it is not there to replace.

    `set_scalar` only REPLACES. That is the right default -- a writer that
    invents keys wherever it fails to find one turns a typo into a new field --
    but it means a genuinely new parameter silently never lands, and the
    caller sees no change and reports success.

    So insertion is explicit and narrow: the parent block must already exist,
    and the key goes in alphabetical position among its scalars, which is where
    a re-serialisation would put it and therefore where the next reader will
    look for it.
    """
    lines = text.splitlines(keepends=True)
    block = find(lines, under, indent=indent - 2)
    if block is None:
        return text, False
    fresh = f"{' ' * indent}{key}: {value}\n"
    at = block.body_end
    for i in range(block.body_start, block.body_end):
        got = _indent_of(lines[i])
        if got != indent:
            continue
        name = lines[i].lstrip(" ").split(":", 1)[0]
        if name > key:
            at = i
            break
    return "".join(lines[:at] + [fresh] + lines[at:]), True


def _looks_like_a_key(stripped: str) -> bool:
    """`name: ...` at the start of a line, not prose that happens to have a colon."""
    head, sep, _ = stripped.partition(":")
    return bool(sep) and head.strip() != "" and " " not in head.strip() \
        and not head.strip().startswith(("-", "'", '"', "#"))
