import re
from dataclasses import dataclass


@dataclass
class Character:
    """Represents a character in the book."""

    name: str
    description: str | None = None


@dataclass
class Segment:
    text: str
    character: Character | None = None
    language: str | None = None

    @classmethod
    def from_text(cls, text: str, default_language: str) -> "Segment":
        quote_pattern = r'<quote(?: name="([^"]*)")?(?: language="([^"]*)")?>([^<]+)</quote>'
        if match := re.match(quote_pattern, text):
            name, language, content = match.groups()
            # Convert empty name to "unknown"
            name = "unknown" if name == "" else name
            return cls(
                text=content.strip(),
                character=Character(name=name) if name else None,
                language=language or default_language,
            )
        return cls(text=text.strip(), language=default_language)


@dataclass
class Chapter:
    """A chapter containing paragraphs."""

    number: int
    segments: list[Segment]
    title: str | None = None


@dataclass
class Part:
    """A major section/part containing chapters."""

    number: int
    chapters: list[Chapter]
    title: str | None = None
