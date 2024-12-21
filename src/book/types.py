from dataclasses import dataclass


@dataclass
class Segment:
    """A segment of text, typically a sentence or dialogue."""

    text: str
    is_dialogue: bool = False
    language: str | None = None


@dataclass
class Paragraph:
    """A paragraph containing one or more segments."""

    segments: list[Segment]


@dataclass
class Chapter:
    """A chapter containing paragraphs."""

    number: int
    title: str | None
    paragraphs: list[Paragraph]


@dataclass
class Part:
    """A major section/part containing chapters."""

    number: int
    title: str | None
    chapters: list[Chapter]
    title: str | None
    chapters: list[Chapter]


@dataclass
class Character:
    """Represents a character in the book."""

    name: str
    description: str | None = None
