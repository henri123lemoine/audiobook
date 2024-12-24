import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm


@dataclass
class QuoteContext:
    text_before: str
    text_after: str
    quote: str
    chapter_title: Optional[str] = None
    previous_quotes: List[Tuple[str, str]] = None  # list of (quote, speaker) pairs


@dataclass
class PredictionResult:
    is_correct: bool
    predicted: str
    real_name: str
    votes: Dict[str, int]
    confidence: float  # Ratio of winning votes to total votes


def create_prompt(quote_context: QuoteContext, characters: list[str]) -> str:
    prompt_parts = []

    # Add character-specific context
    prompt_parts.append("""
Key information about the characters:
- Tomas: A surgeon and womanizer who often speaks about philosophical concepts and relationships
- Tereza: Emotional and vulnerable, often expresses concern or anxiety
- Sabina: An artist, direct and bold in her speech
- The Narrator: Never speaks in dialogue, only provides commentary and analysis

IMPORTANT: If you're not sure about the speaker, respond with "unknown". Only attribute a quote to a character if you're confident.
""")

    if quote_context.chapter_title:
        prompt_parts.append(f"Chapter: {quote_context.chapter_title}")

    if quote_context.previous_quotes:
        prompt_parts.append("\nRecent dialogue:")
        for prev_quote, speaker in quote_context.previous_quotes[-3:]:
            prompt_parts.append(f'{speaker}: "{prev_quote}"')

    prompt_parts.append("\nContext before the quote:")
    prompt_parts.append(quote_context.text_before)

    prompt_parts.append("\nQuote to attribute:")
    prompt_parts.append(f'"{quote_context.quote}"')

    # Look for explicit speaker indicators
    if "dit-il" in quote_context.text_after.lower():
        prompt_parts.append("\nNote: The quote is followed by 'dit-il' (indicating a male speaker)")
    elif "dit-elle" in quote_context.text_after.lower():
        prompt_parts.append(
            "\nNote: The quote is followed by 'dit-elle' (indicating a female speaker)"
        )

    prompt_parts.append(
        """
Who is speaking this quote? Consider:
1. Look for explicit indicators like 'dit-il', 'répondit-elle', etc.
2. The narrator NEVER speaks in quotes
3. Check the conversation flow
4. If you're not sure, respond with "unknown"

Return ONLY ONE name from these options: """
        + " | ".join(characters + ["unknown"])
    )

    return "\n".join(prompt_parts)


def extract_chapter_title(text: str, quote_start: int) -> Optional[str]:
    chapter_pattern = r"\n\n([IVX]+)\n\n"
    matches = list(re.finditer(chapter_pattern, text[:quote_start]))
    if matches:
        return matches[-1].group(1)
    return None


def get_quote_context(text: str, match: re.Match, context_size: int) -> QuoteContext:
    quote_start = match.start()
    quote_end = match.end()

    # Get previous quotes
    previous_quotes = []
    prev_quote_pattern = r'<quote name="([^"]+)">([^<]+)</quote>'
    for prev_match in re.finditer(prev_quote_pattern, text[:quote_start]):
        previous_quotes.append((prev_match.group(2), prev_match.group(1)))

    # Get context before quote
    chapter_start = text.rfind("\n\n\n", 0, quote_start)
    if chapter_start == -1:
        chapter_start = max(0, quote_start - context_size)
    start = max(chapter_start, quote_start - context_size)
    text_before = text[start:quote_start].strip()

    # Get context after quote
    next_quote = text.find("<quote", quote_end)
    if next_quote == -1:
        next_quote = min(len(text), quote_end + context_size)
    text_after = text[quote_end:next_quote].strip()

    chapter_title = extract_chapter_title(text, quote_start)

    return QuoteContext(
        text_before=text_before,
        text_after=text_after,
        quote=match.group(2),
        chapter_title=chapter_title,
        previous_quotes=previous_quotes[-3:] if previous_quotes else None,  # Keep last 3
    )


def get_diverse_predictions(
    quote_context: QuoteContext,
    client: OpenAI,
    characters: list[str],
    n_predictions: int = 10,
) -> List[str]:
    base_prompt = create_prompt(quote_context, characters)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "assistant", "content": "You are an expert literary analyst."},
            {"role": "user", "content": base_prompt},
        ],
        temperature=1.3,
        max_tokens=3,
        n=n_predictions,
    )
    return [choice.message.content.strip().lower() for choice in response.choices]


def clean_prediction(prediction: str) -> str:
    # Handle common variations and typos
    prediction = prediction.strip().lower()
    if prediction in ["teresa", "theresa"]:
        return "tereza"
    if prediction == "thomas":
        return "tomas"
    if prediction.startswith("t ") or prediction == "t":  # Common truncation for Tomas
        return "tomas"
    if prediction in ["_unknown_", "unclear", ""]:
        return "unknown"
    return prediction


def get_ensemble_prediction(
    quote_context: QuoteContext,
    real_name: str,
    client: OpenAI,
    characters: list[str],
) -> PredictionResult:
    predictions = get_diverse_predictions(quote_context, client, characters)
    cleaned_predictions = [clean_prediction(p) for p in predictions]

    vote_counts = Counter(cleaned_predictions)

    # Get the most common prediction
    most_common_prediction = vote_counts.most_common(1)[0][0]
    total_votes = len(predictions)
    confidence = vote_counts[most_common_prediction] / total_votes

    # If confidence is low, return unknown
    if confidence < 0.6 or most_common_prediction not in characters + ["unknown"]:
        most_common_prediction = "unknown"
        confidence = 0.0

    return PredictionResult(
        is_correct=most_common_prediction == real_name.lower(),
        predicted=most_common_prediction,
        real_name=real_name,
        votes=dict(vote_counts),
        confidence=confidence,
    )


def process_quotes(text: str, client: OpenAI, characters: list[str], context_size: int = 3000):
    """Process all quotes in the text and return a list of tuples (quote, predicted_name, confidence)"""
    results = []
    for match in re.finditer(r'<quote name="([^"]+)">([^<]+)</quote>', text):
        quote_context = get_quote_context(text, match, context_size)
        result = get_ensemble_prediction(quote_context, match.group(1), client, characters)
        results.append(
            (
                match.group(2),
                result.predicted if result.confidence >= 0.6 else "",
                result.confidence,
            )
        )
    return results


if __name__ == "__main__":
    from src.book.books import InsoutenableBook
    from src.setting import L_INSOUTENABLE_TXT_PATH
    from src.setting import OPENAI_CLIENT as client

    characters = [c.name for c in InsoutenableBook.CHARACTERS]
    characters.remove("narrator")  # Remove narrator from options
    context_size = 3000

    test_text = L_INSOUTENABLE_TXT_PATH.read_text().split("\n\n\n")[0]

    quotes = []
    for match in re.finditer(r'<quote name="([^"]+)">([^<]+)</quote>', test_text):
        name = match.group(1)
        quote_context = get_quote_context(test_text, match, context_size)
        quotes.append((quote_context, name))

    # Test each quote
    failures = []
    passed = 0
    blank = 0
    all_results = []

    for quote_context, real_name in tqdm(quotes, desc="Testing quotes"):
        result = get_ensemble_prediction(quote_context, real_name, client, characters)
        all_results.append(result)

        if result.predicted == "unknown":
            blank += 1
        elif result.is_correct:
            passed += 1
        else:
            failures.append((quote_context, result))

    print("\nOverall Analysis:")
    print(f"Total quotes: {len(quotes)}")
    print(f"Correct predictions: {passed}")
    print(f"Left blank (unknown): {blank}")
    print(f"Incorrect predictions: {len(failures)}")
    print(f"Accuracy (excluding blanks): {passed/(len(quotes)-blank)*100:.1f}%")
    print(f"Accuracy (including blanks): {passed/len(quotes)*100:.1f}%")

    # Analyze confidence levels
    confidences = [r.confidence for r in all_results]
    avg_confidence = sum(confidences) / len(confidences)
    print(f"\nAverage confidence: {avg_confidence:.2f}")

    # Analyze failures
    if failures:
        print("\nDetailed Failure Analysis:")
        print("-" * 80)
        for quote_context, result in failures:
            print(f"Quote: {quote_context.quote[:100]}...")
            print(f"Context before: {quote_context.text_before[-100:]}...")
            if quote_context.chapter_title:
                print(f"Chapter: {quote_context.chapter_title}")
            print(f"Expected: {result.real_name}")
            print(f"Predicted: {result.predicted}")
            print(f"Vote distribution: {result.votes}")
            print(f"Confidence: {result.confidence:.2f}")
            print()

        # Analyze patterns in failures
        print("\nFailure Patterns:")
        failure_chars = Counter(r.real_name for _, r in failures)
        print("Most commonly misidentified characters:")
        for char, count in failure_chars.most_common():
            print(f"{char}: {count} times")
