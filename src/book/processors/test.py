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

    prompt_parts.append(
        """
Who is speaking this quote? Consider:
1. Look for explicit indicators like 'dit-il', 'répondit-elle', etc.
2. The narrator NEVER speaks in quotes - if it seems like narration, it's likely a character's philosophical moment
3. Check the conversation flow - who is being addressed and who would logically respond
4. Each character's typical speaking style:
   - Tomas often discusses relationships philosophically
   - Tereza expresses emotional concerns
   - Sabina is direct and bold

Return ONLY ONE name from these options: """
        + " | ".join(characters)
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
    n_base_predictions: int = 10,
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
        n=n_base_predictions,
    )
    return [choice.message.content for choice in response.choices]


def apply_post_processing_rules(prediction: str, quote_context: QuoteContext) -> str:
    """Apply post-processing rules to handle common error patterns."""
    # If it looks like a philosophical statement and was attributed to narrator,
    # it's probably Tomas
    if prediction == "narrator" and len(quote_context.quote) > 100:
        return "tomas"

    # If we have previous quotes and this is a short response,
    # give weight to the dialogue context
    if quote_context.previous_quotes and len(quote_context.quote) < 50:
        last_speaker = quote_context.previous_quotes[-1][1].lower()
        if last_speaker != prediction:  # If we're predicting a different speaker
            # Look for question marks in the last quote
            last_quote = quote_context.previous_quotes[-1][0]
            if "?" in last_quote and prediction == "narrator":
                return last_speaker  # It's likely a response from the same speaker

    return prediction


def get_ensemble_prediction(
    quote_context: QuoteContext,
    real_name: str,
    client: OpenAI,
    characters: list[str],
) -> PredictionResult:
    predictions = get_diverse_predictions(quote_context, client, characters)
    vote_counts = Counter(predictions)
    most_common_prediction = vote_counts.most_common(1)[0][0]
    final_prediction = apply_post_processing_rules(most_common_prediction, quote_context)

    if final_prediction != most_common_prediction:
        vote_counts[final_prediction] = vote_counts[most_common_prediction]
        most_common_prediction = final_prediction

    confidence = vote_counts[most_common_prediction] / len(predictions)

    return PredictionResult(
        is_correct=most_common_prediction == real_name.lower(),
        predicted=most_common_prediction,
        real_name=real_name,
        votes=dict(vote_counts),
        confidence=confidence,
    )


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
    all_results = []

    for quote_context, real_name in tqdm(quotes, desc="Testing quotes"):
        result = get_ensemble_prediction(quote_context, real_name, client, characters)
        all_results.append(result)

        if result.is_correct:
            passed += 1
        else:
            failures.append((quote_context, result))

    print("\nOverall Analysis:")
    print(f"Total quotes: {len(quotes)}")
    print(f"Correct predictions: {passed}")
    print(f"Accuracy: {passed/len(quotes)*100:.1f}%")

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
            print(f"{char}: {count} times")
