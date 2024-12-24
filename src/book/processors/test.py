import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional

from openai import OpenAI
from tqdm import tqdm


@dataclass
class QuoteContext:
    text_before: str
    text_after: str
    quote: str
    chapter_title: Optional[str] = None


@dataclass
class PredictionResult:
    is_correct: bool
    predicted: str
    real_name: str
    votes: Dict[str, int]
    confidence: float  # Ratio of winning votes to total votes


def create_prompt(quote_context: QuoteContext, characters: list[str]) -> str:
    prompt_parts = []

    if quote_context.chapter_title:
        prompt_parts.append(f"Chapter: {quote_context.chapter_title}")

    prompt_parts.append("Context before the quote:")
    prompt_parts.append(quote_context.text_before)

    prompt_parts.append("\nQuote:")
    prompt_parts.append(quote_context.quote)

    if quote_context.text_after:
        prompt_parts.append("\nContext after the quote:")
        prompt_parts.append(quote_context.text_after)

    prompt_parts.append(f"\nWho is speaking this quote in 'L'Insoutenable Légèreté de l'Être'?")
    prompt_parts.append(
        f"Return ONE name (nothing else) among the following options: {' | '.join(characters)}"
    )
    prompt_parts.append("\nPay special attention to:")
    prompt_parts.append(
        "1. Dialogue indicators like 'dit-il', 'répondit-elle' that appear after the quote"
    )
    prompt_parts.append("2. The context of the conversation and who is speaking to whom")
    prompt_parts.append("3. The content and style of speech characteristic to each character")

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
    )


def get_ensemble_prediction(
    quote_context: QuoteContext,
    real_name: str,
    client: OpenAI,
    characters: list[str],
    n_predictions: int = 5,
) -> PredictionResult:
    prompt = create_prompt(quote_context, characters)

    # Get multiple predictions with different temperatures
    predictions = []
    temperatures = [0.0, 0.1, 0.2, 0.3, 0.4]  # Use different temperatures for diversity

    for temp in temperatures[:n_predictions]:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=10,
        )
        predicted = response.choices[0].message.content.strip().lower()
        predictions.append(predicted)

    # Count votes
    vote_counts = Counter(predictions)

    # Get the most common prediction
    most_common_prediction = vote_counts.most_common(1)[0][0]

    # Calculate confidence as ratio of winning votes to total votes
    confidence = vote_counts[most_common_prediction] / n_predictions

    return PredictionResult(
        is_correct=most_common_prediction == real_name.lower(),
        predicted=most_common_prediction,
        real_name=real_name,
        votes=dict(vote_counts),
        confidence=confidence,
    )


if __name__ == "__main__":
    from src.book.books import InsoutenableBook
    from src.setting import L_INSOUTENABLE_TXT_PATH, OPENAI_CLIENT

    characters = [c.name for c in InsoutenableBook.CHARACTERS]
    context_size = 2000

    # Test on first part only
    test_text = L_INSOUTENABLE_TXT_PATH.read_text().split("\n\n\n")[0]

    # Extract all quotes with their context
    quotes = []
    for match in re.finditer(r'<quote name="([^"]+)">([^<]+)</quote>', test_text):
        name = match.group(1)
        quote_context = get_quote_context(test_text, match, context_size)
        quotes.append((quote_context, name))

    # Test each quote
    failures = []
    passed = 0
    all_results = []
    progress = tqdm(quotes, desc="Testing quotes")

    for quote_context, real_name in progress:
        result = get_ensemble_prediction(quote_context, real_name, OPENAI_CLIENT, characters)
        all_results.append(result)

        if result.is_correct:
            passed += 1
        else:
            failures.append((quote_context, result))

        progress.set_postfix(
            {
                "accuracy": f"{passed}/{len(quotes)} ({passed/len(quotes)*100:.1f}%)",
                "confidence": f"{result.confidence:.2f}",
            }
        )

    # Print detailed analysis
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
            print(f"Context after: {quote_context.text_after[:100]}...")
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
