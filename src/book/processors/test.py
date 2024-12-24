import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm


class PredictionStrategy(Enum):
    BASE = "base"  # Initial GPT-4-mini attempt
    ESCALATED = "escalated"  # Escalated to GPT-4
    EXTENDED_CONTEXT = "extended_context"  # Using more context


@dataclass
class QuoteContext:
    text_before: str
    text_after: str
    quote: str
    chapter_title: Optional[str] = None
    previous_quotes: List[Tuple[str, str]] = None


@dataclass
class ModelPrediction:
    prediction: str
    confidence: float
    votes: Dict[str, int]


@dataclass
class PredictionResult:
    is_correct: bool
    predicted: str
    real_name: str
    votes: Dict[str, int]
    confidence: float
    strategy_used: PredictionStrategy
    context_size: int
    mini_prediction: Optional[ModelPrediction] = None  # Original GPT-4-mini prediction
    gpt4_prediction: Optional[ModelPrediction] = None  # GPT-4 prediction if used
    extended_prediction: Optional[ModelPrediction] = None  # Extended context prediction if used


def clean_context(text: str) -> str:
    """Remove XML-style quote tags but keep the text and attribution indicators."""
    # Replace <quote name="X">text</quote> with just the text
    text = re.sub(r'<quote name="[^"]+">([^<]+)</quote>', r"\1", text)
    return text


def create_prompt(quote_context: QuoteContext, characters: list[str]) -> str:
    # Clean the context before using it
    clean_before = clean_context(quote_context.text_before)
    clean_after = clean_context(quote_context.text_after)

    text_parts = []
    text_parts.append(clean_before)
    text_parts.append(f"<QUOTE>{quote_context.quote}</QUOTE>")

    # Only take first line of after-context
    first_line = clean_after.split("\n")[0] if clean_after else ""
    if first_line.endswith(":"):
        first_line = first_line[:-1] + "..."
    text_parts.append(first_line)

    prompt = "\n".join(
        [
            '"""',
            "\n".join(text_parts),
            '"""',
            f"\nWho speaks the quote? Return ONE name from: {' | '.join(characters + ['unknown'])}",
        ]
    )

    return prompt


def get_quote_context(text: str, match: re.Match, context_size: int) -> QuoteContext:
    quote_start = match.start()
    quote_end = match.end()

    previous_quotes = []
    prev_quote_pattern = r'<quote name="([^"]+)">([^<]+)</quote>'
    for prev_match in re.finditer(prev_quote_pattern, text[:quote_start]):
        previous_quotes.append((prev_match.group(2), prev_match.group(1)))

    chapter_start = text.rfind("\n\n\n", 0, quote_start)
    if chapter_start == -1:
        chapter_start = max(0, quote_start - context_size)
    start = max(chapter_start, quote_start - context_size)
    text_before = text[start:quote_start].strip()

    next_quote = text.find("<quote", quote_end)
    if next_quote == -1:
        next_quote = min(len(text), quote_end + context_size)
    text_after = text[quote_end:next_quote].strip()

    chapter_title = None
    chapter_matches = list(re.finditer(r"\n\n([IVX]+)\n\n", text[:quote_start]))
    if chapter_matches:
        chapter_title = chapter_matches[-1].group(1)

    return QuoteContext(
        text_before=text_before,
        text_after=text_after,
        quote=match.group(2),
        chapter_title=chapter_title,
        previous_quotes=previous_quotes[-3:] if previous_quotes else None,
    )


def get_predictions(
    quote_context: QuoteContext,
    client: OpenAI,
    characters: list[str],
    model: str = "gpt-4o-mini",
    n_predictions: int = 20,
) -> List[str]:
    base_prompt = create_prompt(quote_context, characters)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "assistant", "content": "You are an expert literary analyst."},
            {"role": "user", "content": base_prompt},
        ],
        temperature=1.0,
        max_tokens=3,
        n=n_predictions,
    )
    return [choice.message.content.strip().lower() for choice in response.choices]


def get_model_prediction(
    quote_context: QuoteContext,
    client: OpenAI,
    characters: list[str],
    model: str = "gpt-4o-mini",
) -> ModelPrediction:
    predictions = get_predictions(quote_context, client, characters, model=model)
    vote_counts = Counter(predictions)
    most_common = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[most_common] / len(predictions)

    return ModelPrediction(prediction=most_common, confidence=confidence, votes=dict(vote_counts))


def get_ensemble_prediction(
    quote_context: QuoteContext,
    real_name: str,
    client: OpenAI,
    characters: list[str],
    context_size: int,
    full_text: str,
    match: re.Match,
    confidence_threshold: float = 0.8,
) -> PredictionResult:
    # First try with GPT-4-mini
    mini_pred = get_model_prediction(quote_context, client, characters, model="gpt-4o-mini")

    strategy = PredictionStrategy.BASE
    final_prediction = mini_pred
    extended_pred = None
    gpt4_pred = None

    # If confidence is low, try with more context
    if mini_pred.confidence < confidence_threshold:
        extended_context = get_quote_context(full_text, match, context_size * 2)
        extended_pred = get_model_prediction(
            extended_context, client, characters, model="gpt-4o-mini"
        )

        if extended_pred.confidence > mini_pred.confidence:
            strategy = PredictionStrategy.EXTENDED_CONTEXT
            final_prediction = extended_pred

    # If still uncertain, escalate to GPT-4
    if final_prediction.confidence < confidence_threshold:
        gpt4_pred = get_model_prediction(quote_context, client, characters, model="gpt-4o")
        if gpt4_pred.confidence > final_prediction.confidence:
            strategy = PredictionStrategy.ESCALATED
            final_prediction = gpt4_pred

    # If still low confidence or invalid prediction, return unknown
    if (
        final_prediction.confidence < confidence_threshold
        or final_prediction.prediction not in characters + ["unknown"]
    ):
        final_prediction.prediction = "unknown"
        final_prediction.confidence = 0.0

    return PredictionResult(
        is_correct=final_prediction.prediction == real_name.lower(),
        predicted=final_prediction.prediction,
        real_name=real_name,
        votes=final_prediction.votes,
        confidence=final_prediction.confidence,
        strategy_used=strategy,
        context_size=context_size,
        mini_prediction=mini_pred,
        extended_prediction=extended_pred,
        gpt4_prediction=gpt4_pred,
    )


if __name__ == "__main__":
    from src.book.books import InsoutenableBook
    from src.setting import L_INSOUTENABLE_TXT_PATH
    from src.setting import OPENAI_CLIENT as client

    characters = [c.name for c in InsoutenableBook.CHARACTERS]
    characters.remove("narrator")
    context_size = 2000

    test_text = L_INSOUTENABLE_TXT_PATH.read_text().split("\n\n\n")[0]

    quotes = []
    for match in re.finditer(r'<quote name="([^"]+)">([^<]+)</quote>', test_text):
        name = match.group(1)
        quote_context = get_quote_context(test_text, match, context_size)
        quotes.append((quote_context, name, match))

    # Test each quote
    failures = []
    passed = 0
    blank = 0
    all_results = []
    strategy_counts = Counter()
    strategy_successes = Counter()

    # Track model comparisons
    escalation_improvements = []  # Cases where GPT-4 corrected GPT-4-mini
    context_improvements = []  # Cases where extended context helped
    failed_escalations = []  # Cases where GPT-4 failed to help

    for quote_context, real_name, match in tqdm(quotes, desc="Testing quotes"):
        result = get_ensemble_prediction(
            quote_context,
            real_name,
            client,
            characters,
            context_size,
            full_text=test_text,
            match=match,
        )
        all_results.append(result)

        strategy_counts[result.strategy_used] += 1
        if result.is_correct:
            strategy_successes[result.strategy_used] += 1

        # Track model comparisons
        if result.gpt4_prediction:
            mini_correct = result.mini_prediction.prediction == real_name.lower()
            gpt4_correct = result.gpt4_prediction.prediction == real_name.lower()

            if not mini_correct and gpt4_correct:
                escalation_improvements.append((quote_context, result))
            elif not mini_correct and not gpt4_correct:
                failed_escalations.append((quote_context, result))

        if result.extended_prediction:
            if (
                result.mini_prediction.prediction != real_name.lower()
                and result.extended_prediction.prediction == real_name.lower()
            ):
                context_improvements.append((quote_context, result))

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

    print("\nModel Comparison Analysis:")
    print(f"\nGPT-4 Escalation Results:")
    print(f"Times GPT-4 improved prediction: {len(escalation_improvements)}")
    print(f"Times GPT-4 failed to improve: {len(failed_escalations)}")
    if escalation_improvements:
        print("\nSuccessful GPT-4 corrections:")
        for quote_context, result in escalation_improvements:
            print(f"\nQuote: {quote_context.quote[:100]}...")
            print(
                f"GPT-4-mini predicted: {result.mini_prediction.prediction} (confidence: {result.mini_prediction.confidence:.2f})"
            )
            print(
                f"GPT-4 corrected to: {result.gpt4_prediction.prediction} (confidence: {result.gpt4_prediction.confidence:.2f})"
            )

    print(f"\nExtended Context Results:")
    print(f"Times extended context helped: {len(context_improvements)}")
    if context_improvements:
        print("\nSuccessful context extensions:")
        for quote_context, result in context_improvements:
            print(f"\nQuote: {quote_context.quote[:100]}...")
            print(
                f"Original prediction: {result.mini_prediction.prediction} (confidence: {result.mini_prediction.confidence:.2f})"
            )
            print(
                f"Extended context prediction: {result.extended_prediction.prediction} (confidence: {result.extended_prediction.confidence:.2f})"
            )

    print("\nStrategy Analysis:")
    for strategy in PredictionStrategy:
        count = strategy_counts[strategy]
        successes = strategy_successes[strategy]
        if count > 0:
            print(f"\n{strategy.value}:")
            print(f"Used {count} times ({count/len(quotes)*100:.1f}% of quotes)")
            print(f"Success rate: {successes/count*100:.1f}%")

    if failures:
        print("\nDetailed Failure Analysis:")
        print("-" * 80)
        for quote_context, result in failures:
            print(f"Quote: {quote_context.quote[:100]}...")
            print(f"Context before: {quote_context.text_before[-100:]}...")
            print(f"Expected: {result.real_name}")
            print(f"Predicted: {result.predicted}")
            print(f"Vote distribution: {result.votes}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Strategy used: {result.strategy_used.value}")
            if result.gpt4_prediction:
                print(
                    f"GPT-4 prediction: {result.gpt4_prediction.prediction} (conf: {result.gpt4_prediction.confidence:.2f})"
                )
            print()

        print("\nFailure Patterns:")
        failure_chars = Counter(r.real_name for _, r in failures)
        print("Most commonly misidentified characters:")
        for char, count in failure_chars.most_common():
            print(f"{char}: {count} times")
