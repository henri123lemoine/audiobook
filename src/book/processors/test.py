import re
from pathlib import Path
from typing import List, Tuple

from openai import OpenAI
from tqdm import tqdm


def create_prompt(
    text_before: str,
    quote: str,
    text_after: str,
    characters: list[str],
    confidence_required: bool = True,
) -> str:
    confidence_note = "Return 'unknown' unless completely certain. " if confidence_required else ""
    return f"""Who is speaking this quote in "L'Insoutenable Légèreté de l'Être"?

Context before:
{text_before}

Quote:
{quote}

Context after:
{text_after}

{confidence_note}Return ONE name (nothing else) among the following options: {' | '.join(characters)}"""


def test_quote(
    quote: str,
    real_name: str,
    context_before: str,
    context_after: str,
    client: OpenAI,
    characters: list[str],
    confidence_required: bool = True,
) -> Tuple[bool, str]:
    prompt = create_prompt(context_before, quote, context_after, characters, confidence_required)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    predicted = response.choices[0].message.content.strip().lower()
    return predicted == real_name, predicted


if __name__ == "__main__":
    from src.book.books import InsoutenableBook
    from src.setting import L_INSOUTENABLE_TXT_PATH, OPENAI_CLIENT

    characters = [c.name for c in InsoutenableBook.CHARACTERS]
    context_size = 1000

    test_text = L_INSOUTENABLE_TXT_PATH.read_text().split("\n\n\n")[0]

    # Extract quotes with context
    quotes = []
    for match in re.finditer(r'<quote name="([^"]+)">([^<]+)</quote>', test_text):
        name = match.group(1)
        quote = match.group(2)
        start = max(0, match.start() - context_size)
        end = min(len(test_text), match.end() + context_size)
        context_before = test_text[start : match.start()].strip()
        context_after = test_text[match.end() : end].strip()
        quotes.append((quote, name, context_before, context_after))

    failures = []
    uncertain_quotes = []
    passed = 0
    progress = tqdm(quotes, desc="First pass (high confidence)")

    # First pass
    for quote, real_name, context_before, context_after in progress:
        success, predicted = test_quote(
            quote,
            real_name,
            context_before,
            context_after,
            OPENAI_CLIENT,
            characters,
            confidence_required=True,
        )
        if predicted == "unknown":
            uncertain_quotes.append((quote, real_name, context_before, context_after))
        elif success:
            passed += 1
        else:
            failures.append((quote, real_name, predicted))
        progress.set_postfix(
            {"accuracy": f"{passed}/{len(quotes)} ({passed/len(quotes)*100:.1f}%)"}
        )

    print(f"\nFirst pass results:")
    print(f"Correct: {passed}")
    print(f"Wrong: {len(failures)}")
    print(f"Uncertain: {len(uncertain_quotes)}")

    # Print first pass failures
    if failures:
        print("\nFirst pass failures:")
        print("-" * 80)
        for quote, real, predicted in failures:
            print(f"Quote: {quote[:100]}...")
            print(f"Expected: {real}")
            print(f"Got: {predicted}")
            print()

    # Second pass with uncertain quotes
    if uncertain_quotes:
        print(f"\nRetrying {len(uncertain_quotes)} uncertain quotes...")
        second_failures = []
        second_pass = tqdm(uncertain_quotes, desc="Second pass")

        for quote, real_name, context_before, context_after in second_pass:
            success, predicted = test_quote(
                quote,
                real_name,
                context_before,
                context_after,
                OPENAI_CLIENT,
                characters,
                confidence_required=False,
            )
            if success:
                passed += 1
            else:
                second_failures.append((quote, real_name, predicted))
            second_pass.set_postfix(
                {"accuracy": f"{passed}/{len(quotes)} ({passed/len(quotes)*100:.1f}%)"}
            )

        # Print second pass failures
        if second_failures:
            print("\nSecond pass failures:")
            print("-" * 80)
            for quote, real, predicted in second_failures:
                print(f"Quote: {quote[:100]}...")
                print(f"Expected: {real}")
                print(f"Got: {predicted}")
                print()

    print(f"\nFinal accuracy: {passed}/{len(quotes)} ({passed/len(quotes)*100:.1f}%)")
