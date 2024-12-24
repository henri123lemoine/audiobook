import re
from pathlib import Path
from typing import List, Tuple

from openai import OpenAI
from tqdm import tqdm


def create_prompt(text_before: str, quote: str, characters: list[str]) -> str:
    return f"""Who is speaking this quote in "L'Insoutenable Légèreté de l'Être"?

Context:
{text_before}

Quote:
{quote}

Return ONE name (nothing else) among the following options: {' | '.join(characters)}"""


def test_quote(
    quote: str, real_name: str, context: str, client: OpenAI, characters: list[str]
) -> Tuple[bool, str]:
    prompt = create_prompt(context, quote, characters)
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
    context_size = 1500

    # Test on first part only
    test_text = L_INSOUTENABLE_TXT_PATH.read_text().split("\n\n\n")[0]

    # Extract all quotes with their context
    quotes = []
    for match in re.finditer(r'<quote name="([^"]+)">([^<]+)</quote>', test_text):
        name = match.group(1)
        quote = match.group(2)
        start = max(0, match.start() - context_size)
        context = test_text[start : match.start()].strip()
        quotes.append((quote, name, context))

    # Test each quote
    failures = []
    passed = 0
    progress = tqdm(quotes, desc="Testing quotes")

    for quote, real_name, context in progress:
        success, predicted = test_quote(quote, real_name, context, OPENAI_CLIENT, characters)
        if success:
            passed += 1
        else:
            failures.append((quote, real_name, predicted))
        progress.set_postfix(
            {"accuracy": f"{passed}/{len(quotes)} ({passed/len(quotes)*100:.1f}%)"}
        )

    # Print failures at the end
    if failures:
        print("\nFailures:")
        print("-" * 80)
        for quote, real, predicted in failures:
            print(f"Quote: {quote[:100]}...")
            print(f"Expected: {real}")
            print(f"Got: {predicted}")
            print()
