import re

from openai import OpenAI


def create_prompt(text_before: str, quote: str, characters: list[str]) -> str:
    return f"""Who is speaking this quote in "L'Insoutenable Légèreté de l'Être"?

Context:
{text_before}

Quote:
{quote}

Return ONE name (nothing else) among the following options: {' | '.join(characters)}"""


def process_dialogue(text: str, client: OpenAI, characters: list[str]) -> str:
    output_text = text
    matches = list(re.finditer(r'<quote name="([^"]*)">([^<]+)</quote>', text))

    for match in matches:
        current_name = match.group(1)
        if current_name and current_name != "unknown":
            continue

        # Get some context to help identify the speaker
        quote_text = match.group(2)
        start = max(0, match.start() - 1500)
        context = text[start : match.start()].strip()

        prompt = create_prompt(context, quote_text, characters)
        # print(prompt)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        new_name = response.choices[0].message.content.strip().lower()
        print(new_name)
        if new_name and new_name != "unknown":
            old_quote = f'<quote name="{current_name if current_name else ""}">'
            new_quote = f'<quote name="{new_name}">'
            output_text = output_text.replace(old_quote + quote_text, new_quote + quote_text)

    return output_text


if __name__ == "__main__":
    from src.book.books.l_insoutenable import InsoutenableBook
    from src.setting import L_INSOUTENABLE_TXT_PATH, OPENAI_CLIENT

    parts = L_INSOUTENABLE_TXT_PATH.read_text().split("\n\n\n")
    test_text = parts[0]
    test_text = "\n\n".join(test_text.split("\n\n")[8:12])

    # Save original speaker assignments to check accuracy
    original_quotes = dict(re.findall(r'<quote name="([^"]+)">([^<]+)</quote>', test_text))
    print(original_quotes)
    # raise

    # Process with blanked out names
    anonymized = re.sub(r'<quote name="[^"]*">', '<quote name="">', test_text)
    characters = [c.name for c in InsoutenableBook.CHARACTERS]
    processed = process_dialogue(anonymized, OPENAI_CLIENT, characters)

    # Compare results
    processed_quotes = dict(re.findall(r'<quote name="([^"]+)">([^<]+)</quote>', processed))

    print("\nResults:")
    print("-" * 40)
    for quote, expected in original_quotes.items():
        predicted = processed_quotes.get(quote, "None")
        print(f"{'✓' if predicted == expected else '✗'} {quote[:50]}...")
        print(f"  Expected: {expected}")
        print(f"  Got: {predicted}\n")

    accuracy = sum(1 for q, e in original_quotes.items() if processed_quotes.get(q) == e)
    print(f"Accuracy: {accuracy}/{len(original_quotes)} ({accuracy/len(original_quotes)*100:.1f}%)")
