from src.setting import OPENAI_CLIENT as client

if __name__ == "__main__":
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a haiku about recursion in programming."},
        ],
    )

    print(completion.choices[0].message.content)
