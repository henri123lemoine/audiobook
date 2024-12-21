# Audiobook

An automatic audiobook generator. Meant as a gift. Mainly focused on L'Insoutenable Légèreté de l'Être for now.

## Features

TODO

## Getting Started

This project uses `uv`. Installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

1. Clone the repository
   ```bash
    git clone https://github.com/henri123lemoine/audiobook.git
    cd audiobook
    ```

2. Run the project! This should take care of installing dependencies, setting up the virtual environment, and running the project.
   ```bash
   uv run -m [file.path.no.ext]
   ```

## TODO

- [ ] Content Extraction
  - [ ] PDF
    - [ ] Fix the PDF cache
  - [ ] Books
    - [ ] Implement getting speakers using LLM & context
    - [ ] L'Insoutenable
      - [ ] Set up the preprocessing to be done directly on the .txt file
      - [ ] Fix the preprocessing
        - [ ] Fix the parts-splitting
        - [ ] Fix the chapter-splitting
        - [ ] Fix the paragraph-splitting (paragraphs)
        - [ ] Fix the segment-splitting (not quite sentences, just switches in speaker, if that makes sense? like a paragraph can have multiple segments if most of it is narrator and then it switches to a character speaking and back to the narrator)
      - [ ] Find good TTS voices to match with each character
- [ ] Audio
  - [ ] Find all the good french voices
- [ ] Other

Additional notes:

- tomas (m) Eng Ger
- tereza (w) Eng
- sabina (w) Eng
- cadavre (w) Eng (multiple cadavres, maybe add an echo sound or smth?)
- beethoven (m) Ger
- tereza mother (w) Eng
- tereza father (m) Eng
- inconnu (m) Eng
- photographe (w) Eng
- redacteur (m) Eng

`<quote name="name">text</quote>` means quote said by someone
`<manual>text</manual>` means go through it manually later
`<quote name="tomas" language="german">` to set language as well (optional)
[IMAGE] lorsqu'il y a une image

current progress: TROISIÈME  PARTIE  LES  MOTS  INCOMPRIS

## Development

To install development dependencies, run `uv pip install -r pyproject.toml --all-extras`.

To run tests, use `uv run pytest`.
