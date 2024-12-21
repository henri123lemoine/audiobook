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
      - [ ] Fix the preprocessing
        - [ ] Fix the parts-splitting
        - [ ] Fix the chapter-splitting
        - [ ] Fix the paragraph-splitting (paragraphs)
        - [ ] Fix the segment-splitting (not quite sentences, just switches in speaker, if that makes sense? like a paragraph can have multiple segments if most of it is narrator and then it switches to a character speaking and back to the narrator)
      - [ ] Find good TTS voices to match with each character
- [ ] Audio
  - [ ] Find all the good french voices
- [ ] Other

## Development

To install development dependencies, run `uv pip install -r pyproject.toml --all-extras`.

To run tests, use `uv run pytest`.
