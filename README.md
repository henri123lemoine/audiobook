# Audiobook

An automatic audiobook generator that converts French literary works into audiobooks using AI voices. Currently supports "L'Insoutenable Légèreté de l'Être" and (WIP) "Absalon, Absalon!".

## Features

- Automated text processing pipeline for complex literary works
- Character voice assignment system with multi-voice support (WIP, not super robust)
- Intelligent quote attribution
- Audio generation using ElevenLabs TTS
- Caching system for PDF processing
- Robust logging system

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

## Current Focus

- Processing complex narrative structures in "Absalon, Absalon !"
  - Handling nested dialogues
  - Managing multiple narrator perspectives
  - Processing complex chapter structures
- Voice assignment improvements
  - Better character voice matching
  - Support for emotional variations
  - Handling of dialect and accent specifications

## Development

To install development dependencies, run `uv sync --all-extras`.

To run tests, use `uv run pytest`.
