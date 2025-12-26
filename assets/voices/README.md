# Voice Reference Audio

This directory contains reference audio files for Chatterbox TTS voice cloning.

## How Voice Cloning Works

Chatterbox can clone any voice from a short audio sample. The generated speech
will mimic the characteristics of the reference audio (tone, pace, accent, etc.).

## Adding a Voice

1. Record or obtain a clean audio sample of the voice you want to use
2. Requirements:
   - Duration: 10-30 seconds (10 sec minimum for good quality)
   - Format: WAV (preferred) or MP3
   - Quality: Clear speech, minimal background noise
   - Content: Natural speech in any language (French preferred for best results)
3. Save the file in this directory with a descriptive name:
   - `default_french.wav` - Used as the default narrator voice
   - `character_name.wav` - For character-specific voices (future)

## Quick Options for Getting a Voice

### Option 1: Record yourself
Record 10-30 seconds of clear French speech using your phone or computer.

### Option 2: Use a TTS service
Generate a sample using a free TTS service:
- [TTSFree.com](https://ttsfree.com/) - Generate and download, commercial use allowed
- [ttsMP3.com](https://ttsmp3.com/text-to-speech/French/) - French TTS

### Option 3: Extract from existing audio
Use a short clip from a royalty-free audiobook or podcast (with permission).

## Usage

Once you have a reference audio file, the CLI will use it by default:

```bash
# Uses assets/voices/default_french.wav
uv run audiobook generate --book absalon

# Or specify a custom reference:
uv run audiobook generate --book absalon -r assets/voices/my_voice.wav

# Or disable voice cloning:
uv run audiobook generate --book absalon --no-reference-audio
```

## Tips for Best Results

- Use clear, natural speech (not overly dramatic or whispered)
- Avoid background music or noise
- Consistent volume throughout
- For French audiobooks, use a French reference for best accent
- The same voice sample will produce consistent output across multiple generations
