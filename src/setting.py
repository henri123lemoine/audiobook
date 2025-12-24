import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# General
DATE = datetime.now().strftime("%Y-%m-%d")
PROJECT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_PATH / "data"
os.makedirs(DATA_PATH, exist_ok=True)

# API keys
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients only if API keys are available (avoid import-time errors)
ELEVENLABS_CLIENT = None
OPENAI_CLIENT = None

if ELEVENLABS_API_KEY:
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_CLIENT = ElevenLabs(api_key=ELEVENLABS_API_KEY)

if OPENAI_API_KEY:
    from openai import OpenAI
    OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
