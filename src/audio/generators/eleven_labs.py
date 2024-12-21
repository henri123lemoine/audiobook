from elevenlabs import play, stream

from src.setting import ELEVENLABS_CLIENT as client

STREAM = False

if STREAM:
    audio_stream = client.text_to_speech.convert_as_stream(
        text="This is a test", voice_id="JBFqnCBsd6RMkjVDRZzb", model_id="eleven_multilingual_v2"
    )

    # option 1: play the streamed audio locally
    stream(audio_stream)

    # option 2: process the audio bytes manually
    for chunk in audio_stream:
        if isinstance(chunk, bytes):
            print(chunk)

else:
    ids = ["aQROLel5sQbj1vuIVi6B"]
    voices = [client.voices.get(voice_id) for voice_id in ids]

    audio = client.generate(text="Bonjour!", voice=voices[0])
    play(audio)
