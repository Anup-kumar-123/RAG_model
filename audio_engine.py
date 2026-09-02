import edge_tts


async def generate_speech_stream(text: str):
    communicate = edge_tts.Communicate(text, voice="en-US-ChristopherNeural")
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            yield chunk["data"]