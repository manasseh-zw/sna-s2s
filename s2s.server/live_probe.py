"""Minimal Gemini Live connection probe for backend debugging."""

import asyncio

from google.genai import types

from llm import (
    DEFAULT_LIVE_MODEL,
    create_gemini_client,
    create_live_connect_config,
    normalize_live_model,
)


async def probe_minimal() -> None:
    client = create_gemini_client()
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.TEXT],
    )

    async with client.aio.live.connect(
        model=normalize_live_model(DEFAULT_LIVE_MODEL),
        config=config,
    ):
        print("minimal_connect_ok")


async def probe_text_with_prefix() -> None:
    client = create_gemini_client()
    config = types.LiveConnectConfig(
        response_modalities=["TEXT"],
    )

    async with client.aio.live.connect(
        model="models/gemini-3.1-flash-live-preview",
        config=config,
    ):
        print("text_with_prefix_connect_ok")


async def probe_text_with_prefix_and_system() -> None:
    client = create_gemini_client()
    config = types.LiveConnectConfig(
        response_modalities=["TEXT"],
        system_instruction=types.Content(parts=[types.Part(text="Pindura nechiShona.")]),
    )

    async with client.aio.live.connect(
        model="models/gemini-3.1-flash-live-preview",
        config=config,
    ):
        print("text_with_prefix_and_system_connect_ok")


async def probe_exact_snippet() -> None:
    client = create_gemini_client()
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        media_resolution="MEDIA_RESOLUTION_MEDIUM",
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name="Zephyr"
                )
            )
        ),
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=104857,
            sliding_window=types.SlidingWindow(target_tokens=52428),
        ),
    )

    async with client.aio.live.connect(
        model="models/gemini-3.1-flash-live-preview",
        config=config,
    ):
        print("exact_snippet_connect_ok")


async def probe_app_config() -> None:
    client = create_gemini_client()
    config = create_live_connect_config()

    async with client.aio.live.connect(
        model=normalize_live_model(DEFAULT_LIVE_MODEL),
        config=config,
    ):
        print("app_config_connect_ok")


async def probe_audio_with_transcription() -> None:
    client = create_gemini_client()
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(parts=[types.Part(text="Pindura nechiShona chete.")]),
    )

    async with client.aio.live.connect(
        model="models/gemini-3.1-flash-live-preview",
        config=config,
    ) as session:
        print("audio_with_transcription_connect_ok")
        await session.send_realtime_input(text="Mhoro, uri sei?")

        async for response in session.receive():
            content = response.server_content
            if content and content.output_transcription and content.output_transcription.text:
                print(f"output_transcription={content.output_transcription.text}")
                break


async def main() -> None:
    probes = [
        ("minimal", probe_minimal),
        ("text_with_prefix", probe_text_with_prefix),
        ("text_with_prefix_and_system", probe_text_with_prefix_and_system),
        ("exact_snippet", probe_exact_snippet),
        ("app_config", probe_app_config),
        ("audio_with_transcription", probe_audio_with_transcription),
    ]

    for name, probe in probes:
        try:
            await probe()
        except Exception as exc:
            print(f"{name}_failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
