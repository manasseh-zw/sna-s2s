"""Realtime speech-to-speech bridge using Gemini Live audio mode and local TTS."""

import asyncio
from base64 import b64encode
import os

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from google.genai import types

from llm import (
    DEFAULT_LIVE_MODEL,
    INTRO_GREETING,
    create_gemini_client,
    create_live_connect_config,
    normalize_live_model,
)
from tts import TTSEngine


def _merge_transcription(previous: str, incoming: str) -> str:
    incoming = incoming.strip()
    if not incoming:
        return previous
    if not previous:
        return incoming
    if incoming.startswith(previous):
        return incoming
    if previous.endswith(incoming):
        return previous
    if previous in incoming:
        return incoming

    spacer = "" if previous.endswith((" ", "\n")) or incoming.startswith((".", ",", "!", "?", ";", ":")) else " "
    return f"{previous}{spacer}{incoming}".strip()


async def _send_json_safe(websocket: WebSocket, payload: dict[str, object]) -> None:
    if websocket.client_state == WebSocketState.CONNECTED:
        await websocket.send_json(payload)


async def _forward_browser_audio(
    websocket: WebSocket,
    session,
) -> None:
    while True:
        message = await websocket.receive()

        if message["type"] == "websocket.disconnect":
            raise WebSocketDisconnect()

        audio_bytes = message.get("bytes")
        if not audio_bytes:
            continue

        await session.send_realtime_input(
            audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
        )


async def _forward_gemini_events(
    websocket: WebSocket,
    session,
    tts: TTSEngine,
) -> None:
    current_transcript = ""
    current_reply = ""
    current_reply_finished = False
    while True:
        turn = session.receive()

        async for message in turn:
            if message.voice_activity is not None:
                voice_activity_type = message.voice_activity.voice_activity_type
                if voice_activity_type == types.VoiceActivityType.ACTIVITY_START:
                    await _send_json_safe(websocket, {"type": "voice_activity_start"})
                elif voice_activity_type == types.VoiceActivityType.ACTIVITY_END:
                    await _send_json_safe(websocket, {"type": "voice_activity_end"})

            server_content = message.server_content
            if server_content is None:
                continue

            if server_content.interrupted is True:
                current_reply = ""
                current_reply_finished = False
                await _send_json_safe(websocket, {"type": "interrupted"})

            if server_content.input_transcription is not None:
                if server_content.input_transcription.text:
                    current_transcript = _merge_transcription(
                        current_transcript,
                        server_content.input_transcription.text,
                    )
                    await _send_json_safe(
                        websocket,
                        {
                            "type": "transcript_partial",
                            "text": current_transcript,
                            "finished": bool(server_content.input_transcription.finished),
                        },
                    )

            if server_content.output_transcription is not None:
                if server_content.output_transcription.text:
                    current_reply = _merge_transcription(
                        current_reply,
                        server_content.output_transcription.text,
                    )
                    current_reply_finished = bool(
                        server_content.output_transcription.finished
                    )
                    await _send_json_safe(
                        websocket,
                        {
                            "type": "reply_partial",
                            "text": current_reply,
                            "finished": current_reply_finished,
                        },
                    )

            if server_content.turn_complete and current_reply:
                wav_bytes = await asyncio.to_thread(tts.synthesize, current_reply)
                await _send_json_safe(
                    websocket,
                    {
                        "type": "turn_complete",
                        "transcript": current_transcript,
                        "reply": current_reply,
                        "wav_base64": b64encode(wav_bytes).decode("ascii"),
                    },
                )
                current_transcript = ""
                current_reply = ""
                current_reply_finished = False


async def run_live_s2s_session(websocket: WebSocket, tts: TTSEngine) -> None:
    client = create_gemini_client()
    model = normalize_live_model(os.getenv("GEMINI_LIVE_MODEL", DEFAULT_LIVE_MODEL))
    config = create_live_connect_config()

    await websocket.accept()

    try:
        async with client.aio.live.connect(model=model, config=config) as session:
            intro_wav = await asyncio.to_thread(tts.synthesize, INTRO_GREETING)
            await _send_json_safe(
                websocket,
                {
                    "type": "intro",
                    "reply": INTRO_GREETING,
                    "wav_base64": b64encode(intro_wav).decode("ascii"),
                },
            )
            await _send_json_safe(websocket, {"type": "ready"})

            browser_task = asyncio.create_task(_forward_browser_audio(websocket, session))
            gemini_task = asyncio.create_task(_forward_gemini_events(websocket, session, tts))

            done, pending = await asyncio.wait(
                {browser_task, gemini_task},
                return_when=asyncio.FIRST_EXCEPTION,
            )

            for task in pending:
                task.cancel()

            for task in pending:
                await asyncio.gather(task, return_exceptions=True)

            for task in done:
                exc = task.exception()
                if exc is not None:
                    raise exc
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await _send_json_safe(websocket, {"type": "error", "message": str(exc)})
    finally:
        if (
            websocket.client_state == WebSocketState.CONNECTED
            and websocket.application_state == WebSocketState.CONNECTED
        ):
            await websocket.close()
