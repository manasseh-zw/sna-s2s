"""Realtime speech-to-speech bridge using Gemini Live text output and local TTS."""

import asyncio
from base64 import b64encode
import os

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from google.genai import types

from llm import DEFAULT_LIVE_MODEL, create_gemini_client, create_live_connect_config
from tts import TTSEngine


def _extract_text(content: types.Content | None) -> str:
    if content is None or not content.parts:
        return ""

    chunks: list[str] = []
    for part in content.parts:
        if getattr(part, "text", None):
            chunks.append(part.text)
    return "".join(chunks).strip()


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

        await session.send(
            input={
                "data": audio_bytes,
                "mime_type": "audio/pcm",
            }
        )


async def _forward_gemini_events(
    websocket: WebSocket,
    session,
    tts: TTSEngine,
) -> None:
    current_transcript = ""
    current_reply = ""

    async for message in session.receive():
        if message.voice_activity is not None:
            voice_activity_type = message.voice_activity.voice_activity_type
            if voice_activity_type == types.VoiceActivityType.ACTIVITY_START:
                await _send_json_safe(websocket, {"type": "voice_activity_start"})
            elif voice_activity_type == types.VoiceActivityType.ACTIVITY_END:
                await _send_json_safe(websocket, {"type": "voice_activity_end"})

        server_content = message.server_content
        if server_content is None:
            continue

        if server_content.input_transcription is not None:
            if server_content.input_transcription.text:
                current_transcript = server_content.input_transcription.text.strip()
                await _send_json_safe(
                    websocket,
                    {
                        "type": "transcript_partial",
                        "text": current_transcript,
                        "finished": bool(server_content.input_transcription.finished),
                    },
                )

        if server_content.model_turn is not None:
            reply_text = _extract_text(server_content.model_turn)
            if reply_text:
                current_reply = reply_text
                await _send_json_safe(
                    websocket,
                    {
                        "type": "reply_partial",
                        "text": current_reply,
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


async def run_live_s2s_session(websocket: WebSocket, tts: TTSEngine) -> None:
    client = create_gemini_client()
    model = os.getenv("GEMINI_LIVE_MODEL", DEFAULT_LIVE_MODEL)
    config = create_live_connect_config()

    await websocket.accept()
    await _send_json_safe(websocket, {"type": "ready"})

    try:
        async with client.aio.live.connect(model=model, config=config) as session:
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
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
