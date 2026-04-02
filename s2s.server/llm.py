"""Gemini LLM client for S2S assistant responses."""

import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"

_SYSTEM_PROMPT = (
    "Iwe uri mubatsiri anopindura nechiShona chakareruka chete. "
    "Usashandise Chirungu. Usashandise manhamba kana digit. "
    "Mhinduro ngadzive pfupi, asi dzinogona kuwedzera zvishoma kana mubvunzo wada kutsanangurwa. "
    "Kana transcript iine zvikanganiso, edza kufungidzira zvinorehwa wobatsira zvine hungwaru. "
    "Kana zvisinganzwisisike zvachose, kumbira mushandisi adzokorore mubvunzo wake."
)


class LLMClient:
    """Calls Gemini model directly via google-genai SDK."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        self._model = model
        self._client = genai.Client(api_key=api_key)
        self._history: list[types.Content] = []

    def respond(self, text: str, *, maintain_context: bool = True) -> str:
        """Send a Shona utterance and return a short Shona reply."""
        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=text)],
        )

        contents: list[types.Content] = []
        if maintain_context:
            contents.extend(self._history)
        contents.append(user_content)

        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=0.6,
            max_output_tokens=140,
            thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        reply = (response.text or "").strip()
        if not reply:
            raise RuntimeError("No text in Gemini response.")

        # Safety cleanup: never return digits.
        reply = re.sub(r"\d+", "", reply)
        reply = re.sub(r"\s{2,}", " ", reply).strip()

        if maintain_context:
            self._history.append(user_content)
            self._history.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=reply)],
                )
            )

        return reply

    def reset_context(self) -> None:
        """Clear conversation history (start a fresh S2S session)."""
        self._history = []
