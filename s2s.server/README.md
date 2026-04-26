# S2S Server

## LLM backend config

The server reads `.env` through the existing `uv run --env-file .env ...` commands in the repo root.

Default online mode uses Gemini:

```env
LLM_BACKEND=gemini
GEMINI_API_KEY=your-key
GEMINI_TEXT_MODEL=gemini-3.1-flash-lite-preview
```

Offline/local mode can point at a local OpenAI-compatible server such as LM Studio:

```env
LLM_BACKEND=local
LOCAL_LLM_MODEL=tiny-aya-earth
LOCAL_LLM_BASE_URL=http://127.0.0.1:1234/v1
LOCAL_LLM_API_KEY=not-needed
```

If you use Ollama instead of LM Studio, keep `LLM_BACKEND=local` and change only:

```env
LOCAL_LLM_MODEL=tiny-aya-earth
LOCAL_LLM_BASE_URL=http://127.0.0.1:11434/v1
```

When the API starts it now logs which backend and model it connected to, so it is easy to confirm the active mode.
