import { createServerFn } from "@tanstack/react-start"

export const speechToSpeech = createServerFn({ method: "POST" })
  .inputValidator((data: FormData) => data)
  .handler(async ({ data }) => {
    const res = await fetch("http://localhost:8000/s2s", {
      method: "POST",
      body: data,
    })

    if (!res.ok) {
      const err = await res.text()
      throw new Error(`S2S server error ${res.status}: ${err}`)
    }

    const payload = (await res.json()) as {
      wav_base64: string
      transcript: string
      reply: string
    }

    return {
      wavBase64: payload.wav_base64,
      transcript: payload.transcript ?? "",
      reply: payload.reply ?? "",
    }
  })
