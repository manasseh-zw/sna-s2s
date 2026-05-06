import { createServerFn } from "@tanstack/react-start"

export const synthesizeSpeech = createServerFn({ method: "POST" })
  .inputValidator((data: { text: string; voice: string }) => data)
  .handler(async ({ data }) => {
    const res = await fetch("http://localhost:8000/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    })

    if (!res.ok) {
      const err = await res.text()
      throw new Error(`TTS server error ${res.status}: ${err}`)
    }

    const arrayBuffer = await res.arrayBuffer()
    // Serialize across the server→client boundary as base64
    const base64 = Buffer.from(arrayBuffer).toString("base64")
    return base64
  })
