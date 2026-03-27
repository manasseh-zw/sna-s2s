import { createServerFn } from "@tanstack/react-start"

export const transcribeAudio = createServerFn({ method: "POST" })
  .inputValidator((data: FormData) => data)
  .handler(async ({ data }) => {
    const res = await fetch("http://localhost:8000/asr", {
      method: "POST",
      body: data,
    })

    if (!res.ok) {
      const err = await res.text()
      throw new Error(`ASR server error ${res.status}: ${err}`)
    }

    const json = await res.json()
    return json.text as string
  })
