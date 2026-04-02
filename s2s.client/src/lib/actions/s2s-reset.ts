import { createServerFn } from "@tanstack/react-start"

export const resetConversation = createServerFn({ method: "POST" }).handler(
  async () => {
    const res = await fetch("http://localhost:8000/s2s/reset", {
      method: "POST",
    })

    if (!res.ok) {
      const err = await res.text()
      throw new Error(`Reset error ${res.status}: ${err}`)
    }

    return { ok: true }
  }
)
