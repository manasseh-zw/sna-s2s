interface LiveKitSessionResponse {
  token: string
  url: string
  room_name: string
  participant_identity: string
}

export async function createLiveKitSession(roomName?: string) {
  const res = await fetch("http://localhost:8000/livekit/token", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(roomName ? { room_name: roomName } : {}),
  })

  if (!res.ok) {
    const detail = await res.text()
    throw new Error(detail || "Failed to create LiveKit session.")
  }

  return (await res.json()) as LiveKitSessionResponse
}
