import ShaderOrb, { type AIOrbState } from "@/components/shader-orb"
import { AnimatePresence, motion } from "motion/react"
import { Mic, PhoneOff } from "lucide-react"
import { useEffect, useRef, useState } from "react"

type S2SPhase =
  | "idle"
  | "connecting"
  | "listening"
  | "processing"
  | "speaking"

interface Turn {
  transcript: string
  reply: string
}

type LiveServerEvent =
  | { type: "intro"; reply: string; wav_base64: string }
  | { type: "ready" }
  | { type: "voice_activity_start" }
  | { type: "voice_activity_end" }
  | { type: "transcript_partial"; text: string; finished?: boolean }
  | { type: "reply_partial"; text: string; finished?: boolean }
  | { type: "turn_complete"; transcript: string; reply: string; wav_base64: string }
  | { type: "interrupted" }
  | { type: "error"; message: string }

function toOrbState(phase: S2SPhase): AIOrbState {
  if (phase === "listening") return "listening"
  if (phase === "processing" || phase === "speaking") return "responding"
  return "idle"
}

function statusLabel(phase: S2SPhase, conversationActive: boolean): string {
  if (!conversationActive) return "Dzvanya Taura Live"

  switch (phase) {
    case "idle":
      return "Ndakamirira"
    case "connecting":
      return "Ndiri kubatana..."
    case "listening":
      return "Ndinonzwa..."
    case "processing":
      return "Ndichifunga..."
    case "speaking":
      return "Ndiri kupindura..."
  }
}

function describeMicError(raw: string): string {
  const message = raw.toLowerCase()

  if (message.includes("notallowederror") || message.includes("permission")) {
    return "Microphone permission denied. Allow mic access in your browser/site settings."
  }
  if (message.includes("notfounderror") || message.includes("device not found")) {
    return "No microphone detected. Plug in a mic and verify your OS input device."
  }
  if (message.includes("notreadableerror") || message.includes("could not start audio source")) {
    return "Microphone is busy in another app. Close other recording apps and try again."
  }
  if (message.includes("securityerror") || message.includes("insecure")) {
    return "Microphone access requires a secure origin (localhost or HTTPS)."
  }

  return raw
}

function downsampleToPcm(
  input: Float32Array,
  inputSampleRate: number,
  outputSampleRate = 16000
): Int16Array {
  if (!input.length) return new Int16Array(0)

  if (inputSampleRate === outputSampleRate) {
    const pcm = new Int16Array(input.length)
    for (let i = 0; i < input.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, input[i] ?? 0))
      pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff
    }
    return pcm
  }

  const ratio = inputSampleRate / outputSampleRate
  const outputLength = Math.max(1, Math.round(input.length / ratio))
  const pcm = new Int16Array(outputLength)

  let inputOffset = 0
  for (let outputIndex = 0; outputIndex < outputLength; outputIndex += 1) {
    const nextInputOffset = Math.min(
      input.length,
      Math.round((outputIndex + 1) * ratio)
    )

    let total = 0
    let count = 0
    for (let i = inputOffset; i < nextInputOffset; i += 1) {
      total += input[i] ?? 0
      count += 1
    }

    const sample = count > 0 ? total / count : 0
    const clamped = Math.max(-1, Math.min(1, sample))
    pcm[outputIndex] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff
    inputOffset = nextInputOffset
  }

  return pcm
}

function createLiveSocketUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws"
  const host = `${window.location.hostname}:8000`
  return `${protocol}://${host}/s2s/live`
}

export function S2SPanel() {
  const [phase, setPhase] = useState<S2SPhase>("idle")
  const [conversationActive, setConversationActive] = useState(false)
  const [hasAttemptedStart, setHasAttemptedStart] = useState(false)
  const [activityLevel, setActivityLevel] = useState(0)
  const [turns, setTurns] = useState<Turn[]>([])
  const [currentTranscript, setCurrentTranscript] = useState("")
  const [currentReply, setCurrentReply] = useState("")
  const [error, setError] = useState("")

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const audioBlobUrlRef = useRef<string | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const gainRef = useRef<GainNode | null>(null)
  const socketRef = useRef<WebSocket | null>(null)
  const phaseRef = useRef<S2SPhase>("idle")
  const micUploadEnabledRef = useRef(true)

  useEffect(() => {
    phaseRef.current = phase
  }, [phase])

  useEffect(() => {
    const audio = new Audio()
    audioRef.current = audio
    const onEnded = () => {
      micUploadEnabledRef.current = true
      if (conversationActive) {
        setPhase("idle")
      }
    }
    audio.addEventListener("ended", onEnded)

    return () => {
      audio.removeEventListener("ended", onEnded)
      audio.pause()
      if (audioBlobUrlRef.current) {
        URL.revokeObjectURL(audioBlobUrlRef.current)
      }
    }
  }, [conversationActive])

  useEffect(() => {
    return () => {
      stopSession()
    }
  }, [])

  const stopPlayback = () => {
    audioRef.current?.pause()
    if (audioRef.current) {
      audioRef.current.currentTime = 0
    }
  }

  const cleanupAudioGraph = () => {
    processorRef.current?.disconnect()
    sourceRef.current?.disconnect()
    gainRef.current?.disconnect()
    streamRef.current?.getTracks().forEach((track) => track.stop())
    audioContextRef.current?.close().catch(() => undefined)

    processorRef.current = null
    sourceRef.current = null
    gainRef.current = null
    streamRef.current = null
    audioContextRef.current = null
  }

  const stopSession = () => {
    micUploadEnabledRef.current = false
    socketRef.current?.close()
    socketRef.current = null
    cleanupAudioGraph()
    stopPlayback()
    setActivityLevel(0)
  }

  const playReply = (wavBase64: string) => {
    micUploadEnabledRef.current = false
    const bytes = Uint8Array.from(atob(wavBase64), (char) => char.charCodeAt(0))
    const outputBlob = new Blob([bytes], { type: "audio/wav" })

    if (audioBlobUrlRef.current) {
      URL.revokeObjectURL(audioBlobUrlRef.current)
    }

    const url = URL.createObjectURL(outputBlob)
    audioBlobUrlRef.current = url

    const audio = audioRef.current
    if (!audio) {
      setPhase("idle")
      return
    }

    audio.src = url
    audio.load()
    setPhase("speaking")
    audio.play().catch(() => setPhase("idle"))
  }

  const handleServerEvent = (event: LiveServerEvent) => {
    switch (event.type) {
      case "intro":
        setCurrentReply(event.reply)
        playReply(event.wav_base64)
        return
      case "ready":
        if (phaseRef.current !== "speaking") {
          setCurrentReply("")
          setPhase("idle")
        }
        return
      case "voice_activity_start":
        stopPlayback()
        micUploadEnabledRef.current = true
        setActivityLevel(1)
        setPhase("listening")
        return
      case "voice_activity_end":
        setActivityLevel(0.2)
        if (phaseRef.current !== "speaking") {
          setPhase("processing")
        }
        return
      case "transcript_partial":
        setCurrentTranscript(event.text)
        return
      case "reply_partial":
        setCurrentReply(event.text)
        setPhase("processing")
        return
      case "interrupted":
        stopPlayback()
        setCurrentReply("")
        setPhase("listening")
        return
      case "turn_complete":
        setTurns((prev) => [
          ...prev,
          { transcript: event.transcript, reply: event.reply },
        ])
        setCurrentTranscript("")
        setCurrentReply("")
        playReply(event.wav_base64)
        return
      case "error":
        setError(event.message)
        setPhase("idle")
    }
  }

  const handleStartConversation = async () => {
    if (conversationActive || phase === "connecting") return

    setHasAttemptedStart(true)
    setError("")
    setCurrentTranscript("")
    setCurrentReply("")
    setTurns([])
    setPhase("connecting")

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: { ideal: 1 },
          echoCancellation: { ideal: true },
          noiseSuppression: { ideal: true },
          autoGainControl: { ideal: true },
        },
      })

      const socket = new WebSocket(createLiveSocketUrl())
      socketRef.current = socket
      streamRef.current = stream

      socket.addEventListener("message", (message) => {
        if (typeof message.data !== "string") return

        try {
          handleServerEvent(JSON.parse(message.data) as LiveServerEvent)
        } catch {
          setError("Live response rakundikana.")
          setPhase("idle")
        }
      })

      socket.addEventListener("close", () => {
        cleanupAudioGraph()
        socketRef.current = null
        setConversationActive(false)
        setActivityLevel(0)
        if (phaseRef.current !== "idle") {
          setPhase("idle")
        }
      })

      socket.addEventListener("error", () => {
        setError("Kubatana neLive API kwaramba.")
        setPhase("idle")
      })

      await new Promise<void>((resolve, reject) => {
        socket.addEventListener("open", () => resolve(), { once: true })
        socket.addEventListener("error", () => reject(new Error("Socket open failed")), {
          once: true,
        })
      })

      const audioContext = new AudioContext()
      await audioContext.resume()

      const source = audioContext.createMediaStreamSource(stream)
      const processor = audioContext.createScriptProcessor(4096, 1, 1)
      const gain = audioContext.createGain()
      gain.gain.value = 0

      processor.onaudioprocess = (processEvent) => {
        if (socket.readyState !== WebSocket.OPEN) return
        if (!micUploadEnabledRef.current) return

        const samples = processEvent.inputBuffer.getChannelData(0)
        const pcm = downsampleToPcm(samples, audioContext.sampleRate)
        if (pcm.byteLength > 0) {
          socket.send(pcm.buffer)
        }
      }

      source.connect(processor)
      processor.connect(gain)
      gain.connect(audioContext.destination)

      audioContextRef.current = audioContext
      sourceRef.current = source
      processorRef.current = processor
      gainRef.current = gain

      micUploadEnabledRef.current = false
      setConversationActive(true)
      setPhase("processing")
    } catch (err) {
      stopSession()
      setConversationActive(false)
      setError(
        err instanceof Error
          ? describeMicError(err.message)
          : "Kutadza kuvhura mic."
      )
      setPhase("idle")
    }
  }

  const handleEndConversation = () => {
    stopSession()
    setConversationActive(false)
    setCurrentTranscript("")
    setCurrentReply("")
    setTurns([])
    setError("")
    setPhase("idle")
  }

  const orbState = toOrbState(phase)
  const label = statusLabel(phase, conversationActive)

  return (
    <div className="flex w-full flex-col items-center gap-6">
      {conversationActive && (
        <div className="flex w-full max-w-md justify-end">
          <button
            type="button"
            onClick={handleEndConversation}
            className="inline-flex items-center gap-2 rounded-full border border-stone-200 px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted"
          >
            <PhoneOff className="h-3.5 w-3.5" />
            Pedza hurukuro
          </button>
        </div>
      )}

      <ShaderOrb
        size={280}
        state={orbState}
        activityLevel={activityLevel}
        color={{
          main: "#EDE7FF",
          low: "#85AFFF",
          mid: "#CDAFFA",
          high: "#A4FCF5",
        }}
      />

      <AnimatePresence mode="wait">
        <motion.p
          key={label}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.25 }}
          className="text-sm text-muted-foreground"
        >
          {label}
        </motion.p>
      </AnimatePresence>

      <p className="max-w-sm text-center text-xs text-muted-foreground">
        Gemini Live inonzwa zvakananga, yobva yatumira mhinduro yemavara kuti
        TTS yako yechiShona ibudise izwi.
      </p>

      {error && (
        <p className="max-w-sm text-center text-xs text-red-500">{error}</p>
      )}

      {!conversationActive && (
        <button
          type="button"
          onClick={handleStartConversation}
          className="inline-flex h-11 items-center justify-center gap-2 rounded-full border border-stone-200 px-5 text-sm font-medium text-foreground transition-colors hover:bg-muted"
        >
          <Mic className="h-4 w-4" />
          <span>{hasAttemptedStart ? "Taura Live zvakare" : "Taura Live"}</span>
        </button>
      )}

      {(currentTranscript || currentReply) && (
        <div className="mt-2 flex w-full max-w-md flex-col gap-2 rounded-xl border border-stone-200/60 bg-white/60 p-4 text-sm backdrop-blur-sm">
          {currentTranscript && (
            <p className="text-muted-foreground">
              <span className="font-medium text-foreground">Iwe: </span>
              {currentTranscript}
            </p>
          )}
          {currentReply && (
            <p className="text-muted-foreground">
              <span className="font-medium text-foreground">AI: </span>
              {currentReply}
            </p>
          )}
        </div>
      )}

      {turns.length > 0 && (
        <div className="mt-2 flex w-full max-w-md flex-col gap-4">
          {turns.map((turn, index) => (
            <div
              key={`${turn.transcript}-${index}`}
              className="flex flex-col gap-1 rounded-xl border border-stone-200/60 bg-white/60 p-4 text-sm backdrop-blur-sm"
            >
              {turn.transcript && (
                <p className="text-muted-foreground">
                  <span className="font-medium text-foreground">Iwe: </span>
                  {turn.transcript}
                </p>
              )}
              {turn.reply && (
                <p className="text-muted-foreground">
                  <span className="font-medium text-foreground">AI: </span>
                  {turn.reply}
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
