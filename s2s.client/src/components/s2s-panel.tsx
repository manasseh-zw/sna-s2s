
import ShaderOrb, { type AIOrbState } from "@/components/shader-orb"
import { speechToSpeech } from "@/lib/actions/s2s"
import { resetConversation } from "@/lib/actions/s2s-reset"
import { useMicVAD, utils } from "@ricky0123/vad-react"
import { AnimatePresence, motion } from "motion/react"
import { Mic, PhoneOff } from "lucide-react"
import { useCallback, useEffect, useRef, useState } from "react"

// ── Types ─────────────────────────────────────────────────────────────────────

type S2SPhase = "idle" | "listening" | "processing" | "speaking"

interface Turn {
  transcript: string
  reply: string
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Map pipeline phase → ShaderOrb state */
function toOrbState(phase: S2SPhase): AIOrbState {
  if (phase === "listening") return "listening"
  if (phase === "processing" || phase === "speaking") return "responding"
  return "idle"
}

/** Status label shown below the orb */
function statusLabel(phase: S2SPhase, conversationActive: boolean): string {
  if (!conversationActive) return "Dzvanya Taura"

  switch (phase) {
    case "idle":
      return "Taura"
    case "listening":
      return "Ndinonzwa..."
    case "processing":
      return "Ndichifunga..."
    case "speaking":
      return "..."
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
  if (message.includes("null stream") || message.includes("audio context") || message.includes("processor adapter")) {
    return "Mic init yakundikana. Dzvanya Taura zvakare kuti tiedze zvakare."
  }

  return raw
}

function hasEnoughSpeech(audio: Float32Array): boolean {
  const minSamples = 16000 * 0.45
  if (audio.length < minSamples) return false

  let sumSquares = 0
  let peak = 0
  for (let i = 0; i < audio.length; i += 1) {
    const v = Math.abs(audio[i] ?? 0)
    sumSquares += v * v
    if (v > peak) peak = v
  }

  const rms = Math.sqrt(sumSquares / audio.length)
  return rms > 0.008 || peak > 0.04
}

// ── Component ─────────────────────────────────────────────────────────────────

export function S2SPanel() {
  const [phase, setPhase] = useState<S2SPhase>("idle")
  const [conversationActive, setConversationActive] = useState(false)
  const [hasAttemptedStart, setHasAttemptedStart] = useState(false)
  const [activityLevel, setActivityLevel] = useState(0)
  const [turns, setTurns] = useState<Turn[]>([])
  const [error, setError] = useState("")
  const [isResetting, setIsResetting] = useState(false)

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const audioBlobUrlRef = useRef<string | null>(null)
  const phaseRef = useRef<S2SPhase>("idle")
  const activityRef = useRef(0)
  const [vadModel, setVadModel] = useState<"legacy" | "v5">("legacy")

  // Keep ref in sync so callbacks always see latest phase
  useEffect(() => {
    phaseRef.current = phase
  }, [phase])

  // ── Audio element events ──────────────────────────────────────────────────

  useEffect(() => {
    const audio = new Audio()
    audioRef.current = audio
    const onEnded = () => setPhase("idle")
    audio.addEventListener("ended", onEnded)
    return () => {
      audio.removeEventListener("ended", onEnded)
      audio.pause()
      if (audioBlobUrlRef.current) {
        URL.revokeObjectURL(audioBlobUrlRef.current)
      }
    }
  }, [])

  // ── speech-end handler ────────────────────────────────────────────────────

  const onSpeechEnd = useCallback(
    async (float32Audio: Float32Array) => {
      // Guard: ignore if we're already processing or speaking
      if (phaseRef.current !== "listening") return

      if (!hasEnoughSpeech(float32Audio)) {
        setError("Handina kunzwa kutaura. Dzvanya Taura, woedza zvakare.")
        setActivityLevel(0)
        activityRef.current = 0
        setPhase("idle")
        return
      }

      setPhase("processing")
      setActivityLevel(0)
      activityRef.current = 0
      setError("")

      try {
        // Convert Float32Array (16 kHz PCM) → WAV Blob
        const wavBuffer = utils.encodeWAV(float32Audio)
        const wavBlob = new Blob([wavBuffer], { type: "audio/wav" })

        const formData = new FormData()
        formData.append("file", wavBlob, "speech.wav")

        const { wavBase64, transcript, reply } = await speechToSpeech({
          data: formData,
        })

        setTurns((prev) => [...prev, { transcript, reply }])

        // Decode base64 WAV and play
        const bytes = Uint8Array.from(atob(wavBase64), (c) => c.charCodeAt(0))
        const outputBlob = new Blob([bytes], { type: "audio/wav" })

        if (audioBlobUrlRef.current) {
          URL.revokeObjectURL(audioBlobUrlRef.current)
        }
        const url = URL.createObjectURL(outputBlob)
        audioBlobUrlRef.current = url

        const audio = audioRef.current
        if (audio) {
          audio.src = url
          audio.load()
          setPhase("speaking")
          audio.play().catch(() => setPhase("idle"))
        } else {
          setPhase("idle")
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : "Something went wrong."
        if (message.includes("S2S server error 422") && message.toLowerCase().includes("no speech detected")) {
          setError("Handina kunzwa kutaura. Dzvanya Taura, woedza zvakare.")
        } else {
          setError(message)
        }
        setPhase("idle")
      }
    },
    [] // no deps needed — we read phase via ref
  )

  // ── VAD ──────────────────────────────────────────────────────────────────

  const vad = useMicVAD({
    startOnLoad: false,
    model: vadModel,
    getStream: async () => {
      try {
        return await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: { ideal: 1 },
            echoCancellation: { ideal: true },
            noiseSuppression: { ideal: true },
            autoGainControl: { ideal: true },
          },
        })
      } catch {
        // Fallback for browsers/devices that reject stricter constraints.
        return navigator.mediaDevices.getUserMedia({ audio: true })
      }
    },
    onSpeechStart: () => {
      if (!conversationActive) return
      // Only start listening if we're idle (not already processing/speaking)
      if (phaseRef.current === "idle") {
        setPhase("listening")
        setError("")
      }
    },
    onSpeechEnd,
    onFrameProcessed: (probabilities) => {
      // Drive the orb's activityLevel from voice probability
      if (phaseRef.current === "listening") {
        const raw = probabilities.isSpeech
        const target = raw < 0.05 ? 0 : raw
        const smoothed = activityRef.current * 0.82 + target * 0.18
        activityRef.current = smoothed
        setActivityLevel(smoothed)
      }
    },
    // VAD assets are served from /public, while ORT runtime artifacts are loaded
    // from onnxruntime-web's dist folder during dev.
    baseAssetPath: "/",
    onnxWASMBasePath: "/node_modules/onnxruntime-web/dist/",
    ortConfig: (ort) => {
      // Avoid threaded worker bootstrapping issues in local dev environments.
      ort.env.wasm.numThreads = 1
    },
    // Silence threshold: ~700 ms quiet = speech end
    positiveSpeechThreshold: 0.5,
    negativeSpeechThreshold: 0.35,
    redemptionMs: 700,
    minSpeechMs: 150,
    preSpeechPadMs: 300,
  })

  // ── Reset ─────────────────────────────────────────────────────────────────

  useEffect(() => {
    if (!vad.errored) return
    setConversationActive(false)
    setPhase("idle")
    setActivityLevel(0)
    activityRef.current = 0
  }, [vad.errored])

  const handleStartConversation = async () => {
    if (vad.loading || phase === "processing" || phase === "speaking") return
    setHasAttemptedStart(true)

    if (vad.errored) {
      setError("")
      setVadModel((prev) => (prev === "legacy" ? "v5" : "legacy"))
      return
    }

    try {
      setError("")
      setPhase("idle")
      await vad.start()
      setConversationActive(true)
    } catch (err) {
      setConversationActive(false)
      setError(err instanceof Error ? describeMicError(err.message) : "Kutadza kuvhura mic.")
    }
  }

  const handleEndConversation = async () => {
    if (isResetting) return
    setIsResetting(true)
    try {
      if (vad.listening) {
        await vad.pause()
      }
      audioRef.current?.pause()
      if (audioRef.current) audioRef.current.currentTime = 0
      await resetConversation()
      setConversationActive(false)
      setTurns([])
      setError("")
      setActivityLevel(0)
      activityRef.current = 0
      setPhase("idle")
    } catch {
      setError("Kutadza kupedza hurukuro.")
    } finally {
      setIsResetting(false)
    }
  }

  const handleRetryMic = () => {
    setError("")
    setHasAttemptedStart(true)
    setConversationActive(false)
    setActivityLevel(0)
    activityRef.current = 0
    setPhase("idle")
    setVadModel((prev) => (prev === "legacy" ? "v5" : "legacy"))
  }

  // ── Render ────────────────────────────────────────────────────────────────

  const orbState = toOrbState(phase)
  const showMicError = hasAttemptedStart && Boolean(vad.errored)
  const label = vad.loading
    ? "Ndinotanga..."
    : showMicError
      ? "Mic yakundikana"
      : statusLabel(phase, conversationActive)

  return (
    <div className="flex w-full flex-col items-center gap-6">
      {(conversationActive || turns.length > 0) && (
        <div className="flex w-full max-w-md justify-end">
          <button
            type="button"
            onClick={handleEndConversation}
            disabled={isResetting || phase === "processing" || phase === "speaking"}
            className="inline-flex items-center gap-2 rounded-full border border-stone-200 px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-40"
          >
            <PhoneOff className="h-3.5 w-3.5" />
            Pedza hurukuro
          </button>
        </div>
      )}

      {/* Orb */}
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

      {/* Status label */}
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

      {/* Error */}
      {error && (
        <p className="max-w-sm text-center text-xs text-red-500">{error}</p>
      )}
      {showMicError && !error && vad.errored && (
        <p className="max-w-sm text-center text-xs text-red-500">
          {describeMicError(vad.errored)}
        </p>
      )}

      {showMicError && (
        <button
          type="button"
          onClick={handleRetryMic}
          className="inline-flex h-10 items-center justify-center rounded-full border border-stone-200 px-4 text-xs text-muted-foreground transition-colors hover:bg-muted"
        >
          Edza mic zvakare
        </button>
      )}

      {!conversationActive && !showMicError && (
        <button
          type="button"
          onClick={handleStartConversation}
          disabled={vad.loading || isResetting}
          className="inline-flex h-11 items-center justify-center gap-2 rounded-full border border-stone-200 px-5 text-sm font-medium text-foreground transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-40"
        >
          <Mic className="h-4 w-4" />
          <span>Taura</span>
        </button>
      )}

      {/* Conversation history */}
      {turns.length > 0 && (
        <div className="mt-2 flex w-full max-w-md flex-col gap-4">
          {turns.map((turn, i) => (
            <div key={i} className="flex flex-col gap-1 rounded-xl border border-stone-200/60 bg-white/60 p-4 text-sm backdrop-blur-sm">
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
