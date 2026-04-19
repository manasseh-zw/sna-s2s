import ShaderOrb, { type AIOrbState } from "@/components/shader-orb"
import { createLiveKitSession } from "@/lib/actions/livekit"
import { AnimatePresence, motion } from "motion/react"
import { Mic, PhoneOff } from "lucide-react"
import {
  ConnectionState,
  Room,
  RoomEvent,
  Track,
  type Participant,
  type RemoteTrack,
  type RemoteTrackPublication,
} from "livekit-client"
import { useEffect, useRef, useState } from "react"

type S2SPhase =
  | "idle"
  | "connecting"
  | "listening"
  | "processing"
  | "speaking"

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
  if (
    message.includes("notreadableerror") ||
    message.includes("could not start audio source")
  ) {
    return "Microphone is busy in another app. Close other recording apps and try again."
  }
  if (message.includes("securityerror") || message.includes("insecure")) {
    return "Microphone access requires a secure origin (localhost or HTTPS)."
  }

  return raw
}

interface SessionMeta {
  roomName: string
  participantIdentity: string
}

const INTRO_AUDIO_PATH = "/livekit-intro.wav"

export function S2SPanel() {
  const [phase, setPhase] = useState<S2SPhase>("idle")
  const [conversationActive, setConversationActive] = useState(false)
  const [hasAttemptedStart, setHasAttemptedStart] = useState(false)
  const [activityLevel, setActivityLevel] = useState(0)
  const [error, setError] = useState("")
  const [sessionMeta, setSessionMeta] = useState<SessionMeta | null>(null)

  const roomRef = useRef<Room | null>(null)
  const remoteAudioElementsRef = useRef<Map<string, HTMLAudioElement>>(new Map())
  const introAudioRef = useRef<HTMLAudioElement | null>(null)
  const phaseRef = useRef<S2SPhase>("idle")
  const remoteSpeakingRef = useRef(false)
  const localSpeakingRef = useRef(false)

  useEffect(() => {
    phaseRef.current = phase
  }, [phase])

  useEffect(() => {
    return () => {
      void stopSession()
    }
  }, [])

  const syncPhaseFromSpeakers = () => {
    if (remoteSpeakingRef.current) {
      setActivityLevel(0.92)
      setPhase("speaking")
      return
    }

    if (localSpeakingRef.current) {
      setActivityLevel(1)
      setPhase("listening")
      return
    }

    setActivityLevel(conversationActive ? 0.15 : 0)
    if (conversationActive) {
      setPhase("idle")
    } else {
      setPhase("idle")
    }
  }

  const clearRemoteAudio = () => {
    remoteAudioElementsRef.current.forEach((element) => {
      element.pause()
      element.remove()
    })
    remoteAudioElementsRef.current.clear()
  }

  const stopIntroAudio = () => {
    introAudioRef.current?.pause()
    if (introAudioRef.current) {
      introAudioRef.current.currentTime = 0
    }
  }

  const stopSession = async () => {
    clearRemoteAudio()
    stopIntroAudio()
    remoteSpeakingRef.current = false
    localSpeakingRef.current = false

    const room = roomRef.current
    roomRef.current = null
    if (room) {
      room.removeAllListeners()
      await room.disconnect()
    }

    setConversationActive(false)
    setSessionMeta(null)
    setActivityLevel(0)
    setPhase("idle")
  }

  const playIntroGreeting = async () => {
    let audio = introAudioRef.current
    if (!audio) {
      audio = new Audio(INTRO_AUDIO_PATH)
      audio.preload = "auto"
      introAudioRef.current = audio
    }

    setActivityLevel(0.88)
    setPhase("speaking")

    try {
      audio.currentTime = 0
      await audio.play()
      await new Promise<void>((resolve, reject) => {
        const onEnded = () => {
          cleanup()
          resolve()
        }
        const onError = () => {
          cleanup()
          reject(new Error("Intro audio failed to play."))
        }
        const cleanup = () => {
          audio?.removeEventListener("ended", onEnded)
          audio?.removeEventListener("error", onError)
        }

        audio?.addEventListener("ended", onEnded, { once: true })
        audio?.addEventListener("error", onError, { once: true })
      })
    } catch {
      // Fall back to immediate startup if the asset fails to play.
    }
  }

  const handleTrackSubscribed = (
    track: RemoteTrack,
    publication: RemoteTrackPublication,
    participant: Participant
  ) => {
    if (track.kind !== Track.Kind.Audio || participant.isLocal) {
      return
    }

    const trackKey = publication.trackSid || track.sid || `${participant.identity}-audio`
    const element = track.attach() as HTMLAudioElement
    element.autoplay = true
    element.dataset.participantIdentity = participant.identity
    element.style.display = "none"
    document.body.appendChild(element)
    remoteAudioElementsRef.current.set(trackKey, element)
  }

  const handleTrackUnsubscribed = (
    track: RemoteTrack,
    publication: RemoteTrackPublication,
    participant: Participant
  ) => {
    const trackKey = publication.trackSid || track.sid || `${participant.identity}-audio`
    const element = remoteAudioElementsRef.current.get(trackKey)
    if (!element) return

    track.detach(element)
    element.remove()
    remoteAudioElementsRef.current.delete(trackKey)
  }

  const handleStartConversation = async () => {
    if (conversationActive || phase === "connecting") return

    setHasAttemptedStart(true)
    setError("")
    setSessionMeta(null)
    setConversationActive(true)

    const introPromise = playIntroGreeting()

    try {
      const session = await createLiveKitSession()
      const room = new Room()
      roomRef.current = room

      room.on(RoomEvent.TrackSubscribed, handleTrackSubscribed)
      room.on(RoomEvent.TrackUnsubscribed, handleTrackUnsubscribed)
      room.on(RoomEvent.ConnectionStateChanged, (state: ConnectionState) => {
        if (state === ConnectionState.Connected) {
          setConversationActive(true)
          if (phaseRef.current !== "speaking") {
            setActivityLevel(0.15)
            setPhase("idle")
          }
        }

        if (state === ConnectionState.Disconnected) {
          void stopSession()
        }
      })
      room.on(RoomEvent.ActiveSpeakersChanged, (speakers: Participant[]) => {
        localSpeakingRef.current = speakers.some(
          (speaker) => speaker.identity === room.localParticipant.identity
        )
        remoteSpeakingRef.current = speakers.some(
          (speaker) => speaker.identity !== room.localParticipant.identity
        )

        if (!localSpeakingRef.current && !remoteSpeakingRef.current) {
          setActivityLevel(0.25)
          if (phaseRef.current === "listening") {
            setPhase("processing")
            return
          }
        }

        syncPhaseFromSpeakers()
      })
      room.on(RoomEvent.MediaDevicesError, (err: Error) => {
        setError(describeMicError(err.message))
      })

      setPhase("connecting")
      await room.connect(session.url, session.token, { autoSubscribe: true })
      await room.startAudio()

      setSessionMeta({
        roomName: session.room_name,
        participantIdentity: session.participant_identity,
      })
      await introPromise
      await room.localParticipant.setMicrophoneEnabled(true)
      setActivityLevel(0.15)
      setPhase("idle")
    } catch (err) {
      await stopSession()
      setError(
        err instanceof Error
          ? describeMicError(err.message)
          : "Kutadza kubatana neLiveKit."
      )
    }
  }

  const handleEndConversation = () => {
    void stopSession()
    setError("")
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
        LiveKit iri kubata room, mic, uye turn-taking. Chain yako ichiri
        kushandisa ASR yako, Gemini Flash, neTTS yako yechiShona.
      </p>

      {sessionMeta && (
        <div className="w-full max-w-md rounded-xl border border-stone-200/60 bg-white/60 p-4 text-xs text-muted-foreground backdrop-blur-sm">
          <p>
            <span className="font-medium text-foreground">Room: </span>
            {sessionMeta.roomName}
          </p>
          <p>
            <span className="font-medium text-foreground">Identity: </span>
            {sessionMeta.participantIdentity}
          </p>
        </div>
      )}

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
    </div>
  )
}
