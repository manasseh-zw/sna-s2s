import ShaderOrb, { type AIOrbState } from "@/components/shader-orb"
import { Button } from "@/components/ui/button"
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ui/conversation"
import { Dialog, DialogContent } from "@/components/ui/dialog"
import { Message, MessageContent } from "@/components/ui/message"
import { createLiveKitSession } from "@/lib/actions/livekit"
import {
  ConnectionState,
  Room,
  RoomEvent,
  Track,
  type Participant,
  type RemoteTrack,
  type RemoteTrackPublication,
} from "livekit-client"
import { AnimatePresence, motion } from "motion/react"
import { Mic, PhoneOff } from "lucide-react"
import { useEffect, useRef, useState } from "react"

type S2SPhase = "idle" | "connecting" | "listening" | "processing" | "speaking"

type ConversationMessageStatus = "final" | "pending"

interface ConversationMessage {
  id: string
  role: "user" | "assistant"
  text: string
  status: ConversationMessageStatus
}

type RoomTurnEventType =
  | "conversation_ready"
  | "user_turn_final"
  | "assistant_turn_started"
  | "assistant_turn_final"

interface RoomTurnEventPayload {
  type: RoomTurnEventType
  id: string
  text?: string
  timestamp?: number
}

const INTRO_AUDIO_PATH = "/livekit-intro.wav"
const INTRO_MESSAGE =
  "Mhoro unogona kutaura zvino, uye ndichakupindura nechishona chakareruka"
const TURN_EVENT_TOPIC = "s2s-turn"

function toOrbState(phase: S2SPhase): AIOrbState {
  if (phase === "listening") return "listening"
  if (phase === "processing" || phase === "speaking") return "responding"
  return "idle"
}

function statusLabel(phase: S2SPhase, conversationActive: boolean): string {
  if (!conversationActive) return "Dzvanya Taura Kuti Utange Hurukuro"

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
  if (
    message.includes("notfounderror") ||
    message.includes("device not found")
  ) {
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

function isRoomTurnEventPayload(value: unknown): value is RoomTurnEventPayload {
  if (!value || typeof value !== "object") return false

  const event = value as Partial<RoomTurnEventPayload>
  return (
    typeof event.type === "string" &&
    typeof event.id === "string" &&
    (event.text === undefined || typeof event.text === "string") &&
    (event.timestamp === undefined || typeof event.timestamp === "number")
  )
}

export function S2SPanel() {
  const [phase, setPhase] = useState<S2SPhase>("idle")
  const [conversationActive, setConversationActive] = useState(false)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [hasAttemptedStart, setHasAttemptedStart] = useState(false)
  const [activityLevel, setActivityLevel] = useState(0)
  const [error, setError] = useState("")
  const [messages, setMessages] = useState<ConversationMessage[]>([])
  const [assistantPending, setAssistantPending] = useState(false)

  const roomRef = useRef<Room | null>(null)
  const remoteAudioElementsRef = useRef<Map<string, HTMLAudioElement>>(
    new Map()
  )
  const introAudioRef = useRef<HTMLAudioElement | null>(null)
  const phaseRef = useRef<S2SPhase>("idle")
  const conversationActiveRef = useRef(false)
  const remoteSpeakingRef = useRef(false)
  const localSpeakingRef = useRef(false)
  const suppressDialogCloseRef = useRef(false)
  const textDecoderRef = useRef(new TextDecoder())

  useEffect(() => {
    phaseRef.current = phase
  }, [phase])

  useEffect(() => {
    conversationActiveRef.current = conversationActive
  }, [conversationActive])

  useEffect(() => {
    return () => {
      void stopSession({ closeDialog: false })
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

    setActivityLevel(conversationActiveRef.current ? 0.15 : 0)
    setPhase("idle")
  }

  const resetConversationState = () => {
    setMessages([
      {
        id: "intro-message",
        role: "assistant",
        text: INTRO_MESSAGE,
        status: "final",
      },
    ])
    setAssistantPending(false)
  }

  const appendMessage = (message: ConversationMessage) => {
    setMessages((current) => {
      if (current.some((entry) => entry.id === message.id)) {
        return current
      }

      return [...current, message]
    })
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

  const stopSession = async ({ closeDialog = true } = {}) => {
    clearRemoteAudio()
    stopIntroAudio()
    remoteSpeakingRef.current = false
    localSpeakingRef.current = false
    conversationActiveRef.current = false

    const room = roomRef.current
    roomRef.current = null
    if (room) {
      room.removeAllListeners()
      await room.disconnect()
    }

    setConversationActive(false)
    resetConversationState()
    setActivityLevel(0)
    setPhase("idle")

    if (closeDialog) {
      suppressDialogCloseRef.current = true
      setDialogOpen(false)
    }
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

    const trackKey =
      publication.trackSid || track.sid || `${participant.identity}-audio`
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
    const trackKey =
      publication.trackSid || track.sid || `${participant.identity}-audio`
    const element = remoteAudioElementsRef.current.get(trackKey)
    if (!element) return

    track.detach(element)
    element.remove()
    remoteAudioElementsRef.current.delete(trackKey)
  }

  const handleRoomDataReceived = (payload: Uint8Array, topic?: string) => {
    if (topic !== TURN_EVENT_TOPIC) return

    try {
      const text = textDecoderRef.current.decode(payload)
      const parsed = JSON.parse(text) as unknown
      if (!isRoomTurnEventPayload(parsed)) return

      switch (parsed.type) {
        case "conversation_ready":
          return
        case "assistant_turn_started":
          setAssistantPending(true)
          return
        case "user_turn_final":
          if (!parsed.text) return
          appendMessage({
            id: parsed.id,
            role: "user",
            text: parsed.text,
            status: "final",
          })
          return
        case "assistant_turn_final":
          if (!parsed.text) return
          setAssistantPending(false)
          appendMessage({
            id: parsed.id,
            role: "assistant",
            text: parsed.text,
            status: "final",
          })
          return
      }
    } catch {
      // Ignore malformed data packets from the room.
    }
  }

  const handleStartConversation = async () => {
    if (conversationActiveRef.current || phaseRef.current === "connecting") {
      return
    }

    setHasAttemptedStart(true)
    setError("")
    resetConversationState()
    setDialogOpen(true)
    setConversationActive(true)
    conversationActiveRef.current = true

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
          conversationActiveRef.current = true
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
      room.on(
        RoomEvent.DataReceived,
        (payload: Uint8Array, _participant, _kind, topic?: string) => {
          handleRoomDataReceived(payload, topic)
        }
      )

      setPhase("connecting")
      await room.connect(session.url, session.token, { autoSubscribe: true })
      await room.startAudio()

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

  const handleDialogOpenChange = (open: boolean) => {
    if (!open && suppressDialogCloseRef.current) {
      suppressDialogCloseRef.current = false
      setDialogOpen(false)
      return
    }

    setDialogOpen(open)

    if (!open) {
      void stopSession({ closeDialog: false })
      setError("")
    }
  }

  const orbState = toOrbState(phase)
  const label = statusLabel(phase, conversationActive)

  return (
    <>
      <div className="flex w-full flex-col items-center gap-6">
        <ShaderOrb
          size={280}
          state={orbState}
          activityLevel={activityLevel}
          color={{
            main: "#F2F7FF",
            low: "#5B8CFF",
            mid: "#8DB5FF",
            high: "#DCEBFF",
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

        {error && !dialogOpen && (
          <p className="max-w-sm text-center text-xs text-red-500">{error}</p>
        )}

        {!conversationActive && (
          <button
            type="button"
            onClick={handleStartConversation}
            className="inline-flex h-11 items-center justify-center gap-2 rounded-full border border-stone-200 px-5 text-sm font-medium text-foreground transition-colors hover:bg-muted"
          >
            <Mic className="h-4 w-4" />
            <span>{hasAttemptedStart ? "Taura zvakare" : "Taura"}</span>
          </button>
        )}
      </div>

      <Dialog open={dialogOpen} onOpenChange={handleDialogOpenChange}>
        <DialogContent
          fullscreen
          showCloseButton={false}
          className="gap-0 overflow-hidden p-0"
        >
          <div className="flex min-h-0 flex-1 flex-col bg-background p-4 sm:p-6">
            <div className="flex items-center justify-between gap-4 px-1 sm:px-2">
              <div className="flex items-center gap-4">
                <ShaderOrb
                  size={72}
                  state={orbState}
                  activityLevel={activityLevel}
                  color={{
                    main: "#F2F7FF",
                    low: "#5B8CFF",
                    mid: "#8DB5FF",
                    high: "#DCEBFF",
                  }}
                />

                <AnimatePresence mode="wait">
                  <motion.p
                    key={`dialog-${label}`}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -6 }}
                    transition={{ duration: 0.25 }}
                    className="text-sm text-muted-foreground"
                  >
                    {label}
                  </motion.p>
                </AnimatePresence>
              </div>

              <Button
                type="button"
                variant="outline"
                size="lg"
                onClick={handleEndConversation}
                className="h-12 rounded-full px-5"
              >
                <PhoneOff className="size-5" />
                Pedza hurukuro
              </Button>
            </div>

            <div className="min-h-0 flex-1 pt-5">
              <Conversation className="h-full rounded-[32px] border border-stone-200/70 bg-white">
                <ConversationContent className="mx-auto flex min-h-full w-full max-w-4xl flex-col gap-1 px-4 py-6 sm:px-8">
                  {messages.map((message) => (
                    <Message from={message.role} key={message.id}>
                      <MessageContent>
                        <p className="whitespace-pre-wrap">{message.text}</p>
                      </MessageContent>
                    </Message>
                  ))}
                  {assistantPending && (
                    <Message from="assistant" key="assistant-pending">
                      <MessageContent>
                        <p className="text-muted-foreground">
                          {phase === "speaking"
                            ? "Ndiri kupindura..."
                            : "Ndichifunga..."}
                        </p>
                      </MessageContent>
                    </Message>
                  )}
                </ConversationContent>
                <ConversationScrollButton className="shadow-none" />
              </Conversation>
            </div>

            {error && (
              <p className="pt-4 text-center text-xs text-red-500">{error}</p>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
}
