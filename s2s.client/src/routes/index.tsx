import { S2SPanel } from "@/components/s2s-panel"
import { Tabs, TabsList, TabsPanel, TabsTab } from "@/components/ui/tabs"
import { TextEffect } from "@/components/ui/text-effect"
import { TextShimmer } from "@/components/ui/text-shimmer"
import { ScrollingWaveform, StaticWaveform } from "@/components/ui/waveform"
import { VoiceInput } from "@/components/voice-input"
import { transcribeAudio } from "@/lib/actions/asr"
import { synthesizeSpeech } from "@/lib/actions/tts"
import { createFileRoute } from "@tanstack/react-router"
import {
  ArrowLeftRight,
  Mic,
  Play,
  RotateCcw,
  Upload,
  Volume2,
} from "lucide-react"
import { useEffect, useRef, useState } from "react"

export const Route = createFileRoute("/")({ component: App, ssr: false })

function App() {
  const [asrState, setAsrState] = useState<
    "idle" | "listening" | "transcribing" | "done" | "error"
  >("idle")
  const [transcript, setTranscript] = useState("")
  const [asrError, setAsrError] = useState("")
  const [isUploading, setIsUploading] = useState(false)
  const uploadInputRef = useRef<HTMLInputElement | null>(null)

  const [ttsText, setTtsText] = useState("")
  const [ttsPhase, setTtsPhase] = useState<"idle" | "processing" | "result">(
    "idle"
  )
  const [ttsPlaying, setTtsPlaying] = useState(false)
  const [ttsSeed, setTtsSeed] = useState(42)
  const [ttsError, setTtsError] = useState("")

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const audioBlobUrlRef = useRef<string | null>(null)

  // ── ASR ────────────────────────────────────────────────────────────────

  const handleAsrStart = () => {
    setTranscript("")
    setAsrError("")
    setAsrState("listening")
  }

  const handleAsrStop = async (blob: Blob) => {
    setAsrState("transcribing")

    try {
      const formData = new FormData()
      formData.append("file", blob, "recording.webm")

      const text = await transcribeAudio({ data: formData })
      setTranscript(text)
      setAsrState("done")
    } catch (err) {
      setAsrError(err instanceof Error ? err.message : "Transcription failed.")
      setAsrState("error")
    }
  }

  const handleAsrReset = () => {
    setTranscript("")
    setAsrError("")
    setAsrState("idle")
  }

  const handleUploadClick = () => {
    uploadInputRef.current?.click()
  }

  const handleAsrUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0]
    if (!file) return

    setIsUploading(true)
    setAsrError("")
    setTranscript("")
    setAsrState("transcribing")

    try {
      const formData = new FormData()
      formData.append("file", file, file.name || "upload.mp3")

      const text = await transcribeAudio({ data: formData })
      setTranscript(text)
      setAsrState("done")
    } catch (err) {
      setAsrError(
        err instanceof Error ? err.message : "Upload transcription failed."
      )
      setAsrState("error")
    } finally {
      setIsUploading(false)
      event.target.value = ""
    }
  }

  // ── TTS ────────────────────────────────────────────────────────────────

  const startTts = async () => {
    if (ttsPhase === "processing") return
    if (!ttsText.trim()) return

    setTtsError("")
    setTtsPlaying(false)
    setTtsPhase("processing")

    // Revoke previous blob URL to avoid memory leaks
    if (audioBlobUrlRef.current) {
      URL.revokeObjectURL(audioBlobUrlRef.current)
      audioBlobUrlRef.current = null
    }

    try {
      const base64 = await synthesizeSpeech({ data: { text: ttsText } })
      const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0))
      const wavBlob = new Blob([bytes], { type: "audio/wav" })
      const url = URL.createObjectURL(wavBlob)
      audioBlobUrlRef.current = url

      if (audioRef.current) {
        audioRef.current.src = url
        audioRef.current.load()
      }

      setTtsSeed((s) => s + 1)
      setTtsPhase("result")
    } catch (err) {
      setTtsError(err instanceof Error ? err.message : "Synthesis failed.")
      setTtsPhase("idle")
    }
  }

  const handleTtsSend = (e: React.FormEvent) => {
    e.preventDefault()
    startTts()
  }

  const handleTtsPlay = () => {
    if (ttsPhase !== "result") return
    const audio = audioRef.current
    if (!audio) return
    setTtsPlaying(true)
    audio.currentTime = 0
    audio.play().catch(() => setTtsPlaying(false))
  }

  const handleTtsRedo = () => {
    audioRef.current?.pause()
    if (audioRef.current) audioRef.current.currentTime = 0
    setTtsPlaying(false)
    setTtsError("")
    setTtsPhase("idle")
  }

  // ── Cleanup ────────────────────────────────────────────────────────────

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    const onEnded = () => setTtsPlaying(false)
    audio.addEventListener("ended", onEnded)
    return () => audio.removeEventListener("ended", onEnded)
  }, [])

  useEffect(() => {
    return () => {
      if (audioBlobUrlRef.current) URL.revokeObjectURL(audioBlobUrlRef.current)
    }
  }, [])

  return (
    <div className="flex min-h-svh flex-col items-center bg-[#f5f4f3] pt-16">
      <Tabs className="w-full max-w-4xl" defaultValue="asr">
        <TabsList className="mx-auto rounded-xl border border-stone-200/50 p-1 **:data-[slot=tab-indicator]:rounded-lg **:data-[slot=tab-indicator]:border **:data-[slot=tab-indicator]:border-stone-100 **:data-[slot=tab-indicator]:shadow-none">
          <TabsTab
            className="h-11 rounded-lg px-4 text-sm sm:h-11 sm:text-sm"
            value="asr"
          >
            <Mic />
            Speech Recognition
          </TabsTab>
          <TabsTab
            className="h-11 rounded-lg px-4 text-sm sm:h-11 sm:text-sm"
            value="tts"
          >
            <Volume2 />
            Text to Speech
          </TabsTab>
          <TabsTab
            className="h-11 rounded-lg px-4 text-sm sm:h-11 sm:text-sm"
            value="s2s"
          >
            <ArrowLeftRight />
            Speech to Speech
          </TabsTab>
        </TabsList>

        {/* ── ASR Panel ─────────────────────────────────────────────────── */}
        <TabsPanel
          className="flex min-h-[60vh] w-full flex-col items-center justify-center gap-8 px-6"
          value="asr"
        >
          <input
            ref={uploadInputRef}
            type="file"
            accept=".mp3,audio/mpeg"
            className="hidden"
            onChange={handleAsrUpload}
          />
          {(asrState === "idle" || asrState === "listening") && (
            <div className="flex flex-col items-center gap-3">
              <VoiceInput onStart={handleAsrStart} onStop={handleAsrStop} />
              <button
                type="button"
                onClick={handleUploadClick}
                disabled={isUploading}
                className="inline-flex h-10 items-center justify-center gap-2 rounded-full border border-stone-200 px-4 text-sm text-foreground transition-colors hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60"
              >
                <Upload className="h-4 w-4" />
                Upload
              </button>
            </div>
          )}
          {asrState === "transcribing" && (
            <div className="text-center">
              <TextShimmer className="text-2xl" duration={1}>
                Transcribing...
              </TextShimmer>
            </div>
          )}
          {asrState === "done" && (
            <>
              <TextEffect
                className="max-w-5xl text-center text-3xl text-foreground"
                per="word"
                preset="fade"
              >
                {transcript || "No speech detected."}
              </TextEffect>
              <button
                className="inline-flex h-12 w-12 items-center justify-center rounded-full border border-stone-200 text-foreground transition-colors hover:bg-muted"
                onClick={handleAsrReset}
                type="button"
              >
                <RotateCcw className="h-5 w-5" />
              </button>
            </>
          )}
          {asrState === "error" && (
            <div className="flex flex-col items-center gap-4 text-center">
              <p className="text-sm text-red-500">{asrError}</p>
              <button
                className="inline-flex h-12 w-12 items-center justify-center rounded-full border border-stone-200 text-foreground transition-colors hover:bg-muted"
                onClick={handleAsrReset}
                type="button"
              >
                <RotateCcw className="h-5 w-5" />
              </button>
            </div>
          )}
        </TabsPanel>

        {/* ── TTS Panel ─────────────────────────────────────────────────── */}
        <TabsPanel
          className="flex min-h-[60vh] items-center justify-center px-6"
          value="tts"
        >
          {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
          <audio ref={audioRef} />
          {ttsPhase !== "result" ? (
            <div className="flex w-full max-w-lg flex-col gap-3">
              <form
                onSubmit={handleTtsSend}
                className="relative w-full rounded-2xl border border-stone-200/70 bg-background p-4"
              >
                <textarea
                  value={ttsText}
                  onChange={(e) => setTtsText(e.target.value)}
                  placeholder="nyora zvaunoda kunzwa"
                  disabled={ttsPhase === "processing"}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault()
                      startTts()
                    }
                  }}
                  className="h-28 w-full resize-none rounded-xl bg-transparent p-3 text-base outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:opacity-60"
                />

                <button
                  type="submit"
                  disabled={ttsPhase === "processing"}
                  className="absolute right-3 bottom-3 inline-flex h-10 w-10 items-center justify-center overflow-hidden rounded-full border border-primary/20 bg-transparent text-foreground disabled:cursor-not-allowed disabled:opacity-60"
                >
                  <span
                    aria-hidden="true"
                    className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,rgba(205,175,250,1),transparent_50%),radial-gradient(ellipse_at_bottom_right,rgba(129,169,248,1),transparent_50%),radial-gradient(ellipse_at_top_left,rgba(247,203,191,1),transparent_50%),radial-gradient(ellipse_at_bottom_left,rgba(164,252,245,1),transparent_50%)]"
                  />
                  {ttsPhase === "processing" ? (
                    <span
                      aria-label="Processing"
                      className="relative inline-block h-4 w-4 animate-spin rounded-full border-2 border-foreground/30 border-t-foreground"
                    />
                  ) : (
                    <Volume2 className="relative h-5 w-5" />
                  )}
                </button>
              </form>
              {ttsError && (
                <p className="text-center text-sm text-red-500">{ttsError}</p>
              )}
            </div>
          ) : (
            <div className="flex w-full max-w-lg flex-col items-center justify-center gap-6">
              <div className="w-full">
                {ttsPlaying ? (
                  <ScrollingWaveform
                    height={90}
                    speed={55}
                    barCount={60}
                    barWidth={4}
                    barGap={2}
                    barRadius={2}
                    barColor="#9A9A99"
                    fadeEdges={true}
                    fadeWidth={30}
                    className="mx-auto"
                  />
                ) : (
                  <StaticWaveform
                    height={90}
                    bars={54}
                    seed={ttsSeed}
                    barWidth={4}
                    barGap={2}
                    barRadius={2}
                    barColor="#9A9A99"
                    fadeEdges={true}
                    fadeWidth={30}
                    className="mx-auto"
                  />
                )}
              </div>

              <div className="mt-2 flex w-full items-center justify-center gap-8">
                <button
                  type="button"
                  onClick={handleTtsPlay}
                  className="relative inline-flex h-12 w-12 items-center justify-center overflow-hidden rounded-full bg-transparent text-foreground"
                >
                  <span
                    aria-hidden="true"
                    className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,rgba(205,175,250,1),transparent_50%),radial-gradient(ellipse_at_bottom_right,rgba(129,169,248,1),transparent_50%),radial-gradient(ellipse_at_top_left,rgba(247,203,191,1),transparent_50%),radial-gradient(ellipse_at_bottom_left,rgba(164,252,245,1),transparent_50%)]"
                  />
                  <Play className="relative h-5 w-5" />
                </button>

                <button
                  type="button"
                  onClick={handleTtsRedo}
                  className="inline-flex h-12 w-12 items-center justify-center rounded-full border border-stone-200/70 text-foreground transition-colors hover:bg-muted"
                >
                  <RotateCcw className="h-5 w-5" />
                </button>
              </div>
            </div>
          )}
        </TabsPanel>

        <TabsPanel
          className="flex min-h-[60vh] items-center justify-center pt-8 pb-10"
          value="s2s"
        >
          <S2SPanel />
        </TabsPanel>
      </Tabs>
    </div>
  )
}
