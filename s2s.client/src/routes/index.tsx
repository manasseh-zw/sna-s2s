import { Tabs, TabsList, TabsPanel, TabsTab } from "@/components/ui/tabs"
import { TextEffect } from "@/components/ui/text-effect"
import { TextShimmer } from "@/components/ui/text-shimmer"
import { VoiceInput } from "@/components/voice-input"
import { ScrollingWaveform, StaticWaveform } from "@/components/ui/waveform"
import { createFileRoute } from "@tanstack/react-router"
import { ArrowLeftRight, Mic, Play, RotateCcw, Volume2 } from "lucide-react"
import { useEffect, useRef, useState } from "react"

export const Route = createFileRoute("/")({ component: App })

function App() {
  const [asrState, setAsrState] = useState<
    "idle" | "listening" | "transcribing" | "done"
  >("idle")
  const [transcript, setTranscript] = useState("")
  const transcriptionTimerRef = useRef<ReturnType<typeof setTimeout> | null>(
    null
  )

  const [ttsText, setTtsText] = useState("")

  const [ttsPhase, setTtsPhase] = useState<
    "idle" | "processing" | "result"
  >("idle")
  const [ttsPlaying, setTtsPlaying] = useState(false)
  const [ttsSeed, setTtsSeed] = useState(42)

  const ttsTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const startTts = () => {
    if (ttsPhase === "processing") return
    if (!ttsText.trim()) return

    if (ttsTimerRef.current) clearTimeout(ttsTimerRef.current)

    setTtsPlaying(false)
    setTtsPhase("processing")

    // Placeholder for the upcoming TTS API call.
    ttsTimerRef.current = setTimeout(() => {
      setTtsSeed((s) => s + 1)
      setTtsPhase("result")
    }, 1600)
  }

  const handleTtsSend = (e: React.FormEvent) => {
    e.preventDefault()
    startTts()
  }

  const handleTtsPlay = () => {
    if (ttsPhase !== "result") return

    setTtsPlaying(true)
    const audio = audioRef.current
    if (audio) {
      audio.currentTime = 0
      audio
        .play()
        .catch(() => setTtsPlaying(false))
    }
  }

  const handleTtsRedo = () => {
    if (ttsTimerRef.current) clearTimeout(ttsTimerRef.current)
    audioRef.current?.pause()
    if (audioRef.current) audioRef.current.currentTime = 0
    setTtsPlaying(false)
    setTtsPhase("idle")
  }

  const handleAsrStart = () => {
    if (transcriptionTimerRef.current)
      clearTimeout(transcriptionTimerRef.current)
    setTranscript("")
    setAsrState("listening")
  }

  const handleAsrStop = () => {
    setAsrState("transcribing")
    transcriptionTimerRef.current = setTimeout(() => {
      setTranscript(
        "This is placeholder output text from your speech recording."
      )
      setAsrState("done")
    }, 1800)
  }

  const handleAsrReset = () => {
    if (transcriptionTimerRef.current)
      clearTimeout(transcriptionTimerRef.current)
    setTranscript("")
    setAsrState("idle")
  }

  useEffect(() => {
    return () => {
      if (transcriptionTimerRef.current)
        clearTimeout(transcriptionTimerRef.current)
    }
  }, [])

  useEffect(() => {
    return () => {
      if (ttsTimerRef.current) clearTimeout(ttsTimerRef.current)
      audioRef.current?.pause()
    }
  }, [])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    const onEnded = () => setTtsPlaying(false)
    audio.addEventListener("ended", onEnded)
    return () => audio.removeEventListener("ended", onEnded)
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

        <TabsPanel
          className="flex min-h-[60vh] w-full flex-col items-center justify-center gap-8 px-6"
          value="asr"
        >
          {(asrState === "idle" || asrState === "listening") && (
            <VoiceInput onStart={handleAsrStart} onStop={handleAsrStop} />
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
                {transcript}
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
        </TabsPanel>

        <TabsPanel
          className="flex min-h-[60vh] items-center justify-center px-6"
          value="tts"
        >
          <audio ref={audioRef} src="/audio.wav" />
          {ttsPhase !== "result" ? (
            <form
              onSubmit={handleTtsSend}
              className="relative w-full max-w-lg rounded-2xl border border-stone-200/70 bg-background p-4"
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
                className="h-28 w-full resize-none rounded-xl bg-transparent p-3 text-base outline-none placeholder:text-muted-foreground disabled:opacity-60 disabled:cursor-not-allowed"
              />

              <button
                type="submit"
                disabled={ttsPhase === "processing"}
                className="absolute right-3 bottom-3 inline-flex h-10 w-10 items-center justify-center overflow-hidden rounded-full border border-primary/20 bg-transparent text-foreground disabled:opacity-60 disabled:cursor-not-allowed"
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

        <TabsPanel className="pt-8" value="s2s">
          <p className="text-center text-xs text-muted-foreground">
            <span aria-hidden="true">🎧 </span>
            Speech-to-Speech placeholder.
          </p>
        </TabsPanel>
      </Tabs>
    </div>
  )
}
