import { Tabs, TabsList, TabsPanel, TabsTab } from "@/components/ui/tabs"
import { TextEffect } from "@/components/ui/text-effect"
import { TextShimmer } from "@/components/ui/text-shimmer"
import { VoiceInput } from "@/components/voice-input"
import { createFileRoute } from "@tanstack/react-router"
import { ArrowLeftRight, Mic, RotateCcw, Volume2 } from "lucide-react"
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

  return (
    <div className="flex min-h-svh flex-col items-center pt-16">
      <Tabs className="w-full max-w-4xl" defaultValue="asr">
        <TabsList className="mx-auto rounded-xl border border-stone-100 p-1 **:data-[slot=tab-indicator]:rounded-lg **:data-[slot=tab-indicator]:border **:data-[slot=tab-indicator]:border-stone-100">
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

        <TabsPanel className="pt-8" value="tts">
          <p className="text-center text-xs text-muted-foreground">
            <span aria-hidden="true">🔊 </span>
            Text-to-Speech placeholder.
          </p>
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
