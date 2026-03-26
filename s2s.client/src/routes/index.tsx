import { Tabs, TabsList, TabsPanel, TabsTab } from "@/components/ui/tabs"
import { VoiceInput } from "@/components/voice-input"
import { createFileRoute } from "@tanstack/react-router"
import { ArrowLeftRight, Mic, Volume2 } from "lucide-react"

export const Route = createFileRoute("/")({ component: App })

function App() {
  return (
    <div className="flex min-h-svh flex-col items-center pt-16">
      <Tabs className="w-full max-w-4xl" defaultValue="asr">
        <TabsList className="mx-auto border border-stone-100 **:data-[slot=tab-indicator]:border **:data-[slot=tab-indicator]:border-stone-100">
          <TabsTab className="text-xs sm:text-xs" value="asr">
            <Mic />
            Automatic Speech Recognition
          </TabsTab>
          <TabsTab className="text-xs sm:text-xs" value="tts">
            <Volume2 />
            Text to Speech
          </TabsTab>
          <TabsTab className="text-xs sm:text-xs" value="s2s">
            <ArrowLeftRight />
            Speech to Speech
          </TabsTab>
        </TabsList>

        <TabsPanel
          className="flex min-h-[60vh] items-center justify-center"
          value="asr"
        >
          <VoiceInput />
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
