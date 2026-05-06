import { Send, Volume2 } from "lucide-react"
import * as React from "react"

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { TooltipProvider } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

export type PromptInputVoiceOption = {
  value: string
  label: string
}

type PromptInputProps = {
  value: string
  onValueChange: (value: string) => void
  onSubmit: () => void
  voice: string
  onVoiceChange: (voice: string) => void
  voices: readonly PromptInputVoiceOption[]
  placeholder?: string
  disabled?: boolean
  isSubmitting?: boolean
  className?: string
}

function PromptInput({
  value,
  onValueChange,
  onSubmit,
  voice,
  onVoiceChange,
  voices,
  placeholder = "Message...",
  disabled = false,
  isSubmitting = false,
  className,
}: PromptInputProps) {
  const textareaRef = React.useRef<HTMLTextAreaElement | null>(null)

  React.useLayoutEffect(() => {
    const textarea = textareaRef.current
    if (!textarea) return
    textarea.style.height = "auto"
    textarea.style.height = `${Math.min(textarea.scrollHeight, 220)}px`
  }, [value])

  const canSend = !disabled && !isSubmitting && value.trim().length > 0

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      if (canSend) onSubmit()
    }
  }

  return (
    <div
      className={cn(
        "flex flex-col rounded-[28px] border border-stone-200/80 bg-white p-2 shadow-sm transition-colors",
        className
      )}
    >
      <textarea
        ref={textareaRef}
        rows={1}
        value={value}
        onChange={(event) => onValueChange(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled}
        className="min-h-24 w-full resize-none border-0 bg-transparent px-3 py-3 text-base text-foreground outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed disabled:opacity-60"
      />

      <TooltipProvider delay={100}>
        <div className="flex items-center gap-2 p-1 pt-0">
          <Select
            value={voice}
            onValueChange={(nextVoice) => {
              if (nextVoice) onVoiceChange(nextVoice)
            }}
          >
            <SelectTrigger
              className="h-12 min-w-40 rounded-xl border-stone-200 bg-white px-4 text-sm font-medium text-foreground shadow-none"
              disabled={disabled}
            >
              <Volume2 className="h-4 w-4 text-muted-foreground" />
              <SelectValue placeholder="Voices" />
            </SelectTrigger>
            <SelectContent
              align="start"
              side="top"
              sideOffset={10}
              alignItemWithTrigger={false}
              className="min-w-40 rounded-2xl border border-stone-200 bg-white p-1 shadow-lg ring-0"
            >
              {voices.map((option) => (
                <SelectItem
                  key={option.value}
                  value={option.value}
                  className="rounded-xl px-3 py-2 text-sm font-medium text-foreground focus:bg-primary/10 focus:text-primary data-[selected]:text-primary"
                >
                  <span className="min-w-0">{option.label}</span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <div className="ml-auto">
            <button
              type="button"
              onClick={onSubmit}
              disabled={!canSend}
              title="Generate speech"
              className="flex h-9 w-9 items-center justify-center rounded-full bg-primary text-primary-foreground transition-colors hover:bg-primary/90 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50"
            >
              {isSubmitting ? (
                <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground/30 border-t-primary-foreground" />
              ) : (
                <Send className="h-4.5 w-4.5" />
              )}
              <span className="sr-only">Send</span>
            </button>
          </div>
        </div>
      </TooltipProvider>
    </div>
  )
}

export { PromptInput }
