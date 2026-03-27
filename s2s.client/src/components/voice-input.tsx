import React from "react"
import { Mic } from "lucide-react"
import { AnimatePresence, motion } from "motion/react"

import { cn } from "@/lib/utils"

interface VoiceInputProps {
  onStart?: () => void
  onStop?: (blob: Blob) => void
}

export function VoiceInput({
  className,
  onStart,
  onStop,
}: React.ComponentProps<"div"> & VoiceInputProps) {
  const [_listening, _setListening] = React.useState<boolean>(false)
  const [_time, _setTime] = React.useState<number>(0)

  const mediaRecorderRef = React.useRef<MediaRecorder | null>(null)
  const chunksRef = React.useRef<BlobPart[]>([])

  React.useEffect(() => {
    let intervalId: NodeJS.Timeout

    if (_listening) {
      intervalId = setInterval(() => {
        _setTime((t) => t + 1)
      }, 1000)
    } else {
      _setTime(0)
    }

    return () => clearInterval(intervalId)
  }, [_listening])

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      chunksRef.current = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" })
        // Stop all tracks so the mic indicator goes away
        stream.getTracks().forEach((t) => t.stop())
        onStop?.(blob)
      }

      recorder.start()
      mediaRecorderRef.current = recorder
      _setListening(true)
      onStart?.()
    } catch {
      console.error("Microphone access denied or unavailable.")
    }
  }

  const stopRecording = () => {
    mediaRecorderRef.current?.stop()
    mediaRecorderRef.current = null
    _setListening(false)
  }

  const onClickHandler = () => {
    if (_listening) {
      stopRecording()
    } else {
      startRecording()
    }
  }

  return (
    <div className={cn("flex flex-col items-center justify-center", className)}>
      <motion.div
        className={cn(
          "relative flex cursor-pointer items-center justify-center overflow-hidden rounded-full p-4 shadow-sm",
          _listening
            ? "border border-input bg-background text-foreground"
            : "border border-stone-200 bg-transparent text-foreground"
        )}
        layout
        transition={{
          layout: {
            duration: 0.4,
          },
        }}
        onClick={onClickHandler}
      >
        {!_listening && (
          <div
            aria-hidden="true"
            className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,rgba(205,175,250,1),transparent_50%),radial-gradient(ellipse_at_bottom_right,rgba(129,169,248,1),transparent_50%),radial-gradient(ellipse_at_top_left,rgba(247,203,191,1),transparent_50%),radial-gradient(ellipse_at_bottom_left,rgba(164,252,245,1),transparent_50%)]"
          />
        )}
        <div className="flex h-8 w-8 items-center justify-center">
          {_listening ? (
            <motion.div
              className="h-5 w-5 rounded-sm bg-[#ef233c]"
              animate={{
                rotate: [0, 180, 360],
              }}
              transition={{
                duration: 2,
                repeat: Number.POSITIVE_INFINITY,
                ease: "easeInOut",
              }}
            />
          ) : (
            <Mic
              className={cn(
                "h-6 w-6",
                _listening ? "text-foreground" : "text-foreground"
              )}
            />
          )}
        </div>
        <AnimatePresence mode="wait">
          {_listening && (
            <motion.div
              initial={{ opacity: 0, width: 0, marginLeft: 0 }}
              animate={{ opacity: 1, width: "9rem", marginLeft: 10 }}
              exit={{ opacity: 0, width: 0, marginLeft: 0 }}
              transition={{
                duration: 0.4,
              }}
              className="flex items-center justify-center gap-2 overflow-hidden"
            >
              {/* Frequency Animation */}
              <div className="flex items-center justify-center gap-1">
                {[...Array(14)].map((_, i) => (
                  <motion.div
                    key={i}
                    className={cn(
                      "w-0.5 rounded-full",
                      _listening ? "bg-foreground" : "bg-foreground"
                    )}
                    initial={{ height: 2 }}
                    animate={{
                      height: _listening
                        ? [2, 5 + Math.random() * 14, 4 + Math.random() * 8, 2]
                        : 2,
                    }}
                    transition={{
                      duration: _listening ? 1 : 0.3,
                      repeat: _listening ? Infinity : 0,
                      delay: _listening ? i * 0.05 : 0,
                      ease: "easeInOut",
                    }}
                  />
                ))}
              </div>
              {/* Timer */}
              <div
                className={cn(
                  "w-11 text-center text-xs",
                  _listening
                    ? "text-muted-foreground"
                    : "text-primary-foreground/80"
                )}
              >
                {formatTime(_time)}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  )
}
