import { defineConfig } from "vite"
import { devtools } from "@tanstack/devtools-vite"
import { tanstackStart } from "@tanstack/react-start/plugin/vite"
import viteReact from "@vitejs/plugin-react"
import viteTsConfigPaths from "vite-tsconfig-paths"
import tailwindcss from "@tailwindcss/vite"
import { nitro } from "nitro/vite"

const config = defineConfig({
  plugins: [
    devtools(),
    nitro(),
    // this is the plugin that enables path aliases
    viteTsConfigPaths({
      projects: ["./tsconfig.json"],
    }),
    tailwindcss(),
    tanstackStart(),
    viteReact(),
  ],
  optimizeDeps: {
    // Pre-bundle VAD + ONNX Runtime so CJS `require("onnxruntime-web/wasm")`
    // is transformed during dependency optimization.
    include: ["@ricky0123/vad-react", "@ricky0123/vad-web", "onnxruntime-web/wasm"],
  },
  ssr: {
    // Never try to bundle these in the SSR build — they're browser-only WASM
    external: ["onnxruntime-web", "@ricky0123/vad-web", "@ricky0123/vad-react"],
  },
  // Treat .wasm and .onnx as static assets
  assetsInclude: ["**/*.wasm", "**/*.onnx"],
})

export default config

