import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules")) {
            if (id.includes("react") || id.includes("react-router")) {
              return "vendor-react"
            }
            if (id.includes("recharts")) {
              return "vendor-charts"
            }
            if (id.includes("leaflet")) {
              return "vendor-maps"
            }
            if (id.includes("hls.js")) {
              return "vendor-media"
            }
            if (id.includes("lucide-react")) {
              return "vendor-icons"
            }
          }
        },
      },
    },
  },
})
