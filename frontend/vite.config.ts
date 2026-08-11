import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Proxies /api/* → http://localhost:8000/api/* during development. Backend routes are
      // /api-prefixed, so the prefix must be preserved. In production FastAPI serves the built
      // frontend directly (same origin, no proxy).
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
