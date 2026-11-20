import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { proxy: {
    '/api': 'http://127.0.0.1:8001',
    '/health': 'http://127.0.0.1:8001',
    '/ready': 'http://127.0.0.1:8001',
    '/docs': 'http://127.0.0.1:8001',
    '/openapi.json': 'http://127.0.0.1:8001',
  } },
})
