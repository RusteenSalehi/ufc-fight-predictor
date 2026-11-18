/// <reference types="vitest/config" />
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'
import { publishData } from './vite.publish-data'

const here = path.dirname(fileURLToPath(import.meta.url))
// Where `uv run python -m ufc.predict refresh` writes the public artifacts.
const publishDir = process.env.UFC_PUBLISH_DIR ?? path.resolve(here, '../data/publish/public')

export default defineConfig({
  plugins: [react(), tailwindcss(), publishData(publishDir)],
  build: { sourcemap: true },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
})
