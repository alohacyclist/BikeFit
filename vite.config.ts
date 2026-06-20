import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// COOP/COEP-Header aktivieren Cross-Origin-Isolation, damit SharedArrayBuffer
// im Browser verfügbar wird (Voraussetzung für WASM-Multi-Threading).
// Wirkt sich nicht auf Single-Thread-Betrieb aus — falls die Header nicht
// gesetzt sind oder fremde Ressourcen sie verletzen, fällt Multi automatisch
// auf Single zurück.
const crossOriginIsolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    headers: crossOriginIsolationHeaders,
  },
  preview: {
    host: true,
    port: 4173,
    headers: crossOriginIsolationHeaders,
  },
  optimizeDeps: {
    // tfjs-tflite ist UMD-only — wird via <script>-Tag in index.html geladen,
    // nicht als ESM-Modul gebundelt. Esbuild soll es nicht anfassen.
    exclude: ['@tensorflow/tfjs-tflite'],
  },
});
