import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
  },
  optimizeDeps: {
    // tfjs-tflite ist UMD-only — wird via <script>-Tag in index.html geladen,
    // nicht als ESM-Modul gebundelt. Esbuild soll es nicht anfassen.
    exclude: ['@tensorflow/tfjs-tflite'],
  },
});
