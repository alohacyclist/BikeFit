import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
  },
  optimizeDeps: {
    // tfjs-tflite lädt einen WASM-Client lazy aus eigenem Bundle — esbuild muss prebundle ausführen
    include: ['@tensorflow/tfjs-tflite'],
  },
  build: {
    rollupOptions: {
      // Internes Lazy-Modul von tfjs-tflite kann Rollup nicht statisch auflösen.
      // Externalisieren — Browser lädt es zur Laufzeit aus dem geshippten dist-Folder.
      external: [/tflite_web_api_client/],
    },
  },
});
