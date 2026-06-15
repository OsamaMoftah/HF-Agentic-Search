import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/',
  build: { outDir: 'dist', assetsDir: 'assets' },
  server: {
    port: 5173,
    proxy: {
      '/weave': 'http://localhost:7860',
      '/state': 'http://localhost:7860',
      '/sprites': 'http://localhost:7860',
    },
  },
});
