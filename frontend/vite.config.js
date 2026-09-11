import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const port = env.VISION_PORT || '8765';
  const apiTarget = env.VITE_API_PROXY_TARGET || `http://127.0.0.1:${port}`;

  return {
    plugins: [react()],
    server: {
      host: true,
      port: 5173,
      proxy: {
        '/api': { target: apiTarget, changeOrigin: true },
        '/video_feed': { target: apiTarget, changeOrigin: true },
      },
    },
    build: { outDir: 'dist', emptyOutDir: true },
  };
});
