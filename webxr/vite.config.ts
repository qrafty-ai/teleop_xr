import { defineConfig } from "vite";
import mkcert from "vite-plugin-mkcert";
import path from "path";

export default defineConfig({
  plugins: [
    mkcert(),
  ],
  server: { host: "0.0.0.0", port: 8081, open: true },
  build: {
    outDir: "dist",
    sourcemap: process.env.NODE_ENV !== "production",
    target: "esnext",
    rollupOptions: { input: "./index.html" },
  },
  esbuild: { target: "esnext" },
  optimizeDeps: {
    esbuildOptions: { target: "esnext" },
  },
  resolve: {
    alias: {
      three: path.resolve(__dirname, "node_modules/aframe/node_modules/three"),
    },
  },
  publicDir: "public",
  base: "./",
});
