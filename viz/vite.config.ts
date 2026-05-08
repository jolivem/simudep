import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  root: ".",
  publicDir: "public",
  server: {
    fs: {
      // Allow Vite to serve files from the runs/ folder at the repo root.
      allow: [resolve(__dirname, ".."), resolve(__dirname)],
    },
  },
});
