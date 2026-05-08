import type { Genome, RunData, RunMeta } from "./types";

export async function loadRun(baseUrl: string): Promise<RunData> {
  const base = baseUrl.replace(/\/$/, "");
  const [genome, meta, trajBuf] = await Promise.all([
    fetchJson<Genome>(`${base}/genome.json`),
    fetchJson<RunMeta>(`${base}/meta.json`),
    fetchBinary(`${base}/trajectory.bin`),
  ]);

  const trajectory = new Float32Array(trajBuf);
  const expected = meta.n_frames * meta.nq;
  if (trajectory.length !== expected) {
    throw new Error(
      `trajectory.bin size mismatch: got ${trajectory.length} floats, expected ${expected} (n_frames=${meta.n_frames}, nq=${meta.nq})`,
    );
  }

  return { genome, meta, trajectory };
}

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url} failed: ${r.status}`);
  const text = await r.text();
  // Vite's SPA fallback returns index.html when a static file is missing,
  // which causes a confusing JSON parse error. Detect it explicitly.
  if (text.startsWith("<!") || text.startsWith("<html")) {
    throw new Error(
      `${url} not found (the dev server returned the index.html fallback). ` +
        `Did you run 'simudep train' or 'simudep inspect-one' ? ` +
        `Use ?run=/runs/<your-run>/selected/latest in the URL.`,
    );
  }
  return JSON.parse(text) as T;
}

async function fetchBinary(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url} failed: ${r.status}`);
  const ct = r.headers.get("content-type") ?? "";
  if (ct.includes("text/html")) {
    throw new Error(`${url} not found (got index.html fallback).`);
  }
  return await r.arrayBuffer();
}
