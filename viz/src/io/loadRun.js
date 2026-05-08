export async function loadRun(baseUrl) {
    const base = baseUrl.replace(/\/$/, "");
    const [genome, meta, trajBuf] = await Promise.all([
        fetchJson(`${base}/genome.json`),
        fetchJson(`${base}/meta.json`),
        fetchBinary(`${base}/trajectory.bin`),
    ]);
    const trajectory = new Float32Array(trajBuf);
    const expected = meta.n_frames * meta.nq;
    if (trajectory.length !== expected) {
        throw new Error(`trajectory.bin size mismatch: got ${trajectory.length} floats, expected ${expected} (n_frames=${meta.n_frames}, nq=${meta.nq})`);
    }
    return { genome, meta, trajectory };
}
async function fetchJson(url) {
    const r = await fetch(url);
    if (!r.ok)
        throw new Error(`fetch ${url} failed: ${r.status}`);
    return (await r.json());
}
async function fetchBinary(url) {
    const r = await fetch(url);
    if (!r.ok)
        throw new Error(`fetch ${url} failed: ${r.status}`);
    return await r.arrayBuffer();
}
