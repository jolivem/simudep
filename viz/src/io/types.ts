// TypeScript mirror of the JSON schema produced by simudep.io.genome_json.
// Keep in sync with training/src/simudep/io/genome_json.py.

export type Vec3 = readonly [number, number, number];
export type Vec4 = readonly [number, number, number, number];

export interface Joint {
  axis: Vec3;
  anchor_parent: Vec3;
  range: readonly [number, number];
  kp: number;
  kd: number;
}

export interface Segment {
  size: Vec3;
  geom_pos: Vec3;
  mass: number;
  color: Vec4;
  /** Present on every non-root segment, undefined on the root. */
  joint?: Joint;
  children: Segment[];
}

export interface SequenceStep {
  duration: number;
  targets: number[];
}

export interface SequenceJSON {
  cycle_duration: number;
  steps: SequenceStep[];
}

export interface Genome {
  version: number;
  id: string;
  initial_root_pos: Vec3;
  root: Segment;
  sequence: SequenceJSON;
  fitness: { distance: number; energy: number; score: number } | null;
}

export interface RunMeta {
  version: number;
  fps: number;
  dt: number;
  duration: number;
  timestep: number;
  n_frames: number;
  nq: number;
  n_joints: number;
  body_names: string[];
  fitness: { distance: number; energy: number; score: number } | null;
}

export interface RunData {
  genome: Genome;
  meta: RunMeta;
  /** Row-major Float32Array of shape [n_frames, nq]. */
  trajectory: Float32Array;
}
