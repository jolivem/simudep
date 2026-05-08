import * as THREE from "three";
import type { Genome, Segment, Vec3 } from "../io/types";

/** Index of a joint in DFS order (matches qpos indexing past the freejoint). */
export interface JointBinding {
  /** The Three.js group whose rotation must be set to (axis, qpos[7+jointIndex]). */
  group: THREE.Group;
  /** Joint axis as a unit vector. */
  axis: THREE.Vector3;
}

export interface BuiltCreature {
  /** Top-level Three.js group whose pos/quat are driven by the freejoint qpos. */
  root: THREE.Group;
  /** Joints in DFS order; entry i corresponds to qpos[7 + i]. */
  joints: JointBinding[];
}

/**
 * Reconstruct the creature as a Three.js hierarchy.
 *
 *   - The returned `root` group represents the root body. Its position and
 *     quaternion are driven by the freejoint qpos at animation time.
 *   - Inside `root`, the box geom is offset by `geom_pos` and shaped from
 *     `size` (half-extents → full extents).
 *   - Each child segment becomes a sub-group positioned at
 *     `joint.anchor_parent` relative to its parent. Rotating that group
 *     around `joint.axis` reproduces the MuJoCo hinge rotation.
 */
export function buildCreature(genome: Genome): BuiltCreature {
  const joints: JointBinding[] = [];
  const root = new THREE.Group();
  root.name = `creature:${genome.id}`;

  addGeom(root, genome.root);
  for (const child of genome.root.children) {
    addChild(root, child, joints);
  }

  return { root, joints };
}

function addChild(parent: THREE.Object3D, seg: Segment, joints: JointBinding[]): void {
  if (!seg.joint) {
    throw new Error("non-root segment must have a joint");
  }
  const group = new THREE.Group();
  group.position.set(...(seg.joint.anchor_parent as Vec3));
  parent.add(group);

  joints.push({
    group,
    axis: new THREE.Vector3(...(seg.joint.axis as Vec3)).normalize(),
  });

  addGeom(group, seg);
  for (const sub of seg.children) {
    addChild(group, sub, joints);
  }
}

function addGeom(group: THREE.Object3D, seg: Segment): void {
  const [hx, hy, hz] = seg.size;
  const geometry = new THREE.BoxGeometry(2 * hx, 2 * hy, 2 * hz);
  const [r, g, b, a] = seg.color;
  const material = new THREE.MeshStandardMaterial({
    color: new THREE.Color(r, g, b),
    roughness: 0.6,
    transparent: a < 1,
    opacity: a,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(...(seg.geom_pos as Vec3));
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  group.add(mesh);
}
