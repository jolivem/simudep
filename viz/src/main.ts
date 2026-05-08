import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { loadRun } from "./io/loadRun";
import { buildCreature } from "./scene/CreatureBuilder";
import { Animator } from "./scene/Animator";

// Phase 2: default to the most recently written individual; `?run=...` overrides.
// Run-picker UI comes in Phase 5.
const url = new URL(window.location.href);
const RUN_URL = url.searchParams.get("run") ?? "/runs/inspect/selected/latest";

const canvas = document.getElementById("app") as HTMLCanvasElement;
const hud = document.getElementById("hud") as HTMLDivElement;

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight, false);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

// MuJoCo's world is Z-up; mirror that so axes match.
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111418);

const camera = new THREE.PerspectiveCamera(
  55,
  window.innerWidth / window.innerHeight,
  0.05,
  200,
);
camera.up.set(0, 0, 1);
camera.position.set(1.6, -2.4, 1.4);

scene.add(new THREE.HemisphereLight(0xa6c8ff, 0x303838, 0.7));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(4, -3, 8);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -8;
sun.shadow.camera.right = 8;
sun.shadow.camera.top = 8;
sun.shadow.camera.bottom = -8;
sun.shadow.camera.near = 0.5;
sun.shadow.camera.far = 40;
sun.shadow.bias = -0.0005;
scene.add(sun);

// Ground: a clearly visible warm slate, easy to read against the dark background.
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(80, 80),
  new THREE.MeshStandardMaterial({ color: 0x6f7884, roughness: 0.95 }),
);
ground.receiveShadow = true;
scene.add(ground);

// Grid on the floor with bright meter cells + an emphasized axis cross at origin.
const grid = new THREE.GridHelper(40, 40, 0xeeeeee, 0x3d4651);
grid.rotation.x = Math.PI / 2; // PlaneGeometry/GridHelper default normal is +Y; rotate for Z-up.
grid.position.z = 0.001; // avoid z-fighting with the ground
(grid.material as THREE.Material).opacity = 0.7;
(grid.material as THREE.Material).transparent = true;
scene.add(grid);

// X (red), Y (green), Z (blue) axes at the origin to make orientation obvious.
const axes = new THREE.AxesHelper(0.6);
axes.position.z = 0.002;
scene.add(axes);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.3);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.4;
controls.maxDistance = 30;
controls.maxPolarAngle = Math.PI * 0.49; // keep the camera above the ground

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight, false);
});

let animator: Animator | null = null;
let creatureRoot: THREE.Group | null = null;
let followCamera = true;
let lastFollowTarget = new THREE.Vector3(0, 0, 0.3);

window.addEventListener("keydown", (e) => {
  if (e.key === "f" || e.key === "F") {
    followCamera = !followCamera;
    hud.textContent = `follow camera: ${followCamera ? "on" : "off"} — press F to toggle`;
  }
});

void boot();

async function boot(): Promise<void> {
  try {
    hud.textContent = `loading ${RUN_URL} ...`;
    const run = await loadRun(RUN_URL);
    const built = buildCreature(run.genome);
    scene.add(built.root);
    creatureRoot = built.root;
    animator = new Animator(built, run);
    animator.reset();

    const f = run.meta.fitness;
    const fitTxt = f
      ? `dist=${f.distance.toFixed(3)}m  energy=${f.energy.toFixed(2)}  score=${f.score.toFixed(3)}`
      : "no fitness";
    hud.textContent =
      `${run.genome.id} · ${run.meta.n_frames} frames @ ${run.meta.fps} fps · ${fitTxt}` +
      ` · drag to orbit · F: toggle follow`;
  } catch (err) {
    hud.textContent = `error: ${(err as Error).message}`;
    throw err;
  }
}

const clock = new THREE.Clock();
function tick(): void {
  const dt = Math.min(clock.getDelta(), 0.1);
  if (animator) animator.step(dt);

  if (creatureRoot && followCamera) {
    // Smoothly slide both the camera and its orbit target so the creature
    // stays centered while preserving the user-chosen orbit angle.
    const target = creatureRoot.position;
    const desired = new THREE.Vector3(target.x, target.y, target.z + 0.05);
    const delta = desired.clone().sub(lastFollowTarget);
    controls.target.add(delta);
    camera.position.add(delta);
    lastFollowTarget.copy(desired);
  } else if (creatureRoot) {
    lastFollowTarget.copy(creatureRoot.position);
  }

  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();
