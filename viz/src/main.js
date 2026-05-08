import * as THREE from "three";
import { loadRun } from "./io/loadRun";
import { buildCreature } from "./scene/CreatureBuilder";
import { Animator } from "./scene/Animator";
// Phase 1: hardcoded run path. Run-picker UI comes in Phase 5.
const RUN_URL = "/runs/tetrapod/selected/tetrapod_ref";
const canvas = document.getElementById("app");
const hud = document.getElementById("hud");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight, false);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
// MuJoCo's world is Z-up; mirror that so axes match.
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d10);
scene.fog = new THREE.Fog(0x0b0d10, 12, 40);
const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.05, 100);
camera.up.set(0, 0, 1);
camera.position.set(1.2, -1.6, 1.0);
camera.lookAt(0, 0, 0.2);
scene.add(new THREE.AmbientLight(0xffffff, 0.35));
const sun = new THREE.DirectionalLight(0xffffff, 1.1);
sun.position.set(3, -2, 6);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -5;
sun.shadow.camera.right = 5;
sun.shadow.camera.top = 5;
sun.shadow.camera.bottom = -5;
sun.shadow.camera.near = 0.5;
sun.shadow.camera.far = 30;
scene.add(sun);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({ color: 0x1f242b, roughness: 0.95 }));
ground.receiveShadow = true;
scene.add(ground);
// Subtle grid for orientation.
const grid = new THREE.GridHelper(20, 40, 0x2a3340, 0x1a2028);
grid.rotation.x = Math.PI / 2;
scene.add(grid);
window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight, false);
});
let animator = null;
let creatureRoot = null;
void boot();
async function boot() {
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
        hud.textContent = `${run.genome.id} · ${run.meta.n_frames} frames @ ${run.meta.fps} fps · ${fitTxt}`;
    }
    catch (err) {
        hud.textContent = `error: ${err.message}`;
        throw err;
    }
}
const clock = new THREE.Clock();
function tick() {
    const dt = Math.min(clock.getDelta(), 0.1);
    if (animator)
        animator.step(dt);
    if (creatureRoot) {
        // Simple follow camera: keep target on the creature root.
        const target = creatureRoot.position;
        camera.lookAt(target.x, target.y, target.z + 0.1);
    }
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
}
tick();
