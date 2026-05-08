import * as THREE from "three";
/**
 * Animator that drives a BuiltCreature from a recorded qpos trajectory.
 *
 * The qpos layout produced by simudep is:
 *
 *   [root_x, root_y, root_z,
 *    root_qw, root_qx, root_qy, root_qz,    ← MuJoCo WXYZ convention
 *    j0, j1, ..., j_{N-1}]
 *
 * `step(dt)` advances animation time, looping at the end of the recording.
 * Frames are linearly interpolated for translation and joint angles, and
 * SLERP-interpolated for the root quaternion.
 */
export class Animator {
    creature;
    run;
    nq;
    nFrames;
    frameDt;
    tmpQA = new THREE.Quaternion();
    tmpQB = new THREE.Quaternion();
    /** Current playback time in seconds, modulo loop duration. */
    time = 0;
    /** Multiplier on dt applied during step(); 1 = real time. */
    speed = 1;
    constructor(creature, run) {
        this.creature = creature;
        this.run = run;
        this.nq = run.meta.nq;
        this.nFrames = run.meta.n_frames;
        this.frameDt = run.meta.dt;
        if (this.creature.joints.length !== run.meta.n_joints) {
            throw new Error(`joint count mismatch: creature has ${this.creature.joints.length}, run has ${run.meta.n_joints}`);
        }
    }
    reset() {
        this.time = 0;
        this.applyFrame(0);
    }
    step(dt) {
        if (this.nFrames <= 1) {
            this.applyFrame(0);
            return;
        }
        const total = this.nFrames * this.frameDt;
        this.time = (this.time + dt * this.speed) % total;
        if (this.time < 0)
            this.time += total;
        const exact = this.time / this.frameDt;
        const i0 = Math.floor(exact) % this.nFrames;
        const i1 = (i0 + 1) % this.nFrames;
        this.applyInterpolated(i0, i1, exact - Math.floor(exact));
    }
    applyFrame(i) {
        const off = i * this.nq;
        const q = this.run.trajectory;
        this.creature.root.position.set(q[off], q[off + 1], q[off + 2]);
        // MuJoCo quaternion is (w, x, y, z); Three.js is (x, y, z, w).
        this.creature.root.quaternion.set(q[off + 4], q[off + 5], q[off + 6], q[off + 3]);
        for (let j = 0; j < this.creature.joints.length; j++) {
            const angle = q[off + 7 + j];
            const binding = this.creature.joints[j];
            binding.group.quaternion.setFromAxisAngle(binding.axis, angle);
        }
    }
    applyInterpolated(i0, i1, t) {
        const off0 = i0 * this.nq;
        const off1 = i1 * this.nq;
        const q = this.run.trajectory;
        const px = lerp(q[off0], q[off1], t);
        const py = lerp(q[off0 + 1], q[off1 + 1], t);
        const pz = lerp(q[off0 + 2], q[off1 + 2], t);
        this.creature.root.position.set(px, py, pz);
        this.tmpQA.set(q[off0 + 4], q[off0 + 5], q[off0 + 6], q[off0 + 3]);
        this.tmpQB.set(q[off1 + 4], q[off1 + 5], q[off1 + 6], q[off1 + 3]);
        this.creature.root.quaternion.copy(this.tmpQA).slerp(this.tmpQB, t);
        for (let j = 0; j < this.creature.joints.length; j++) {
            const a0 = q[off0 + 7 + j];
            const a1 = q[off1 + 7 + j];
            const binding = this.creature.joints[j];
            binding.group.quaternion.setFromAxisAngle(binding.axis, lerp(a0, a1, t));
        }
    }
}
function lerp(a, b, t) {
    return a + (b - a) * t;
}
