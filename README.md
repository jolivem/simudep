# simudep

Evolutionary creature simulator (Karl Sims style).

Creatures = tree of segments connected by motorized 1-DOF revolute joints.
A genetic algorithm evolves their topology while CMA-ES tunes the cyclic
control sequence. Fitness = distance travelled on a flat ground, penalized
by motor energy.

- **Training**: Python + MuJoCo MJX (JAX, CUDA). See [training/](training/).
- **Visualization**: Vite + TypeScript + Three.js. Trajectories are
  pre-computed in Python and replayed in the browser (no physics in the web).
  See [viz/](viz/).
- **Plan**: see [PLAN.md](PLAN.md).

## Prerequisites

- NVIDIA GPU with CUDA 12 driver (Phase 0 was validated on RTX 4060 / WSL2).
- [`uv`](https://docs.astral.sh/uv/) for Python.
- Node.js 20+ for the viz.

## Quick start

```bash
# Verify the runtime (JAX must list a CudaDevice)
npm run doctor

# Run Python tests
npm run test:py

# Start the viz dev server
npm run viz:dev
```

## Layout

```
training/   Python — genome model, MJX rollouts, GA + CMA-ES, CLI.
viz/        TypeScript — Three.js viewer that animates pre-computed trajectories.
runs/       Output of training runs (gitignored). Symlinked from viz/public/runs.
PLAN.md     Implementation plan, phase by phase.
```
