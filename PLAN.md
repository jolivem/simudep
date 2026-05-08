# Plan — Simudep : simulateur évolutionnaire de créatures

## Contexte

Projet greenfield (le repo `/home/michel/github/simudep/` ne contient qu'un `.gitignore` Node.js) : on construit un simulateur évolutionnaire façon Karl Sims.

- **Créature** = arbre de segments ("bâtons") reliés par des joints **revolute (1 DOF)** motorisés. Topologie en arbre uniquement pour commencer.
- **Comportement** = séquence cyclique de "steps" qui imposent à chaque joint un angle cible pour une durée donnée.
- **Objectif évolutif** = maximiser la distance parcourue sur un sol plat, avec **pénalité énergétique** sur les moteurs.
- **Algorithme** = **GA** pour la topologie + **CMA-ES** pour la séquence (co-évolution).
- **Architecture** = **hybride** : entraînement Python avec exploitation GPU, visualisation web.

## Décisions actées

| Sujet | Choix | Raison principale |
|---|---|---|
| Moteur physique training | **MuJoCo MJX** | Brax est officiellement déprécié au profit de MJX ; Genesis trop jeune ; MJX = standard mature corps articulés sur GPU |
| Description du modèle | **MJCF généré dynamiquement** | Format natif MuJoCo, un MJCF par topologie unique |
| Topologies hétérogènes | **Hash topologique + grouping + JIT cache LRU** | JAX exige des shapes statiques ; on regroupe les individus à topologie identique pour `vmap` dense, fallback padding si >50 topologies/gen |
| Optim séquence | **CMA-ES** (lib `cma`) à topologie fixée | Convergence beaucoup plus rapide que mutations aléatoires sur paramètres continus |
| Visualisation | **Three.js + replay de trajectoire pré-calculée** (pas de Rapier, pas de re-simulation côté web) | Élimine le risque de divergence physique entre les deux moteurs ; déterministe, léger |
| Échange Python ↔ Web | **JSON pour génomes** + **Float32 binaire pour trajectoires** | Génomes lisibles à l'œil pour debug ; trajectoires compactes et triviales à parser en JS |
| Layout | **Monorepo** `training/` (Python) + `viz/` (TS) + `runs/` (data partagée) | Pas de monorepo manager lourd ; les deux stacks vivent côte à côte indépendamment |
| Workflow utilisateur | CLI Python pour entraîner, Vite dev server pour visualiser, **watch sur `runs/`** côté web | Simple, pas de websocket à câbler dans le MVP |
| Package manager Python | **uv** | Standard moderne, rapide |
| Modèle moteur articulaire | **PD** (`kp`, `kd`) | Suffisant ; muscles Hill = phase optionnelle |
| Sol | Plat, **friction isotrope** | Anisotrope si on observe des créatures qui glissent en arrière |
| Self-collisions | **Désactivées** dans le MVP | Comme Karl Sims original, économise du compute |
| Représentation séquence | **8 steps fixes** par individu, durées et angles optimisés par CMA-ES | Simplifie CMA-ES (vecteur de taille connue) |
| Stockage population | `individuals.jsonl.gz` par génération | Compresse bien, lisible avec `zcat` |
| Reproductibilité | Tolérance dérive bit-exact (réductions GPU non-déter) | Reproductibilité statistique seulement |

## Architecture (vue d'ensemble)

```
                    ┌──────────────────────────┐
                    │   training/  (Python)    │
                    │                          │
   GA topologie ───►│  Pop de génomes          │
                    │  ↓                       │
                    │  group by topo_hash      │
                    │  ↓                       │
   CMA-ES sequence ►│  vmap MJX rollout (GPU)  │
                    │  ↓                       │
                    │  fitness + selection     │
                    └──────────┬───────────────┘
                               │ écrit
                               ▼
                    ┌──────────────────────────┐
                    │   runs/<name>/...        │
                    │     genome.json          │
                    │     trajectory.bin       │
                    │     meta.json            │
                    └──────────┬───────────────┘
                               │ fetch (Vite watch)
                               ▼
                    ┌──────────────────────────┐
                    │   viz/  (TypeScript)     │
                    │                          │
                    │  RunPicker → genome.json │
                    │  CreatureBuilder (Three) │
                    │  Animator (qpos frames)  │
                    └──────────────────────────┘
```

Point clé : **côté web on ne re-simule pas la physique**. On reconstruit la géométrie depuis le génome, on charge `qpos(t)` (positions généralisées au cours du temps) depuis `trajectory.bin`, et on anime les nœuds Three.js frame par frame.

## Layout du repo

```
simudep/
  README.md
  pyproject.toml                 # uv, racine
  package.json                   # workspace racine (scripts d'orchestration)
  .gitignore
  CLAUDE.md

  training/
    pyproject.toml
    src/simudep/
      genome/
        types.py                 # dataclasses Genome, Sequence, Joint
        random_init.py
        mutation.py              # topo + continues
        crossover.py
        canonical.py             # hash topologique canonique
      mjcf/
        builder.py               # genome -> MJCF XML
        actuator.py              # config moteurs PD
      sim/
        rollout.py               # mjx.step jitté + vmap
        grouping.py              # groupby topo_hash, JIT cache LRU
        fitness.py               # distance - alpha * energy
      evo/
        ga.py
        cmaes.py                 # via lib `cma`
        loop.py                  # co-évolution GA/CMA-ES
      io/
        run_writer.py
        trajectory.py            # qpos -> Float32 .bin
        genome_json.py
      cli/
        __main__.py              # `python -m simudep ...`
        train.py
        replay.py                # rollout déterministe d'un individu sélectionné
        inspect.py
    tests/
      test_canonical.py
      test_mjcf_roundtrip.py
      test_rollout_determinism.py

  viz/
    package.json
    tsconfig.json
    vite.config.ts
    index.html
    src/
      main.ts
      ui/
        RunPicker.ts
        Controls.ts
      scene/
        Scene.ts
        CreatureBuilder.ts       # genome.json -> Three.Group
        Animator.ts               # applique qpos(t) frame par frame
      io/
        loadRun.ts
        types.ts                 # types miroir du schéma JSON
    public/
      runs -> ../../runs         # symlink vers le dossier partagé

  runs/                          # gitignored
    .gitkeep

  scripts/
    new_run.sh
```

## Schémas d'échange

### `genome.json`
```json
{
  "version": 1,
  "id": "ind_0042_007",
  "topology_hash": "a7f3...",
  "root": {
    "size": [0.3, 0.1, 0.1],
    "mass": 1.5,
    "color": [0.8, 0.4, 0.2],
    "children": [
      {
        "joint": {
          "axis": [0, 0, 1],
          "anchor_parent": [0.15, 0, 0],
          "anchor_child": [-0.1, 0, 0],
          "range": [-1.57, 1.57],
          "kp": 50.0, "kd": 2.0
        },
        "segment": { "size": [0.2, 0.05, 0.05], "mass": 0.4, "color": [...] },
        "children": []
      }
    ]
  },
  "sequence": {
    "cycle_duration": 2.0,
    "steps": [
      { "duration": 0.5, "targets": [0.3, -0.2, 0.0, 0.5] },
      ...
    ]
  },
  "fitness": { "distance": 12.34, "energy": 5.6, "score": 6.74 }
}
```

`targets` ordonné par DFS canonique de l'arbre. Le web réplique cet ordre.

### `trajectory.bin`
`Float32Array` packé : `[n_frames, 7 + n_joints]` row-major.
- 7 = position root (3) + quaternion root (4)
- Puis l'angle de chaque joint dans l'ordre DFS canonique

Pour 30 s à 60 Hz × 12 joints ≈ 50 KB. `meta.json` contient `dt`, `n_frames`, `n_joints`, `body_names`.

## Roadmap par phases

Critère "fait" mesurable et démontrable seul à chaque phase.

### Phase 0 — Bootstrap & vérification GPU (1-2 j)
- `uv init` + `pyproject.toml` (mujoco, mujoco-mjx, jax[cuda12], cma, numpy).
- `viz/` Vite + TS + Three.js, scène vide avec un cube qui tourne.
- CI minimal : `ruff`, `tsc --noEmit`, `pytest -q` (vide pour l'instant).
- **Vérification GPU critique** : `uv run python -c "import jax; print(jax.devices())"` doit afficher au moins `[CudaDevice(0)]`. **Si KO**, on bascule sur Option A (100% web CPU) — pas de blocage : la stack training Python est isolée.
- **Fait quand** : commande GPU OK, `npm --prefix viz run dev` ouvre la scène Three.js, CI verte sur un commit vide.

### Phase 1 — Genome → MJCF → simulation single creature (3-4 j)
- `genome/types.py` : dataclasses arborescentes (`Segment`, `Joint`, `Genome`, `Sequence`).
- `mjcf/builder.py` : `Genome → str` MJCF valide (un body/segment, joint revolute, actuator general PD).
- Test roundtrip : MuJoCo charge sans warning, simulation 5 s stable.
- Tétrapode hardcodé pour validation visuelle.
- `io/genome_json.py` + `io/trajectory.py` : exporter JSON + binaire.
- `viz/scene/CreatureBuilder.ts` + `Animator.ts` : reconstruire l'arbre Three.js, appliquer qpos(t).
- **Fait quand** : un script `inspect_one.py` simule le tétrapode 5 s, exporte un run, et le viewer web l'anime correctement (pattes synchrones avec le sim Python).

### Phase 2 — Boucle évolutive minimale (4-5 j)
- `genome/random_init.py`, `mutation.py` (topo + continues), `crossover.py`.
- `genome/canonical.py` : hash canonique de la topologie.
- `evo/ga.py` : GA simple (tournament, élitisme, pop=32, 30 gens).
- `sim/rollout.py` : version naïve sans batching pour démarrer.
- `sim/fitness.py` : `distance_x - alpha * Σ(action² · dt)`.
- CLI `simudep train --pop 32 --gens 30 --out runs/test1`.
- **Fait quand** : courbe de fitness moyenne strictement croissante en fenêtre glissante ; le top-1 final avance >2× plus loin que la moyenne du gen 0.

### Phase 3 — Parallélisation MJX par groupe topologique (3-4 j)
- `sim/grouping.py` : groupby `topo_hash`.
- `sim/rollout.py` : `jit(vmap(rollout_single))` par groupe, JIT cache LRU (50 entrées).
- Mesure : temps gen pour pop=128 sur GPU vs séquentiel CPU.
- **Fait quand** : speedup ≥10× sur GPU vs séquentiel pour pop=128 ; 50 gens × pop=128 finissent en <10 min.

### Phase 4 — CMA-ES sur séquence (3 j)
- `evo/cmaes.py` : wrapping de la lib `cma`, optimise les `targets[]` + `duration[]` de chaque step.
- Boucle co-évolutive : pour chaque individu GA, K itérations CMA-ES (typiquement K=10) avant scoring.
- Warm-start : si la topologie ne change pas parent→enfant, on initialise CMA-ES depuis la séquence du parent.
- L'évaluation des `n_samples` candidates CMA-ES exploite vmap MJX.
- **Fait quand** : sur une topologie fixée, CMA-ES améliore la fitness >50 % en 20 itérations vs init aléatoire ; intégré dans la boucle principale sans exploser le runtime.

### Phase 5 — UI sélection + replay (3-4 j)
- CLI `simudep replay --run runs/test1 --gen 49 --rank 0` : refait un rollout déterministe et écrit `selected/<id>/{genome.json, trajectory.bin, meta.json}`.
- `viz/ui/RunPicker.ts` : lister `runs/*/`, dropdown génération + rang.
- `viz/ui/Controls.ts` : play/pause, vitesse, scrubber.
- Watch sur `runs/` côté Vite (HMR auto-refresh).
- **Fait quand** : l'utilisateur clique un individu dans l'UI et voit sa créature marcher 5 s en boucle, fluide à 60 fps.

### Phase 6 — Polish & stabilité (4-5 j)
- Validation déterminisme : même seed → mêmes fitness à tolérance près (1e-3, dérive GPU acceptée).
- Checkpoint/resume : `--resume runs/xxx`.
- Logs structurés JSONL, graphes fitness (matplotlib → PNG dans le run).
- README détaillé : "0 → première créature qui marche en <30 min sur ta machine".
- **Fait quand** : un tiers peut suivre le README et obtenir une viz de créature évoluée sur la même machine.

### Phases optionnelles (post-MVP)
- **Phase 7** — Sol non plat, friction anisotrope, multi-fitness (jump, swim).
- **Phase 8** — UI web pour piloter un training (FastAPI + websocket pour streamer la fitness en live).
- **Phase 9** — Topologies en graphe (cycles), joints 3 DOF (spherical), self-collisions.

## Fichiers structurants à écrire en premier

Ils définissent les contrats inter-modules :

- [training/src/simudep/genome/types.py](training/src/simudep/genome/types.py) — modèle de données central
- [training/src/simudep/mjcf/builder.py](training/src/simudep/mjcf/builder.py) — pont génome → physique
- [training/src/simudep/sim/rollout.py](training/src/simudep/sim/rollout.py) — boucle de simulation jit/vmap
- [training/src/simudep/io/genome_json.py](training/src/simudep/io/genome_json.py) — schéma d'échange (autorité)
- [viz/src/scene/CreatureBuilder.ts](viz/src/scene/CreatureBuilder.ts) — reconstruction côté web

## Vérification end-to-end

À l'issue de la Phase 5 :

1. **Smoke test stack training** :
   ```
   uv run python -c "import jax, mujoco, mujoco.mjx; print(jax.devices())"
   ```
   doit lister un device CUDA.

2. **Run d'entraînement court** :
   ```
   uv run simudep train --pop 32 --gens 20 --out runs/smoke --seed 42
   ```
   Vérifier : `runs/smoke/generations.jsonl` montre une fitness max strictement croissante en fenêtre, dossier `population/gen_*/individuals.jsonl.gz` lisible avec `zcat`.

3. **Replay déterministe** :
   ```
   uv run simudep replay --run runs/smoke --gen 19 --rank 0
   ```
   Produit `runs/smoke/selected/<id>/{genome.json,trajectory.bin,meta.json}`.
   Lancer 2× la même commande → fichiers binaires identiques (à 1e-3 près après reload).

4. **Viz** :
   ```
   npm --prefix viz run dev
   ```
   Ouvrir `http://localhost:5173`, sélectionner `runs/smoke` → `gen 19` → `rank 0`. La créature doit s'animer en boucle, à 60 fps, pendant la durée définie dans `meta.json`.

5. **Tests automatisés** :
   ```
   cd training && uv run pytest -q
   cd ../viz && npm test
   ```
   Tous verts. En particulier `test_rollout_determinism.py` (même seed = même fitness) et `test_canonical.py` (deux génomes structurellement identiques → même hash).

## Risques & mitigations

| Risque | Mitigation |
|---|---|
| GPU pas NVIDIA / CUDA pas installé | Vérification dès Phase 0. Fallback sur Option A 100% web si bloquant. |
| >50 topologies par génération → JIT cache thrashing | Fallback option A : padder à `max_segments=12` avec masque sur joints désactivés. À surveiller en Phase 3. |
| MJX rollout non bit-exact entre runs | Tolérance 1e-3 sur fitness ; reproductibilité statistique (mêmes courbes GA) suffit. |
| Créatures qui exploitent des bugs physiques (ex: vibrations exploitant l'intégrateur) | Solver MuJoCo `implicit` + dt raisonnable (1ms). Inspection visuelle des top performers en Phase 5. |
| Énergie = 0 favorise l'inaction | Régler `alpha` empiriquement ; commencer petit (1e-3) puis augmenter. Si ça reste mou, multi-fitness avec floor sur la distance. |
