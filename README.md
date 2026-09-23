# flybrain

**Reinforcement learning on a biologically-inspired fly connectome.**

`flybrain` trains a simulated fruit fly to navigate a procedurally generated
obstacle course by routing its sensory input through a small neural circuit
modeled on real *Drosophila* neuroanatomy — a compound-eye vision layer, a
heading-compass ring, a steering relay, and three descending motor
outputs — instead of a generic feed-forward policy network.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Visualizing a Trained Agent](#visualizing-a-trained-agent)
- [Model Checkpoints](#model-checkpoints)
- [Background: Real vs. Synthetic Connectome](#background-real-vs-synthetic-connectome)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

The agent flies down a long corridor scattered with procedurally placed
buildings. Its only sensory input is a set of ray-cast distance readings (a
stand-in for a compound eye). Those readings are projected into an
83-neuron biologically-themed graph — loosely modeled on the fly's optic
flow → central-complex → descending-neuron pathway — and a Soft Actor-Critic
(SAC) agent learns to control the strength of that circuit's synapses,
along with a small policy head, to steer the fly safely and efficiently to
the far end of the course.

This is a research/demo project, not a biological simulator: the physics,
vision, and neural circuit are all deliberately simplified so the whole
pipeline trains on a single machine.

## How It Works

```
Ray-cast vision  →  Synthetic connectome (83 neurons)  →  SAC policy  →  Wing/steer commands
   (fly_env.py)         (connectome_data.py)               (train.py)      (fly_env.py)
```

1. **Environment** (`fly_env.py`) — a Gymnasium environment simulating
   simplified flight physics, a ray-cast compound eye, and a procedurally
   generated field of obstacles.
2. **Connectome** (`connectome_data.py`) — builds and caches the 83-neuron
   adjacency graph the actor's brain module is built from (vision → heading
   ring → steering bridge → 3 descending motor neurons).
3. **Training** (`train.py`) — trains the connectome-based actor with Soft
   Actor-Critic. Learns per-synapse gains on the fixed graph plus a small
   downstream policy network; logs to TensorBoard.
4. **Visualization** (`live_game_view.py`, `run_mujoco_simulation.py`) —
   replays a trained checkpoint in a 3D scene (Ursina or native MuJoCo)
   alongside a live dashboard of which neurons are firing.

## Project Structure

```
flybrain/
├── connectome_data.py         # Builds the synthetic 83-neuron connectome
├── download_flywire_data.py   # Fetches real FlyWire connectome annotations
├── fly_env.py                 # Gymnasium flight environment
├── train.py                   # SAC training loop for the connectome actor
├── live_game_view.py          # Ursina-based 3D live viewer + neuron dashboard
├── run_mujoco_simulation.py   # Native MuJoCo viewer for a trained agent
├── requirements.txt           # Python dependencies
├── .gitignore
│
├── fly_brain_env/             # Local virtual environment (not tracked)
├── fly_tensorboard/           # TensorBoard run logs (not tracked)
└── *.pt                       # Cached connectome / trained checkpoints (not tracked)
```

Checkpoints and cached tensors (`cx_connectome.pt`, `real_fly_circuit.pt`,
`fly_connectome_actor.pt`, `fly_connectome_actor_V4.pt`) are listed in
`.gitignore` and generated locally the first time you run the scripts — they
are not committed to the repository.

## Installation

Requires **Python 3.10+**.

```bash
git clone https://github.com/<your-username>/flybrain.git
cd flybrain

# Create and activate a virtual environment
python -m venv fly_brain_env
source fly_brain_env/Scripts/activate      # Windows (Git Bash)
# fly_brain_env\Scripts\activate.bat       # Windows (cmd)
# source fly_brain_env/bin/activate        # macOS/Linux

pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py
```

This will:
- build (or load a cached copy of) the synthetic connectome via
  `connectome_data.py`,
- train the connectome-based actor with SAC against `fly_env.py`,
- write TensorBoard logs to `fly_tensorboard/`,
- save the trained weights to a `.pt` checkpoint.

Monitor training with:

```bash
tensorboard --logdir fly_tensorboard
```

### Visualizing a Trained Agent

Once a checkpoint exists, replay it in 3D:

```bash
python live_game_view.py            # Ursina-based scene + live neuron dashboard
# or
python run_mujoco_simulation.py     # Native MuJoCo viewer
```

Both scripts load the same trained actor checkpoint and render the fly
navigating the obstacle course while a side panel shows which of the 83
neurons are firing in real time.

## Model Checkpoints

| File | Produced by | Contents |
|---|---|---|
| `cx_connectome.pt` / `real_fly_circuit.pt` | `connectome_data.py` | The fixed 83×83 synthetic connectome adjacency matrix (cached so it isn't rebuilt every run). |
| `fly_connectome_actor.pt` | `train.py` | Trained actor weights (synaptic gains + policy head) from the current training run. |
| `fly_connectome_actor_V4.pt` | `train.py` | A versioned/snapshot checkpoint from an earlier training run, kept for comparison. |

None of these are tracked in git — regenerate them locally by running
`train.py`.

## Background: Real vs. Synthetic Connectome

The real adult *Drosophila* brain (mapped by the FlyWire consortium) has
roughly **140,000 neurons** and **~50 million synapses**. The circuit used
in this project is an **83-neuron simplification** of one real, well-studied
motif in that brain:

| Real biology | This project |
|---|---|
| Dozens of Lobula Plate Tangential Cell types reading compound-eye motion | 16 generic "optic flow" input neurons |
| E-PG ring-attractor heading compass (ellipsoid body) | 32-neuron heading ring |
| P-EN steering relay (protocerebral bridge) | 32-neuron steering bridge |
| ~1,300 descending neurons, ~200 Hz haltere feedback | 3 descending motor neurons, ~25 Hz control loop |

`download_flywire_data.py` fetches real FlyWire cell-type annotations for
reference; the trainable connectome itself remains a hand-authored,
biologically-themed graph rather than the literal measured wiring, since
simulating the full connectome at every environment step is well beyond
what this project's training loop is built for.

## Limitations

- Flight physics is a simplified point-mass model, not real aerodynamics or
  muscle dynamics.
- Vision is a small set of ray-cast distances, not an image or photoreceptor
  array.
- The connectome is fixed in structure; only synapse *strength* is learned,
  not the wiring itself.
- Designed to run end-to-end on a single machine — not a claim of
  biological fidelity.

## Roadmap

- [ ] Wire real FlyWire connectivity data into the trainable circuit
      (currently downloaded but not yet used by training)
- [ ] Integrate a real biomechanical fly body (e.g. DeepMind's `flybody`)
      once the connectome-driven policy is trained end-to-end
- [ ] Expand the descending-neuron set beyond the current 3 output channels

## License

No license has been added to this repository yet. If you intend to publish
or share it, add a `LICENSE` file (e.g. MIT, Apache-2.0) before doing so.