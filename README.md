# StarCraft II RL Agent: AlphaZero-style Prototype

> **Status: Work In Progress (Experimental)**  
> Current version is highly unstable and under active development.

## Overview

This repository hosts an experimental Reinforcement Learning (RL) agent designed for a StarCraft II-based custom environment (Random Tower Defense / RTD scenario). The project implements an **AlphaZero-style architecture**, combining **Monte Carlo Tree Search (MCTS)** with a deep neural network for policy and value estimation.

The primary goal is to explore complex reward shaping and pathfinding logic within a custom grid environment inspired by the **“New Lottery Defense / New Random Tower Defense”** arcade map mechanics.

---

## Disclaimer: AI-Assisted “Vibe Coding” Project

Please note that this project is intentionally treated as a **“vibe coding” experiment**:

- **AI-assisted implementation:**  
  A significant portion of the codebase was generated with LLM-based tools to rapidly prototype ideas (e.g. MCTS logic, environment wrappers, training loop scaffolding).

- **Human role – architecture & design:**  
  My main focus has been on **designing reward functions, defining state/action representations, and structuring the overall logic**, while offloading low-level boilerplate and syntax to AI.

- **Prototype state (not production-ready):**  
  The code is in a **raw research-prototype stage**. It likely contains:
  - Bugs
  - Unoptimized routines
  - Redundant or dead code

  Treat this repository as a **research log / sandbox**, not as polished or stable software.

---

## File Structure

Core components:

- **`alpha_mcts.py`**  
  Implementation of Monte Carlo Tree Search:
  - Selection
  - Expansion
  - Simulation
  - Backpropagation

- **`alpha_env.py`**  
  Custom environment wrapper:
  - Defines state space and action space
  - Encodes reward logic
  - Interfaces with the RTD-style game state

- **`alpha_model.py`**  
  Neural network architecture definition for:
  - Policy head (action probabilities)
  - Value head (state value estimation)

- **`alpha_train.py`**  
  Main training loop and self-play orchestration:
  - Generates self-play games using MCTS + current network
  - Stores experience for training
  - Periodically updates the model from replay data

- **`alpha_rtd.py`**  
  Entry point for running the agent in a specific RTD / tower defense scenario.

- **`alpha_common.py`**  
  Shared utilities and constants:
  - Path handling
  - Grid / waypoint loading
  - Config helpers

- **`config.json`**  
  Central configuration for hyperparameters:
  - Learning rate
  - Number of MCTS simulations
  - Replay buffer sizes
  - Training schedule, etc.

- **`mob_path_waypoints_v2.csv`**  
  Waypoint data that defines creep/mob movement paths.

- **`grid with lane and slot.csv`**  
  Grid layout specification for the map:
  - Lane definitions
  - Valid tower slots
  - Blocked cells / walls

---

## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/7riangle/sc2-rtd-alphazero.git
cd sc2-rtd-alphazero
```

---

### 2. Install dependencies

Make sure you have **Python 3.8+** installed.

If you have a `requirements.txt` file (recommended):

```bash
pip install -r requirements.txt
```

Or install core dependencies manually:

```bash
pip install numpy pandas gymnasium torch
```

Depending on your setup, you may also need:

```bash
pip install matplotlib tqdm
```

---

### 3. Data / File Placement (Important)

**CSV file locations are currently hard-coded.**

To run the code successfully:

- Ensure that all CSV data files are placed in the **repository root**, e.g.:

  - `mob_path_waypoints_v2.csv`  
  - `grid with lane and slot.csv`

- The scripts currently assume these files are in the **current working directory**.

> Do **not** move these CSVs into a separate `data/` folder unless you also update the paths in `alpha_common.py` (or any other script that loads them).

---

### 4. Run the Agent

There are two main entry points, depending on what you want to test.

#### 4.1 Start the training loop (self-play)

```bash
python alpha_train.py
```

This launches the AlphaZero-style pipeline:

- Self-play episodes using MCTS + current network
- Replay buffer filling
- Periodic network updates

#### 4.2 Run the RTD scenario

```bash
python alpha_rtd.py
```

This runs the agent in the specific **Random Tower Defense** scenario with the current model parameters and environment logic.

---

## 🛠️ Planned Improvements (To-Do)

- [ ] Debug MCTS expansion logic in edge cases (e.g. terminal or illegal states).
- [ ] Refactor environment path handling to support relative `data/` directories.
- [ ] Optimize reward function design to avoid degenerate local optima.
- [ ] Remove redundant imports and dead code; split monolithic scripts where needed.
- [ ] Add basic evaluation scripts and simple baselines for comparison.

---

## Project Scope & Philosophy

This repository is a **sandbox** for experimenting with:

- AlphaZero-style RL on custom, combinatorial environments
- Reward shaping for strategic/tactical decision-making
- Connections between **computational modelling habits** in linguistics/phonology and RL-based simulation in games

Feedback and issues are welcome — but please be aware that:
- The API may change without warning,
- Code is intentionally exploratory,
- There are no stability guarantees.

Use at your own risk, tinker freely, and feel free to fork and adapt for your own experiments 🚀
