# New Random Tower Defense (SC2 Arcade) – AlphaZero-style Prototype 🤖

> **Status: Experimental Prototype**  
> This code is work-in-progress and not a finished game AI.

## Overview

This repository contains a small **Python prototype** that applies an **AlphaZero-style search (MCTS + neural network)** to a **simplified, offline version of the StarCraft II arcade map _“New Random Tower Defense”_**.

- It is **not** a general StarCraft II reinforcement learning agent.  
- It does **not** play full SC2 matches or interact with the SC2 engine directly.  
- Instead, it runs on a **custom grid-based environment** that approximates some of the decision structure of the arcade tower defense game (e.g. tower placement choices along fixed mob paths).

The main goal of this project is to experiment with:
- representing tower-defense states as a discrete grid,
- using **Monte Carlo Tree Search (MCTS)** guided by a neural network,
- and testing whether AlphaZero-style self-play can learn sensible tower placement policies in this toy setting.

This repository should be viewed as a **personal experiment / learning project**, not as a polished or competitive AI.

---

## Notes on Implementation

- A fair amount of the boilerplate (e.g. MCTS scaffolding, environment wrappers) was written with the help of LLM-based tools, then manually edited.
- My focus has been on:
  - defining the basic **state representation** (grid, lanes, tower slots),
  - sketching **reward signals**,
  - and wiring up an AlphaZero-style loop at a prototype level.
- The code is still rough:
  - some parts may be redundant or unoptimized,
  - there may be logical bugs in edge cases,
  - and hyperparameters are not tuned.

Feel free to treat this as a **starting point or reference**, not as a final implementation.

---

## File Structure

Core files:

- **`alpha_mcts.py`**  
  Monte Carlo Tree Search implementation:
  - selection  
  - expansion  
  - simulation  
  - backpropagation  

- **`alpha_env.py`**  
  Custom environment wrapper:
  - defines the grid-based state and action space,
  - encodes a simplified tower-defense reward,
  - tracks mob progress along lanes.

- **`alpha_model.py`**  
  Neural network used by the MCTS policy/value guidance:
  - policy head (action probabilities over legal tower placements),
  - value head (estimated outcome of a state).

- **`alpha_train.py`**  
  AlphaZero-style training loop:
  - self-play episodes using MCTS + current network,
  - storage of game histories,
  - periodic training steps on collected data.

- **`alpha_rtd.py`**  
  Example script for running the agent in a specific Random Tower Defense scenario.

- **`alpha_common.py`**  
  Shared helpers:
  - loading grid/waypoint CSVs,
  - common configuration utilities.

- **`config.json`**  
  Central configuration for hyperparameters (learning rate, number of simulations, etc.).

Data files:

- **`mob_path_waypoints_v2.csv`**  
  Approximate creep/mob path waypoints for the offline grid simulation.

- **`grid with lane and slot.csv`**  
  Grid layout specification:
  - lane paths,
  - valid tower slots,
  - blocked cells.

---

## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/7riangle/sc2-rtd-alphazero.git
cd sc2-rtd-alphazero
```

---

### 2. Install dependencies

Requires **Python 3.8+**.

If you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install core libraries manually:

```bash
pip install numpy pandas gymnasium torch
```

Optionally, for plotting and progress bars:

```bash
pip install matplotlib tqdm
```

---

### 3. Data / File Placement

Currently, the code expects the CSV files to be in the **repository root directory**:

- `mob_path_waypoints_v2.csv`  
- `grid with lane and slot.csv`

If you move them into a `data/` folder, you will need to update the paths in `alpha_common.py` (and any other script that loads these files).

---

### 4. Running the Prototype

#### 4.1 Training (self-play)

```bash
python alpha_train.py
```

This starts an AlphaZero-style self-play loop in the offline grid environment:

- the agent generates games using MCTS + the current network,  
- game data is stored,  
- and the network is updated from these examples.

#### 4.2 Running a scenario

```bash
python alpha_rtd.py
```

This runs the agent in a configured tower-defense scenario, using the current model parameters and environment settings.

---

## Current Limitations & To-Do

- [ ] MCTS behaviour in terminal/edge states needs further debugging.  
- [ ] Environment code should be refactored to support flexible relative paths (e.g. `data/` directory).  
- [ ] Reward signals are very simple and may need redesign to avoid degenerate behaviour.  
- [ ] Code structure could be simplified (some functions are longer than necessary).  
- [ ] No formal evaluation or baselines are implemented yet.

---

## Scope

To avoid confusion:

- This project **does not** connect to the live StarCraft II client.  
- It **does not** control units or play ladder games.  
- It is a **standalone Python simulation** that borrows the basic idea of the **“New Random Tower Defense”** arcade map and applies an AlphaZero-style loop to a simplified grid version of that idea.

If you are interested in this kind of experiment, feel free to fork, modify, or strip down the code for your own tower-defense or grid-based toy environments.
