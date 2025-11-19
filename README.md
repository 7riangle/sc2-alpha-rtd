# New Random Tower Defense (SC2 Arcade) – AlphaZero-style Prototype 🤖

> **Status: Experimental Prototype**  
> This repository is an AI-assisted “vibe coding” experiment, not a finished or polished game AI.

---

## What This Is (and What It Is Not)

This repository contains a small **Python prototype** that applies an **AlphaZero-style search (MCTS + neural network)** to a **simplified, offline version** of the StarCraft II arcade map _“New Random Tower Defense”_.

To avoid confusion:

- It is **not** a general StarCraft II reinforcement learning agent.  
- It **does not** connect to the live SC2 client or play ladder games.  
- It **does not** represent my low-level implementation skills in RL or deep learning.

Instead, this code runs a **standalone grid-based simulation** that imitates some of the decision structure of the arcade tower defense game (e.g. tower placement along fixed mob paths), in order to test an AlphaZero-style loop in a toy setting.

---

## 🤖 AI-Assisted “Vibe Coding” Disclaimer

This project was built as an explicit **AI-assisted “vibe coding” experiment**:

- **Most of the concrete Python code was written by AI (LLMs).**  
  - MCTS scaffolding  
  - Environment wrappers  
  - Training loop boilerplate  
  - Model class structure  

- **My contribution is primarily conceptual and architectural:**
  - Describing the mechanics and constraints of **New Random Tower Defense**  
  - Proposing **state and action representations** (grid, lanes, tower slots)  
  - Designing and iterating on **reward ideas**  
  - Steering and editing AI-generated code to roughly match those ideas  

Please **do not** treat this repository as a portfolio of “pure hand-written RL code.”  
It is closer to a log of how I used AI tools to prototype and explore an idea.

---

## Overview of the Prototype

The main goals of this prototype are:

- to represent tower-defense states on a **discrete grid**,  
- to use **Monte Carlo Tree Search (MCTS)** guided by a neural network,  
- and to see whether **AlphaZero-style self-play** can discover sensible tower placement policies in a simplified offline environment.

It should be viewed as:

- experimental,
- unstable,
- and primarily educational / exploratory.

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
  - defines the grid-based state and action space  
  - encodes a simplified tower-defense reward  
  - tracks mob progress along lanes  

- **`alpha_model.py`**  
  Neural network used by MCTS:
  - policy head (action probabilities over legal tower placements)  
  - value head (estimated outcome of a state)  

- **`alpha_train.py`**  
  AlphaZero-style training loop:
  - runs self-play episodes using MCTS + current network  
  - stores game histories  
  - periodically updates the network from collected data  

- **`alpha_rtd.py`**  
  Example script for running the agent in a specific Random Tower Defense scenario.

- **`alpha_common.py`**  
  Shared helpers:
  - loading grid/waypoint CSVs  
  - configuration utilities  

- **`config.json`**  
  Central configuration for hyperparameters (learning rate, number of simulations, etc.).

Data files:

- **`mob_path_waypoints_v2.csv`**  
  Approximate creep/mob path waypoints for the offline grid simulation.

- **`grid with lane and slot.csv`**  
  Grid layout specification:
  - lane paths  
  - valid tower slots  
  - blocked cells  

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

- MCTS behaviour in terminal / edge states needs further debugging.  
- Environment code should be refactored to support flexible relative paths (e.g. `data/` directory).  
- Reward signals are simple and may need redesign to avoid degenerate behaviour.  
- Code structure could be simplified (some functions are longer than necessary).  
- No formal evaluation or baselines are implemented yet.  

---

## Scope & Intent

To be explicit:

- This project **does not** connect to the live StarCraft II client.  
- It **does not** control units or play online games.  
- It is a **standalone Python simulation** that borrows the basic idea of the _“New Random Tower Defense”_ arcade map and applies an AlphaZero-style search to a simplified grid version of that idea.  
- The code is **largely AI-generated**, based on my high-level descriptions of the game logic and reward structure.

If you are interested in this kind of experiment, feel free to fork, modify, or strip down the code for your own tower-defense or grid-based toy environments.
