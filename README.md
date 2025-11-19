# StarCraft II RL Agent: AlphaZero-style Prototype 🤖

> 🚧 **Status: Work In Progress (Experimental)**
>
> *Current version is highly unstable and under active development.*

## 📌 Overview
This repository hosts an experimental Reinforcement Learning (RL) agent designed for a StarCraft II-based custom environment (RTD/Tower Defense scenario). The project implements an **AlphaZero-style architecture**, utilizing **Monte Carlo Tree Search (MCTS)** combined with a deep neural network for value and policy approximation.

The primary goal of this project is to explore complex reward shaping and pathfinding logic within a custom grid environment.

## ⚠️ Disclaimer: AI-Assisted "Vibe Coding" Project

**Please note:** This project represents a **"Vibe Coding" experiment**.
* **AI-Assisted Implementation:** A significant portion of the codebase was generated using LLM-based tools to rapidly prototype theoretical ideas (e.g., MCTS logic, custom environment wrappers).
* **Focus on Architecture:** My primary role has been **architecting the reward functions, defining state representations, and directing the logical flow**, while leveraging AI for syntax implementation.
* **Current State:** The code is currently in a **raw, prototype stage**. It contains known bugs, unoptimized routines, and redundant code blocks. It is intended as a **research log** rather than production-ready software.

## 📂 File Structure

* **`alpha_mcts.py`**: Implementation of Monte Carlo Tree Search (Selection, Expansion, Simulation, Backpropagation).
* **`alpha_env.py`**: Custom environment wrapper defining the state space, action space, and reward logic.
* **`alpha_model.py`**: Neural network architecture definition.
* **`alpha_train.py`**: Main training loop and self-play logic.
* **`alpha_rtd.py`**: Main entry point for the specific RTD scenario.
* **`config.json`**: Centralized configuration for hyperparameters (learning rate, simulation counts, etc.).
* **`mob_path_waypoints_v2.csv`**: Waypoint data defining mob movement paths.
* **`grid with lane and slot.csv`**: Grid layout definitions for the environment.

## ⚙️ Setup & Usage Note

**⚠️ Important: File Placement**
To run this code, please ensure that all data files (`.csv`) are placed in the **same directory (root)** as the Python scripts.
* The scripts are currently hardcoded to look for `mob_path_waypoints_v2.csv` and `grid with lane and slot.csv` in the current working directory.
* **Do not move CSV files to a separate `data/` folder** without modifying `alpha_common.py`.

## 🛠️ Planned Improvements (To-Do)
* [ ] Debugging MCTS expansion logic in edge cases.
* [ ] Refactoring environment paths to support relative directories.
* [ ] Optimizing the reward function to prevent local optima.
* [ ] Cleaning up redundant dependencies.

---
*This repository is a sandbox for testing computational phonology and RL concepts. Feedback is welcome, but please expect breaking changes.*
