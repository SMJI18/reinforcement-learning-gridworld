# Reinforcement Learning Gridworld (MDP)

This project implements a Gridworld reinforcement learning environment to study policy learning and convergence behavior using classical reinforcement learning algorithms.

The system models a **Markov Decision Process (MDP)** where an agent navigates a grid containing walls, bombs, and a goal state while learning optimal actions.

The project was developed as part of research work accepted at **ICACIT 2026**.

---

## Algorithms Implemented

The following reinforcement learning and dynamic programming algorithms are implemented:

* **Value Iteration**
* **Policy Iteration**
* **Q-Learning**

These algorithms are compared in terms of:

* Policy convergence
* Episodic return
* Goal reach rate
* Learning stability

---

## Environment

The environment is a **4×4 Gridworld** with configurable scenarios.

Each scenario may contain:

* Walls (impassable cells)
* Bombs (negative reward terminal states)
* Goal state (positive reward)
* Step penalty to encourage efficient paths

The agent learns an optimal policy to reach the goal while avoiding bombs.

---

## Scenarios Tested

The system evaluates agent behavior across multiple environments:

1. **Single Wall Scenario**
2. **Walls + Bomb Scenario**
3. **Corner Bomb Scenario**
4. **Narrow Passage Scenario**

These scenarios test different decision-making conditions such as risk avoidance and constrained navigation.

---

## Evaluation Metrics

The algorithms are evaluated using:

* **Goal Reach Percentage**
* **Average Episodic Return**
* **Standard Deviation of Returns**
* **Policy Convergence Behavior**

The program generates comparison results for each scenario.

---

## Visualizations

The project generates multiple visual outputs for analysis:

* Value function heatmaps
* Policy maps
* Convergence graphs
* Q-learning reward curves

All generated figures are saved in the **outputs/** directory.

---

## Technologies Used

* Python
* NumPy
* Matplotlib

---

## Running the Project

Install dependencies:

```bash
pip install numpy matplotlib
```

Run the simulation:

```bash
python MDP_GridWorld_agent.py
```

Output visualizations will be saved in the **outputs/** directory.

---

## Example Output

The system produces visualizations such as:

* Value heatmaps showing learned state values
* Optimal policy maps
* Algorithm convergence plots

These help compare how different reinforcement learning algorithms behave in the same environment.

---

## Research Context

This project is part of research work titled:

**"Simulation of a Self-Learning Grid World Agent using Markov Decision Processes and Reinforcement Learning"**

Accepted at **ICACIT 2026**.

---

## Author

Snehil Mishra
Computer Science Undergraduate
KLE Technological University
