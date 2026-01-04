# 🐴 AI Search Algorithms - The Horses Puzzle Solver

This project implements a comprehensive suite of **Artificial Intelligence Search Algorithms** to solve a complex grid-based navigation puzzle.
The objective is to navigate a team of pieces ("horses") from a starting configuration to specific goal positions on a 6x6 grid, avoiding obstacles and minimizing movement costs.

## 🧠 Core Logic: The "Smart" Heuristic
To optimize the search process, the project utilizes a highly advanced heuristic function based on the **Hungarian Algorithm (Linear Assignment Problem)**.

1.  **BFS Precomputation:** At initialization, the algorithm maps the shortest path distance from *every* square to *every other* square on the board (ignoring dynamic obstacles).
2.  **Optimal Assignment:** For any given board state, we calculate the cost matrix between all current horses and all goal positions.
3.  **Linear Sum Assignment:** We use `scipy.optimize.linear_sum_assignment` to find the perfect one-to-one matching that minimizes the total distance.
    * *Result:* An **admissible** and **consistent** heuristic that drastically prunes the search tree.

---

## 🚀 Algorithms Implemented

We implemented and compared 5 different search strategies:

### 1. A* Search (A-Star) ⭐
* **Type:** Informed Global Search.
* **Mechanism:** Uses $f(n) = g(n) + h(n)$ to guarantee the shortest path.
* **Key Feature:** Uses a Priority Queue to always expand the most promising node first.

### 2. Hill Climbing 🧗
* **Type:** Local Search (Greedy).
* **Mechanism:** Iteratively moves to the neighbor with the best heuristic score.
* **Enhancement:** Includes **Random Restarts** (up to 5 attempts) to escape local maxima/plateaus where the simple greedy approach gets stuck.

### 3. Simulated Annealing 🔥
* **Type:** Probabilistic Local Search.
* **Concept:** Mimics the cooling of metals. Allows "bad moves" initially (high temperature) to escape local optima, then gradually becomes stricter.
* **Math:** The probability of accepting a worse move is calculated via the Metropolis criterion: $P = e^{\Delta E / T}$.
* **Output:** The code displays the calculated probability for candidate moves during the process.

### 4. Local Beam Search 🔦
* **Type:** Parallel Local Search.
* **Mechanism:** Instead of tracking one state, it tracks the **$k$ best states** (Beam Width = 3) simultaneously.
* **Advantage:** Information is shared between the parallel threads, reducing the risk of getting stuck compared to standard Hill Climbing.

### 5. Genetic Algorithm 🧬
* **Type:** Evolutionary Computation.
* **Mechanism:** Simulates Natural Selection.
    * **Population:** Starts with a diverse pool of random paths.
    * **Selection:** Roulette Wheel selection based on fitness (heuristic).
    * **Crossover:** Merges two parent paths if they share a common board state.
    * **Mutation:** Adds random valid steps to existing paths to introduce diversity.
    * **Elitism:** Preserves the top 2 solutions for the next generation.

---

## 🎮 Game Rules & Map Legend

The board is a **6x6 grid**. Pieces move like Chess Knights ("L" shape).

| Symbol | Meaning | Cost/Rule |
| :---: | :--- | :--- |
| ` ` | **Empty** | Cost to leave: **1** |
| `&` | **Swamp** | Cost to leave: **2** (Hard terrain) |
| `@` | **Wall** | **Impassable** |
| `=` | **Water** | **Impassable** |
| `*` | **Horse** | The agent to be moved |

---

## 🛠️ Tech Stack & Requirements

* **Language:** Python 3.x
* **Libraries:**
    * `numpy`: Matrix operations.
    * `scipy`: For the Hungarian Algorithm implementation.
🏃‍♂️ How to Run
Clone the repository:

Bash

git clone [https://github.com/ShaharB11/AI-Search-Algorithms-Python.git](https://github.com/ShaharB11/AI-Search-Algorithms-Python.git)
Run the main script:

Bash

python q1.b.py
Note: You can modify the find_path call at the bottom of the script to switch between algorithms (1-5) or change the test boards.

📊 Example Output (Simulated Annealing)
Plaintext

Check: (2,2) to (3,4) | Heuristic: 4.0 | Probability: 0.135
Check: (5,5) to (4,3) | Heuristic: 2.0 | Probability: 1.0
--> Final Selected Move: (5,5) to (4,3)
-------------------------------------------

Board 1 (starting position):
   1 2 3 4 5 6
1: . . . . . .
...
📝 Author
Shahar
To install dependencies:
```bash
pip install numpy scipy
