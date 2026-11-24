# HW6 submission

## AI-use declaration

I used **Google Gemini** as an AI collaborator during this assignment for:
- Reviewing and double-checking my Part A hand calculations and MDP explanations after I write them by myself on paper.
- Helping with Python scripts for Part B (`part_b_frozenlake.py`) and Part C (`part_c_solver.py`).
- Refining the English phrasing and formatting of my analysis in `SUBMISSION.md` based on my own observed experimental outputs.

All code was executed locally in my PyCharm environment, all outputs in `outputs/` were generated from my own runs, and all calculations and conclusions were verified against the assignment specification.

## Setup and reproduction

- **Python:** 3.14.0
- **gymnasium:** 1.4.0
- **numpy:** 2.5.5

```bash
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
pip install gymnasium==1.4.0

# Run Part B (FrozenLake simulations)
python part_b_frozenlake.py

# Run Part C (Value Iteration solver, rule checks, and simulations)
python part_c_solver.py
```

## Part A — Paper calculations

 /Part1 (Path from Repository Root)

## Part B — FrozenLake

### Fixed policy, written before the experiments

In `FrozenLake-v1` (`4x4`), actions are encoded as `0 = LEFT`, `1 = DOWN`, `2 = RIGHT`, `3 = UP`. States are indexed row by row from `0` (top-left `S`) to `15` (bottom-right `G`), with holes at `5, 7, 11, 12`.

| State | Row, Col | Tile | Action | Reasoning |
| --- | --- | --- | --- | --- |
| 0 | (0, 0) | S | DOWN (1) | Follow the safe left column (`0 -> 4 -> 8`) away from holes `5` and `7`. |
| 1 | (0, 1) | F | RIGHT (2) | Avoid hole `5` directly below; move toward the central column (`2 -> 6 -> 10 -> 14`). |
| 2 | (0, 2) | F | DOWN (1) | Shortest path downward along column 2 toward the goal. |
| 3 | (0, 3) | F | LEFT (0) | Avoid hole `7` directly below; rejoin column 2 at state `2`. |
| 4 | (1, 0) | F | DOWN (1) | Avoid hole `5` to the right; continue down the left edge toward row 2. |
| 6 | (1, 2) | F | DOWN (1) | Avoid holes `5` (left) and `7` (right); move straight down toward `10`. |
| 8 | (2, 0) | F | RIGHT (2) | Avoid hole `12` directly below; turn right onto the safe middle corridor. |
| 9 | (2, 1) | F | DOWN (1) | Move away from hole `5` above and enter the bottom row at `13`. |
| 10 | (2, 2) | F | DOWN (1) | Avoid hole `11` to the right; step directly above the final corridor at `14`. |
| 13 | (3, 1) | F | RIGHT (2) | Move away from hole `12` to the left toward `14`. |
| 14 | (3, 2) | F | RIGHT (2) | Step directly into the goal `G` at state `15`. |

**Overall reasoning:** Every nonterminal state is mapped to a cycle-free shortest safe path to `G` (`0 -> 4 -> 8 -> 9 -> 13 -> 14 -> 15` from the start, taking 6 steps deterministically), while states outside the main path (reachable when slipping occurs) also recover safely toward `G`. This policy is kept strictly unchanged across all four experimental conditions.

### Results

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| False | random | 11 | 0 | 50 | 0 | 0.0% | 8.14 |
| False | random | 22 | 1 | 49 | 0 | 2.0% | 7.56 |
| False | random | 33 | 0 | 50 | 0 | 0.0% | 6.88 |
| False | fixed | 11 | 50 | 0 | 0 | 100.0% | 6.00 |
| False | fixed | 22 | 50 | 0 | 0 | 100.0% | 6.00 |
| False | fixed | 33 | 50 | 0 | 0 | 100.0% | 6.00 |
| True | random | 11 | 0 | 50 | 0 | 0.0% | 8.06 |
| True | random | 22 | 0 | 50 | 0 | 0.0% | 7.60 |
| True | random | 33 | 0 | 50 | 0 | 0.0% | 7.98 |
| True | fixed | 11 | 1 | 49 | 0 | 2.0% | 4.24 |
| True | fixed | 22 | 2 | 48 | 0 | 4.0% | 5.48 |
| True | fixed | 33 | 1 | 49 | 0 | 2.0% | 4.44 |

**Saved artifacts:**
- All 600 per-episode rows: [`outputs/part_b_episodes.csv`](outputs/part_b_episodes.csv)
- Text-rendered episode: [`outputs/part_b_rendered_episode.txt`](outputs/part_b_rendered_episode.txt)
- Example successful and failed traces: [`outputs/part_b_traces.txt`](outputs/part_b_traces.txt)

**Analysis:**
- **What slipping changed:** With `is_slippery=False`, the hand-crafted fixed policy reaches `G` in 100.0% of episodes (50/50 in every batch) in exactly 6.00 steps, while the random policy reaches `G` only once across all 150 episodes (0.7% overall). Turning `is_slippery=True` completely breaks the deterministic shortest-path fixed policy: success collapses from 100.0% down to 2.0%–4.0% (only 4 goals out of 150 episodes), and mean episode length drops from 6.00 to 4.24–5.48 steps because perpendicular slips push the agent straight into nearby holes (`5`, `7`, `11`, `12`). For the random policy, slipping leaves success near zero (0/150 goals) with similar mean episode lengths (7.60–8.06 steps).
- **Consistency across batches:** Deterministic fixed runs have zero variance across batches (50/50 and 6.00 steps every time). Under stochasticity (`random` or `is_slippery=True`), batch outcomes fluctuate: for example, `is_slippery=True, fixed` ranges from 1 to 2 goals per 50 episodes (2.0%–4.0%) and mean steps range from 4.24 to 5.48, while `is_slippery=False, random` varies between 0 and 1 goal (0.0%–2.0%). No episodes hit the 100-step time limit in any condition because holes absorb wandering trajectories quickly.
- **Why one successful episode is insufficient evidence:** In `outputs/part_b_traces.txt`, episode `11020` (`is_slippery=True, fixed`) succeeds in 14 steps after bouncing around states `4`, `9`, `10`, `13`, and `14`, whereas episode `11000` under the exact same policy fails in just 2 steps (`0 --DOWN--> 1`, then `1 --RIGHT--> 5 [hole]`). Looking only at episode `11020` would make the policy look effective on slippery ice, when in reality it fails 96%–98% of the time. Repeated seeded batches are necessary to estimate true performance reliably.
- **Why these runs do not train a policy:** During all 600 episodes, neither policy updates any value estimates, Q-tables, transition models, or action mappings based on observed rewards or outcomes. Each episode simply executes the same static rule (`FIXED_POLICY` or uniform random sampling) from scratch, so this is pure **policy evaluation / simulation**, not learning or training.

## Part C — Custom solver

### Compare code with paper

Running `part_c_solver.py` on `paper_map` reproduces the exact Part A paper results with **zero discrepancies**.

**$V_1$ map (after 1 iteration):**

```text
+----------+----------+----------+----------+
|  -1.00   |  -1.00   |  10.00   | G (0.0)  |
+----------+----------+----------+----------+
|  -1.00   |    #     |  -1.00   | H (0.0)  |
+----------+----------+----------+----------+
|  -1.00   |  -1.00   |  -1.00   |  -1.00   |
+----------+----------+----------+----------+
```

**Requested $V_2$ values (1-indexed coordinates):**
- **`(1,2)` `[M]`:** `V2 = 8.000000` (`Q2`: `up=-3.9000`, `right=8.0000`, `down=-3.9000`, `left=-1.9000`) — matches paper (`8.0`).
- **`(1,3)` `[.]`:** `V2 = 10.000000` (`Q2`: `up=8.0000`, `right=10.0000`, `down=-1.9000`, `left=-3.9000`) — matches paper (`10.0`).
- **`(2,3)` `[I]`:** `V2 = 5.210000` (`Q2`: `up=5.2100`, `right=-7.3900`, `down=-2.7100`, `left=-0.9100`) — matches paper (`5.21`).

Because both the paper calculations and the solver perform synchronous Bellman backups from $V_0 = 0$ with $\gamma = 0.9$, reward-on-entry (including staying in place on blocked moves), and `0.8 / 0.1 / 0.1` slip probabilities on `I`, every $V_1$, $Q_2$, and $V_2$ value matched identically.

### Values, policies and convergence

**Saved value/policy maps and simulation logs:**
- Original map (`paper_map`) values and policy: [`outputs/part_c_paper_map.txt`](outputs/part_c_paper_map.txt)
- Extended map (`extended_map`) values and policy: [`outputs/part_c_extended_map.txt`](outputs/part_c_extended_map.txt)
- All 300 simulated episodes: [`outputs/part_c_episodes.csv`](outputs/part_c_episodes.csv)

**Convergence and tie-breaking:**
- **`paper_map`:** Converged in **12 iterations** (safeguard: `10,000`), final max $\vert{}\Delta V\vert{} = 0.000000217 < 10^{-6}$.
- **`extended_map`:** Converged in **11 iterations** (safeguard: `10,000`), final max $\vert{}\Delta V\vert{} = 0.000000241 < 10^{-6}$.
- **Tie-breaking rule:** If multiple actions have $Q$-values within `1e-12` of the maximum, the solver selects the first action in the deterministic order defined in `data/maps.json`: `up`, `right`, `down`, `left`. No ties occurred at convergence on either map.

**`paper_map` converged $V^*$ and $\pi^*$ (`V*(S) = 1.535824`):**

```text
+----------+----------+----------+----------+    +--------+--------+--------+--------+
|  4.2000  |  8.0000  | 10.0000  | G (0.0)  |    |  .:>   |  M:>   |  .:>   |   G    |
+----------+----------+----------+----------+    +--------+--------+--------+--------+
|  2.7800  |    #     |  5.8242  | H (0.0)  |    |  .:^   |   #    |  I:^   |   H    |
+----------+----------+----------+----------+    +--------+--------+--------+--------+
|  1.5358  |  2.8176  |  4.2418  |  2.8176  |    |  S:>   |  .:>   |  .:^   |  .:<   |
+----------+----------+----------+----------+    +--------+--------+--------+--------+
```

**`extended_map` converged $V^*$ and $\pi^*$ (`V*(S) = 1.502000`):**

```text
+----------+----------+----------+----------+    +--------+--------+--------+--------+
|  4.2000  |  8.0000  | 10.0000  | G (0.0)  |    |  .:>   |  M:>   |  .:>   |   G    |
+----------+----------+----------+----------+    +--------+--------+--------+--------+
|  2.7800  |    #     |  5.8242  | H (0.0)  |    |  .:^   |   #    |  I:^   |   H    |
+----------+----------+----------+----------+    +--------+--------+--------+--------+
|  1.5020  |  2.8176  |  4.2418  |  2.8176  |    |  S:^   |  D:>   |  .:^   |  W:<   |
+----------+----------+----------+----------+    +--------+--------+--------+--------+
```

### Simulation results

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |
| paper_map | 11 | 42 | 8 | 0 | 4.98 | 0.8913 |
| paper_map | 22 | 46 | 4 | 0 | 5.04 | 1.9148 |
| paper_map | 33 | 48 | 2 | 0 | 5.04 | 2.4921 |
| extended_map | 11 | 50 | 0 | 0 | 5.00 | 1.5020 |
| extended_map | 22 | 50 | 0 | 0 | 5.00 | 1.5020 |
| extended_map | 33 | 50 | 0 | 0 | 5.00 | 1.5020 |

### Explain your results

- **Which values and actions changed with sand (`D`) and shallow water (`W`):**
  - Only the start tile **`S` at `(3,1)`** changed: its optimal value dropped from **`1.5358`** (`paper_map`) to **`1.5020`** (`extended_map`), and its optimal action flipped from **`right` (`>`)** to **`up` (`^`)**.
  - On `paper_map`, the lower path through Ice `I` (`S -> (3,2) -> (3,3) -> I -> (1,3) -> G`) gives $Q^*(S, \text{right}) = -1 + 0.9 \times 2.817582 = 1.535824$, narrowly beating the deterministic upper path through Mud `M` (`S -> (2,1) -> (1,1) -> M -> (1,3) -> G`), which gives $Q^*(S, \text{up}) = -1 + 0.9(-1) + 0.9^2(-3) + 0.9^3(-1) + 0.9^4(10) = 1.502000$ by only `0.0338`.
  - On `extended_map`, placing Sand `D` (entry reward `-2`) at `(3,2)` reduces $Q^*(S, \text{right})$ by `1.0` to `0.535824`, making the deterministic upper route via Mud (`1.502000`) strictly better. Note that $V^*(3,2)$ itself remains `2.8176` because rewards depend on the **destination** tile entered, not the tile being exited: moving from `D` into `(3,3)` still costs `-1`.
  - Placing Shallow Water `W` (entry reward `-5`) at `(3,4)` changed **no values or actions**: from `(3,4)`, the optimal move is `left` into `(3,3)` (cost `-1`), and no optimal trajectory from any other state ever enters `(3,4)`.
- **How the policy handled Ice (`I`) and where cost vs. risk mattered:**
  - At `(2,3)` (`I`), the optimal action on both maps is **`up` (`^`)**, giving $V^*(2,3) = 5.8242$. Going `up` reaches `(1,3)` with probability `0.8`, bounces off wall `#` back onto `(2,3)` with probability `0.1`, and slips into Hole `H` with probability `0.1`. This yields a theoretical probability of reaching `(1,3)` (and then `G`) of $\frac{0.8}{0.8 + 0.1} = \frac{8}{9} \approx 88.89\%$ and falling into `H` of $\frac{1}{9} \approx 11.11\%$.
  - On `paper_map`, accepting this `11.11%` hole risk yields a slightly higher expected discounted return than paying the guaranteed `-3` cost of Mud `M`. Once Sand `D` adds an extra `-1` entry cost to the lower route in `extended_map`, the risk-adjusted edge disappears and the policy avoids `I` completely from `S`, achieving `100%` goal completion (`50/50` in every batch) instead of `136/150` goals (`90.7%`) and `14/150` holes (`9.3%`) on `paper_map`.
- **How finite-horizon simulations relate to infinite-horizon computed values:**
  - Value iteration computes the exact **infinite-horizon expected discounted return** $V^*(S)$ from the transition model. Finite simulations estimate this expectation through Monte Carlo sampling over a capped horizon (`100` steps) and finite batch size (`50` episodes).
  - On `extended_map`, the optimal path from `S` never visits `I`, so every episode is deterministic and finishes in `5` steps; consequently, every batch mean discounted return (`1.5020`) equals $V^*(S) = 1.502000$ exactly.
  - On `paper_map`, stochastic slipping on `I` creates sampling variance across 50-episode batches: batch means (`0.8913`, `1.9148`, `2.4921`) fluctuate around $V^*(S) = 1.535824$ depending on how many episodes slipped into `H` (`8`, `4`, and `2` holes respectively; 150-episode overall mean `1.7661`, mean steps `5.02` vs. theoretical `5.00`). No episode hit the 100-step limit because the probability of staying on `I` for $k$ consecutive steps decays as $0.1^k$.

### Rule checks

Full verification output is saved in [`outputs/part_c_rule_checks.txt`](outputs/part_c_rule_checks.txt). All checks passed before running the solver:

- **Boundary & wall behavior:** Moving `up` (map boundary) or `down` (wall `#`) from `(1,2)` (`M`) leaves the agent at `(1,2)` with probability `1.0` and applies the Mud destination reward `-3.0`.
- **Slip probabilities:** From `(2,3)` (`I`), transition probabilities sum to `1.000000` for all four actions (`up`, `right`, `down`, `left`). For `up`, the left perpendicular slip resolves against wall `#` into staying at `(2,3)` (`p=0.1`, reward `-1.0`), while the right slip enters Hole `(2,4)` (`p=0.1`, reward `-10.0`).
- **Floor rewards:** Entering `M` gives `-3.0`, entering `D` gives `-2.0`, entering `W` gives `-5.0`, entering `G` gives `+10.0` (with no extra floor cost), and entering `H` gives `-10.0`.
- **Terminal handling:** Taking actions after reaching `G` or `H` keeps the agent in the terminal state with reward `0.0` (no further rewards after termination).