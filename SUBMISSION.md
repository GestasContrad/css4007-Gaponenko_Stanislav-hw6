# HW6 submission

## AI-use declaration

Which tools did you use, and for what?

## Setup and reproduction

- **Python:** 3.14.0
- **gymnasium:** 1.4.0
- **numpy:** 2.5.5

```bash
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
pip install gymnasium==1.4.0
python part_b_frozenlake.py

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

Show V1 and the requested V2 values. Explain discrepancies.

### Values, policies and convergence

Link both value/policy maps. Record iteration counts, final value changes and tie-breaking rule.

### Simulation results

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |

### Explain your results

Which values/actions changed with sand and water? Where did cost and risk matter? How do simulations relate to the computed values?

### Rule checks

Show boundary/wall behavior, slip probabilities, floor rewards and terminal handling.

