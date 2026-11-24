# HW6 submission

## AI-use declaration

Which tools did you use, and for what?

## Setup and reproduction

Record dependency versions, installation commands and exact run commands.

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

Link episode rows and example traces in outputs/. Explain slipping, variation across batches and why these runs do not train a policy.

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

