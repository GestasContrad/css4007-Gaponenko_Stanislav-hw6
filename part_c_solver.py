import csv
import json
import random
from pathlib import Path

ACTION_DELTAS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

PERPENDICULAR_ACTIONS = {
    "up": ("left", "right"),
    "down": ("left", "right"),
    "left": ("up", "down"),
    "right": ("up", "down"),
}

ACTION_ARROWS = {
    "up": "^",
    "right": ">",
    "down": "v",
    "left": "<",
}

TIE_EPS = 1e-12


class GridWorldMDP:
    """Data-driven Grid World MDP whose floor rules come strictly from config."""

    def __init__(self, name: str, grid_rows: list[str], config: dict) -> None:
        self.name = name
        self.grid_rows = grid_rows
        self.rows = len(grid_rows)
        self.cols = len(grid_rows[0])
        self.tiles = config["tiles"]
        self.actions = list(config["actions"])
        self.discount = float(config["discount"])
        self.initial_value = float(config["initial_value"])
        self.terminal_value = float(config["terminal_value"])

        self.states = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.is_traversable(r, c)
        ]
        self.start_state = self._find_tile("S")

    def _find_tile(self, symbol: str) -> tuple[int, int]:
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid_rows[r][c] == symbol:
                    return (r, c)
        raise ValueError(f"Tile '{symbol}' not found in map '{self.name}'.")

    def tile_symbol(self, state: tuple[int, int]) -> str:
        r, c = state
        return self.grid_rows[r][c]

    def tile_rule(self, state: tuple[int, int]) -> dict:
        return self.tiles[self.tile_symbol(state)]

    def is_traversable(self, r: int, c: int) -> bool:
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        return self.tiles[self.grid_rows[r][c]].get("traversable", True)

    def is_terminal(self, state: tuple[int, int]) -> bool:
        return bool(self.tile_rule(state).get("terminal", False))

    def move_once(self, state: tuple[int, int], direction: str) -> tuple[int, int]:
        r, c = state
        dr, dc = ACTION_DELTAS[direction]
        nr, nc = r + dr, c + dc
        if not self.is_traversable(nr, nc):
            return state
        return (nr, nc)

    def get_transitions(self, state: tuple[int, int], action: str) -> list[tuple[tuple[int, int], float, float]]:
        """Return list of (next_state, prob, reward) for taking action from state."""
        if self.is_terminal(state):
            return [(state, 1.0, 0.0)]

        rule = self.tile_rule(state)
        if rule.get("movement") == "slippery":
            p_intended = float(rule["intended"])
            p_perp = float(rule["perpendicular_each"])
            p1, p2 = PERPENDICULAR_ACTIONS[action]
            raw_moves = [(action, p_intended), (p1, p_perp), (p2, p_perp)]
        else:
            raw_moves = [(action, 1.0)]

        prob_by_next: dict[tuple[int, int], float] = {}
        for direction, prob in raw_moves:
            nxt = self.move_once(state, direction)
            prob_by_next[nxt] = prob_by_next.get(nxt, 0.0) + prob

        return [
            (nxt, prob, float(self.tile_rule(nxt)["reward"]))
            for nxt, prob in prob_by_next.items()
        ]

    def step(self, state: tuple[int, int], action: str, rng: random.Random) -> tuple[tuple[int, int], float, bool]:
        """Sample a single transition from the model."""
        if self.is_terminal(state):
            return state, 0.0, True

        transitions = self.get_transitions(state, action)
        u = rng.random()
        cumulative = 0.0
        chosen_next, chosen_reward = transitions[-1][0], transitions[-1][2]
        for nxt, prob, rew in transitions:
            cumulative += prob
            if u < cumulative:
                chosen_next, chosen_reward = nxt, rew
                break

        return chosen_next, chosen_reward, self.is_terminal(chosen_next)


def q_value(mdp: GridWorldMDP, values: dict[tuple[int, int], float], state: tuple[int, int], action: str) -> float:
    return sum(
        prob * (reward + mdp.discount * values[nxt])
        for nxt, prob, reward in mdp.get_transitions(state, action)
    )


def value_iteration(
    mdp: GridWorldMDP,
    tolerance: float,
    max_iterations: int,
) -> dict:
    values = {s: mdp.initial_value for s in mdp.states}
    for s in mdp.states:
        if mdp.is_terminal(s):
            values[s] = mdp.terminal_value

    v1_snapshot = None
    v2_snapshot = None
    v2_q_details = {}
    iterations = 0
    final_delta = 0.0
    converged = False

    for k in range(1, max_iterations + 1):
        new_values: dict[tuple[int, int], float] = {}
        max_delta = 0.0

        for s in mdp.states:
            if mdp.is_terminal(s):
                new_values[s] = mdp.terminal_value
                continue

            q_by_action = {a: q_value(mdp, values, s, a) for a in mdp.actions}
            best_v = max(q_by_action.values())
            new_values[s] = best_v
            max_delta = max(max_delta, abs(best_v - values[s]))

            if k == 2:
                v2_q_details[s] = q_by_action

        values = new_values
        iterations = k
        final_delta = max_delta

        if k == 1:
            v1_snapshot = dict(values)
        elif k == 2:
            v2_snapshot = dict(values)

        if max_delta < tolerance:
            converged = True
            break

    # Extract greedy policy using final converged values
    # Tie-breaking rule: deterministic action order from maps.json ("up", "right", "down", "left")
    policy: dict[tuple[int, int], str] = {}
    ties: dict[tuple[int, int], list[str]] = {}
    final_q: dict[tuple[int, int], dict[str, float]] = {}

    for s in mdp.states:
        if mdp.is_terminal(s):
            continue
        q_by_action = {a: q_value(mdp, values, s, a) for a in mdp.actions}
        final_q[s] = q_by_action
        best_q = max(q_by_action.values())
        tied_actions = [a for a in mdp.actions if abs(q_by_action[a] - best_q) <= TIE_EPS]
        policy[s] = tied_actions[0]
        if len(tied_actions) > 1:
            ties[s] = tied_actions

    return {
        "values": values,
        "policy": policy,
        "ties": ties,
        "final_q": final_q,
        "v1": v1_snapshot,
        "v2": v2_snapshot,
        "v2_q_details": v2_q_details,
        "iterations": iterations,
        "final_delta": final_delta,
        "converged": converged,
    }


def format_value_grid(mdp: GridWorldMDP, values: dict[tuple[int, int], float], decimals: int = 4) -> str:
    col_w = 10
    sep = "+" + "+".join(["-" * col_w] * mdp.cols) + "+"
    lines = [sep]
    for r in range(mdp.rows):
        cells = []
        for c in range(mdp.cols):
            sym = mdp.grid_rows[r][c]
            if not mdp.is_traversable(r, c):
                text = "#"
            elif mdp.is_terminal((r, c)):
                text = f"{sym} ({values[(r, c)]:.1f})"
            else:
                text = f"{values[(r, c)]:.{decimals}f}"
            cells.append(text.center(col_w))
        lines.append("|" + "|".join(cells) + "|")
        lines.append(sep)
    return "\n".join(lines)


def format_policy_grid(mdp: GridWorldMDP, policy: dict[tuple[int, int], str]) -> str:
    col_w = 8
    sep = "+" + "+".join(["-" * col_w] * mdp.cols) + "+"
    lines = [sep]
    for r in range(mdp.rows):
        cells = []
        for c in range(mdp.cols):
            sym = mdp.grid_rows[r][c]
            if not mdp.is_traversable(r, c):
                text = "#"
            elif mdp.is_terminal((r, c)):
                text = sym
            else:
                arrow = ACTION_ARROWS[policy[(r, c)]]
                text = f"{sym}:{arrow}"
            cells.append(text.center(col_w))
        lines.append("|" + "|".join(cells) + "|")
        lines.append(sep)
    return "\n".join(lines)


def simulate_policy(
    mdp: GridWorldMDP,
    policy: dict[tuple[int, int], str],
    batch_seeds: list[int],
    episodes_per_batch: int,
    max_steps: int,
) -> tuple[list[dict], list[dict]]:
    episode_rows = []
    batch_summaries = []

    for batch_seed in batch_seeds:
        goals = 0
        holes = 0
        time_limits = 0
        steps_sum = 0
        disc_return_sum = 0.0

        for i in range(episodes_per_batch):
            episode_seed = 1000 * batch_seed + i
            rng = random.Random(episode_seed)

            state = mdp.start_state
            steps = 0
            total_reward = 0.0
            discounted_return = 0.0
            discount_pow = 1.0
            terminated = False

            while not terminated and steps < max_steps:
                action = policy[state]
                next_state, reward, terminated = mdp.step(state, action, rng)
                total_reward += reward
                discounted_return += discount_pow * reward
                discount_pow *= mdp.discount
                steps += 1
                state = next_state

            final_sym = mdp.tile_symbol(state)
            if terminated and final_sym == "G":
                outcome = "goal"
                goals += 1
            elif terminated and final_sym == "H":
                outcome = "hole"
                holes += 1
            else:
                outcome = "time_limit"
                time_limits += 1

            steps_sum += steps
            disc_return_sum += discounted_return

            episode_rows.append(
                {
                    "map": mdp.name,
                    "batch_seed": batch_seed,
                    "episode_seed": episode_seed,
                    "steps": steps,
                    "total_reward": round(total_reward, 6),
                    "discounted_return": round(discounted_return, 6),
                    "outcome": outcome,
                }
            )

        batch_summaries.append(
            {
                "map": mdp.name,
                "batch_seed": batch_seed,
                "goals": goals,
                "holes": holes,
                "time_limits": time_limits,
                "mean_steps": steps_sum / episodes_per_batch,
                "mean_discounted_return": disc_return_sum / episodes_per_batch,
            }
        )

    return episode_rows, batch_summaries


def run_rule_checks(paper_mdp: GridWorldMDP, ext_mdp: GridWorldMDP) -> str:
    lines = ["=== Part C Rule Verification Checks ==="]

    # 1 Wall and boundary behavior (stay in place + receive current tile reward)
    b_trans = paper_mdp.get_transitions((0, 1), "up")     # (1,2) Mud -> up (boundary)
    w_trans = paper_mdp.get_transitions((0, 1), "down")   # (1,2) Mud -> down (wall #)
    assert b_trans == [((0, 1), 1.0, -3.0)], f"Boundary check failed: {b_trans}"
    assert w_trans == [((0, 1), 1.0, -3.0)], f"Wall check failed: {w_trans}"
    lines.append(
        f"[PASS] Boundary check at (1,2) Mud going 'up': {b_trans} "
        "(stays at (1,2), receives Mud reward -3.0)"
    )
    lines.append(
        f"[PASS] Wall check at (1,2) Mud going 'down' into '#': {w_trans} "
        "(stays at (1,2), receives Mud reward -3.0)"
    )

    # 2 Ice probabilities sum to 1 and resolve walls separately per slip direction
    ice_state = (1, 2)  # 1-indexed (2,3)
    for action in paper_mdp.actions:
        trans = paper_mdp.get_transitions(ice_state, action)
        prob_sum = sum(p for _, p, _ in trans)
        assert abs(prob_sum - 1.0) < 1e-12, f"Ice prob sum != 1 for {action}: {prob_sum}"
        lines.append(
            f"[PASS] Ice (2,3) action '{action}': transitions={trans}, sum(P)={prob_sum:.6f}"
        )

    # Specifically verify ice 'up' hitting wall on left slip (stays at (2,3)) and hole on right slip
    ice_up = dict((nxt, (p, r)) for nxt, p, r in paper_mdp.get_transitions(ice_state, "up"))
    assert ice_up == {
        (0, 2): (0.8, -1.0),
        (1, 2): (0.1, -1.0),
        (1, 3): (0.1, -10.0),
    }, f"Unexpected ice 'up' transitions: {ice_up}"
    lines.append(
        "[PASS] Ice (2,3) 'up' resolves left-slip against wall '#' into stay-in-place (0.1, reward -1.0) "
        "and right-slip into Hole (0.1, reward -10.0)."
    )

    # 3 Specified mud, sand, and shallow water rewards
    mud_enter = ext_mdp.get_transitions((0, 0), "right")   # (1,1) -> (1,2) M
    sand_enter = ext_mdp.get_transitions((2, 0), "right")  # (3,1) -> (3,2) D
    water_enter = ext_mdp.get_transitions((2, 2), "right") # (3,3) -> (3,4) W
    assert mud_enter == [((0, 1), 1.0, -3.0)]
    assert sand_enter == [((2, 1), 1.0, -2.0)]
    assert water_enter == [((2, 3), 1.0, -5.0)]
    lines.append(f"[PASS] Entering Mud 'M' (1,1)->(1,2): reward={mud_enter[0][2]}")
    lines.append(f"[PASS] Entering Sand 'D' (3,1)->(3,2): reward={sand_enter[0][2]}")
    lines.append(f"[PASS] Entering Shallow Water 'W' (3,3)->(3,4): reward={water_enter[0][2]}")

    # 4 Terminal handling: entering G gives +10 (no floor cost), entering H gives -10, and no rewards after termination
    goal_enter = paper_mdp.get_transitions((0, 2), "right")  # (1,3) -> (1,4) G
    hole_enter = paper_mdp.get_transitions((2, 3), "up")     # (3,4) -> (2,4) H
    after_goal = paper_mdp.get_transitions((0, 3), "left")
    after_hole = paper_mdp.get_transitions((1, 3), "left")
    assert goal_enter == [((0, 3), 1.0, 10.0)]
    assert hole_enter == [((1, 3), 1.0, -10.0)]
    assert after_goal == [((0, 3), 1.0, 0.0)]
    assert after_hole == [((1, 3), 1.0, 0.0)]
    lines.append(f"[PASS] Entering Goal 'G' (1,3)->(1,4): reward={goal_enter[0][2]} (no extra floor cost)")
    lines.append(f"[PASS] Entering Hole 'H' (3,4)->(2,4): reward={hole_enter[0][2]}")
    lines.append(f"[PASS] Post-termination at Goal 'G': transitions={after_goal} (no further reward)")
    lines.append(f"[PASS] Post-termination at Hole 'H': transitions={after_hole} (no further reward)")

    return "\n".join(lines) + "\n"


def build_map_report(mdp: GridWorldMDP, res: dict, paper_v2_positions: list[list[int]] | None = None) -> str:
    lines = [
        f"=== Map: {mdp.name} ===",
        f"Converged: {res['converged']} | Iterations: {res['iterations']} | Final max |delta|: {res['final_delta']:.9f}",
        "Tie-breaking rule: deterministic action order from data/maps.json (up, right, down, left); tolerance=1e-12.",
        f"Detected ties at convergence: {res['ties'] if res['ties'] else 'None'}",
        "",
    ]

    if paper_v2_positions is not None:
        lines.append("V1 Map (after 1 iteration):")
        lines.append(format_value_grid(mdp, res["v1"], decimals=2))
        lines.append("")
        lines.append("Requested V2 values (1-indexed coordinates):")
        for r1, c1 in paper_v2_positions:
            s = (r1 - 1, c1 - 1)
            q_str = ", ".join(f"{a}={val:.4f}" for a, val in res["v2_q_details"][s].items())
            lines.append(f"  ({r1},{c1}) [{mdp.tile_symbol(s)}]: V2 = {res['v2'][s]:.6f}  | Q2: {q_str}")
        lines.append("")

    lines.append("Converged Value Map (V*):")
    lines.append(format_value_grid(mdp, res["values"], decimals=4))
    lines.append("")
    lines.append("Extracted Optimal Policy Map (pi*):")
    lines.append(format_policy_grid(mdp, res["policy"]))
    lines.append("")
    lines.append(f"Start state (3,1) optimal value V*(S) = {res['values'][mdp.start_state]:.6f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    config = json.loads(Path("data/maps.json").read_text(encoding="utf-8"))
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    tolerance = float(config["solver"]["tolerance"])
    max_iterations = int(config["solver"]["max_iterations"])
    sim_cfg = config["simulation"]

    paper_mdp = GridWorldMDP("paper_map", config["paper_map"], config)
    ext_mdp = GridWorldMDP("extended_map", config["extended_map"], config)

    # 1 Rule checks before trusting solver
    checks_text = run_rule_checks(paper_mdp, ext_mdp)
    checks_path = outputs_dir / "part_c_rule_checks.txt"
    checks_path.write_text(checks_text, encoding="utf-8")
    print(checks_text)

    # 2 Solve both maps with Value Iteration
    paper_res = value_iteration(paper_mdp, tolerance, max_iterations)
    ext_res = value_iteration(ext_mdp, tolerance, max_iterations)

    paper_report = build_map_report(paper_mdp, paper_res, config["paper_v2_positions"])
    ext_report = build_map_report(ext_mdp, ext_res)

    paper_map_path = outputs_dir / "part_c_paper_map.txt"
    ext_map_path = outputs_dir / "part_c_extended_map.txt"
    paper_map_path.write_text(paper_report, encoding="utf-8")
    ext_map_path.write_text(ext_report, encoding="utf-8")

    print(paper_report)
    print(ext_report)

    # 3 Simulate optimal policies on both maps
    all_episode_rows = []
    all_summaries = []

    for mdp, res in [(paper_mdp, paper_res), (ext_mdp, ext_res)]:
        ep_rows, summaries = simulate_policy(
            mdp,
            res["policy"],
            sim_cfg["batch_seeds"],
            int(sim_cfg["episodes_per_batch"]),
            int(sim_cfg["max_steps"]),
        )
        all_episode_rows.extend(ep_rows)
        all_summaries.extend(summaries)

    csv_path = outputs_dir / "part_c_episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "map",
                "batch_seed",
                "episode_seed",
                "steps",
                "total_reward",
                "discounted_return",
                "outcome",
            ],
        )
        writer.writeheader()
        writer.writerows(all_episode_rows)

    print("=== Simulation Results ===")
    print("| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for row in all_summaries:
        print(
            f"| {row['map']} | {row['batch_seed']} | {row['goals']} | {row['holes']} | "
            f"{row['time_limits']} | {row['mean_steps']:.2f} | {row['mean_discounted_return']:.4f} |"
        )

    print(f"\nSaved rule checks to {checks_path}")
    print(f"Saved paper map results to {paper_map_path}")
    print(f"Saved extended map results to {ext_map_path}")
    print(f"Saved {len(all_episode_rows)} episode rows to {csv_path}")


if __name__ == "__main__":
    main()