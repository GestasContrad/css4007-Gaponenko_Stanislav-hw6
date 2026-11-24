import csv
import random
import re
from pathlib import Path

import gymnasium as gym

ACTION_NAMES = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
GOAL_STATE = 15
HOLE_STATES = {5, 7, 11, 12}

# Fixed policy designed from the 4x4 map before running experiments
FIXED_POLICY = {
    0: 1,   # DOWN
    1: 2,   # RIGHT
    2: 1,   # DOWN
    3: 0,   # LEFT
    4: 1,   # DOWN
    6: 1,   # DOWN
    8: 2,   # RIGHT
    9: 1,   # DOWN
    10: 1,  # DOWN
    13: 2,  # RIGHT
    14: 2,  # RIGHT
}

BATCH_SEEDS = [11, 22, 33]
EPISODES_PER_BATCH = 50
MAX_STEPS = 100
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def choose_action(state: int, policy_name: str, rng: random.Random) -> int:
    if policy_name == "random":
        return rng.randrange(4)
    if policy_name == "fixed":
        return FIXED_POLICY[state]
    raise ValueError(f"Unknown policy: {policy_name}")


def classify_outcome(final_state: int, terminated: bool) -> str:
    # Prioritize terminal outcomes even if termination occurs on the last allowed step
    if terminated or final_state == GOAL_STATE or final_state in HOLE_STATES:
        if final_state == GOAL_STATE:
            return "goal"
        if final_state in HOLE_STATES:
            return "hole"
    return "time_limit"


def run_episode(
    env: gym.Env,
    policy_name: str,
    episode_seed: int,
    max_steps: int = MAX_STEPS,
    record_trace: bool = False,
) -> dict:
    obs, _ = env.reset(seed=episode_seed)
    rng = random.Random(episode_seed)

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    trace_lines = []

    if record_trace:
        rendered = strip_ansi(env.render())
        trace_lines.append(f"Initial state: {obs}\n{rendered}")

    while not (terminated or truncated) and steps < max_steps:
        action = choose_action(int(obs), policy_name, rng)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        total_reward += float(reward)

        if record_trace:
            rendered = strip_ansi(env.render())
            trace_lines.append(
                f"Step {steps}: state {obs} --{ACTION_NAMES[action]}--> "
                f"state {next_obs}, reward={reward}, "
                f"terminated={terminated}, truncated={truncated}\n{rendered}"
            )

        obs = next_obs

    outcome = classify_outcome(int(obs), terminated)
    return {
        "steps": steps,
        "total_reward": total_reward,
        "outcome": outcome,
        "final_state": int(obs),
        "trace": "\n".join(trace_lines),
    }


def main() -> None:
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    conditions = [
        (False, "random"),
        (False, "fixed"),
        (True, "random"),
        (True, "fixed"),
    ]

    all_rows = []
    summary_rows = []

    # Track example episodes to re-run with text rendering for saved traces
    rendered_sample_meta = None
    success_trace_meta = None
    failed_trace_meta = None

    for is_slippery, policy_name in conditions:
        condition_label = f"slippery_{str(is_slippery).lower()}_{policy_name}"
        env = gym.make(
            "FrozenLake-v1",
            map_name="4x4",
            is_slippery=is_slippery,
            max_episode_steps=MAX_STEPS,
        )

        try:
            for batch_seed in BATCH_SEEDS:
                goals = 0
                holes = 0
                time_limits = 0
                steps_sum = 0

                for i in range(EPISODES_PER_BATCH):
                    episode_seed = 1000 * batch_seed + i
                    result = run_episode(env, policy_name, episode_seed, MAX_STEPS)

                    outcome = result["outcome"]
                    if outcome == "goal":
                        goals += 1
                    elif outcome == "hole":
                        holes += 1
                    else:
                        time_limits += 1
                    steps_sum += result["steps"]

                    all_rows.append(
                        {
                            "condition": condition_label,
                            "batch_seed": batch_seed,
                            "episode_seed": episode_seed,
                            "policy": policy_name,
                            "steps": result["steps"],
                            "total_reward": result["total_reward"],
                            "outcome": outcome,
                        }
                    )

                    # Pick clean representative traces from slippery fixed policy
                    if is_slippery and policy_name == "fixed":
                        if rendered_sample_meta is None:
                            rendered_sample_meta = (is_slippery, policy_name, batch_seed, episode_seed)
                        if outcome == "goal" and success_trace_meta is None:
                            success_trace_meta = (is_slippery, policy_name, batch_seed, episode_seed)
                        if outcome != "goal" and failed_trace_meta is None:
                            failed_trace_meta = (is_slippery, policy_name, batch_seed, episode_seed)

                success_pct = (goals / EPISODES_PER_BATCH) * 100.0
                mean_steps = steps_sum / EPISODES_PER_BATCH
                summary_rows.append(
                    {
                        "slippery": str(is_slippery),
                        "policy": policy_name,
                        "batch_seed": batch_seed,
                        "goals": goals,
                        "holes": holes,
                        "time_limits": time_limits,
                        "success_pct": f"{success_pct:.1f}%",
                        "mean_steps": f"{mean_steps:.2f}",
                    }
                )
        finally:
            env.close()

    # 1 Save per-episode CSV
    csv_path = outputs_dir / "part_b_episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "batch_seed",
                "episode_seed",
                "policy",
                "steps",
                "total_reward",
                "outcome",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)


    def capture_rendered(meta: tuple) -> str:
        is_slip, pol, b_seed, ep_seed = meta
        render_env = gym.make(
            "FrozenLake-v1",
            map_name="4x4",
            is_slippery=is_slip,
            max_episode_steps=MAX_STEPS,
            render_mode="ansi",
        )
        try:
            res = run_episode(render_env, pol, ep_seed, MAX_STEPS, record_trace=True)
        finally:
            render_env.close()
        header = (
            f"Condition: is_slippery={is_slip}, policy={pol}, "
            f"batch_seed={b_seed}, episode_seed={ep_seed}\n"
            f"Outcome: {res['outcome']}, steps={res['steps']}, total_reward={res['total_reward']}\n"
            + "=" * 60
            + "\n"
        )
        return header + res["trace"]

    # 2 Save one rendered episode (Req 1)
    rendered_path = outputs_dir / "part_b_rendered_episode.txt"
    sample_meta = success_trace_meta or rendered_sample_meta
    rendered_path.write_text(capture_rendered(sample_meta), encoding="utf-8")

    # 3 Save one successful and one failed trace (Sum req)
    traces_parts = []
    if success_trace_meta is not None:
        traces_parts.append("### Successful Trace\n\n" + capture_rendered(success_trace_meta))
    else:
        traces_parts.append("### Successful Trace\n\nNo successful episode observed in slippery fixed-policy runs.")

    if failed_trace_meta is not None:
        traces_parts.append("### Failed Trace\n\n" + capture_rendered(failed_trace_meta))
    else:
        traces_parts.append("### Failed Trace\n\nNo failed episode observed in slippery fixed-policy runs.")

    traces_path = outputs_dir / "part_b_traces.txt"
    traces_path.write_text("\n\n".join(traces_parts) + "\n", encoding="utf-8")

    # 4 Print Markdown sum table
    print("\n| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in summary_rows:
        print(
            f"| {row['slippery']} | {row['policy']} | {row['batch_seed']} | "
            f"{row['goals']} | {row['holes']} | {row['time_limits']} | "
            f"{row['success_pct']} | {row['mean_steps']} |"
        )

    print(f"\nSaved {len(all_rows)} episode rows to {csv_path}")
    print(f"Saved rendered episode to {rendered_path}")
    print(f"Saved example traces to {traces_path}")


if __name__ == "__main__":
    main()