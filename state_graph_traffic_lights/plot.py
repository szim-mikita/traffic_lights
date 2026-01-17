import numpy as np
import random
import matplotlib.pyplot as plt
import csv  # <-- ADDED

from gym_envs.envs.state_traffic_env import StateTrafficEnv
from train import DQNAgent, DQNConfig


# -------------------------
# CONFIG (edit here)
# -------------------------
MODEL_PATH = "traffic_dqn_works.pt"

# IMPORTANT: keep these SMALL to run fast
SIMULATION_TIME_SEC = 3600     # stop early (e.g. 120, 300, 600)
TRAFFIC_SCALE = 8
RENDER_MODE = "console"
SEED = 0

CSV_PATH = "policy_metrics.csv"  # <-- ADDED


class CyclicPolicy:
    def __init__(self):
        self.seq = [0, 0, 1, 1, 2, 2]
        self.i = 0

    def act(self, obs):
        a = self.seq[self.i % len(self.seq)]
        self.i += 1
        return a


class RandomPolicy:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def act(self, obs):
        return random.randrange(self.action_dim)


def cars_waiting_from_obs(obs):
    # Your env exposes detector vehicle numbers here
    return float(np.sum(np.asarray(obs["vehicle_count"], dtype=np.float32)))


# -------------------------
# ADDED: TraCI metrics helpers (avg speed + avg CO2)
# -------------------------
def avg_speed_and_co2_from_env(env):
    """
    Uses env.edges (TraCI edge IDs) to compute per-step average speed and CO2 emission.
    Returns: (avg_speed_mps, avg_co2_g_per_s)  [units depend on SUMO]
    """
    speeds = []
    co2s = []
    for edge in getattr(env, "edges", []):
        try:
            speeds.append(env.traci.edge.getLastStepMeanSpeed(edge))
        except Exception:
            pass
        try:
            co2s.append(env.traci.edge.getCO2Emission(edge))
        except Exception:
            pass

    avg_speed = float(np.mean(speeds)) if speeds else 0.0
    avg_co2 = float(np.mean(co2s)) if co2s else 0.0
    return avg_speed, avg_co2


def rollout_waiting_curve(env, policy):
    obs, _ = env.reset()
    xs_time = []
    ys_waiting = []

    # ADDED: per-step logs for CSV
    rewards = []
    avg_speeds = []
    avg_co2s = []

    # We step until env says done (based on current_step >= time_limit)
    done = False
    while not done:
        a = policy.act(obs)
        obs, reward, done, truncated, info = env.step(a)

        xs_time.append(env.current_step)  # "simulation seconds" counter inside env
        ys_waiting.append(cars_waiting_from_obs(obs))

        # ADDED
        rewards.append(float(reward))

        # NOTE: your env imported traci globally; easiest is to reference it via env module scope
        # But since you already have env.edges, we can call TraCI through the global traci module.
        # If your env doesn't expose traci, you can still import traci here.
        try:
            import traci
            speeds = []
            co2s = []
            for edge in getattr(env, "edges", []):
                try:
                    speeds.append(traci.edge.getLastStepMeanSpeed(edge))
                except Exception:
                    pass
                try:
                    co2s.append(traci.edge.getCO2Emission(edge))
                except Exception:
                    pass
            avg_speed = float(np.mean(speeds)) if speeds else 0.0
            avg_co2 = float(np.mean(co2s)) if co2s else 0.0
        except Exception:
            avg_speed, avg_co2 = 0.0, 0.0

        avg_speeds.append(avg_speed)
        avg_co2s.append(avg_co2)

        if truncated:
            break

    return (
        np.array(xs_time, dtype=np.int32),
        np.array(ys_waiting, dtype=np.float32),
        np.array(rewards, dtype=np.float32),
        np.array(avg_speeds, dtype=np.float32),
        np.array(avg_co2s, dtype=np.float32),
    )


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Create env (verbose=False to run fast)
    env = StateTrafficEnv(
        render_mode=RENDER_MODE,
        simulation_time=SIMULATION_TIME_SEC,
        traffic_scale=TRAFFIC_SCALE,
    )

    action_dim = int(env.action_space.n)

    # Load agent
    agent = DQNAgent(action_dim=action_dim, cfg=DQNConfig())
    agent.load(MODEL_PATH)
    print(f"Loaded model: {MODEL_PATH}")

    class DQNPolicy:
        def act(self, obs):
            return agent.act(obs, eval_mode=True)

    dqn_policy = DQNPolicy()
    cyc_policy = CyclicPolicy()
    rnd_policy = RandomPolicy(action_dim)

    t_dqn, w_dqn, r_dqn, s_dqn, co2_dqn = rollout_waiting_curve(env, dqn_policy)
    t_cyc, w_cyc, r_cyc, s_cyc, co2_cyc = rollout_waiting_curve(env, cyc_policy)
    t_rnd, w_rnd, r_rnd, s_rnd, co2_rnd = rollout_waiting_curve(env, rnd_policy)

    env.close()

    # -------------------------
    # ADDED: Write CSV
    # -------------------------
    def write_policy_rows(writer, name, t, w, r, spd, co2):
        n = min(len(t), len(w), len(r), len(spd), len(co2))
        for i in range(n):
            writer.writerow([name, int(t[i]), float(w[i]), float(r[i]), float(spd[i]), float(co2[i])])

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "time_s", "waiting_cars", "reward", "avg_speed", "avg_co2"])
        write_policy_rows(writer, "DQN", t_dqn, w_dqn, r_dqn, s_dqn, co2_dqn)
        write_policy_rows(writer, "Cyclic", t_cyc, w_cyc, r_cyc, s_cyc, co2_cyc)
        write_policy_rows(writer, "Random", t_rnd, w_rnd, r_rnd, s_rnd, co2_rnd)

    print(f"Saved CSV: {CSV_PATH}")

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(t_dqn, w_dqn, label="DQN")
    plt.plot(t_cyc, w_cyc, label="Cyclic 0,0,1,1,2,2", linestyle="--")
    plt.plot(t_rnd, w_rnd, label="Random", linestyle=":")
    plt.xlabel("Simulation time (seconds)")
    plt.ylabel("Cars waiting (sum of detector vehicle_count)")
    plt.title(f"Cars waiting over time (scale={TRAFFIC_SCALE}, sim_time={SIMULATION_TIME_SEC}s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

