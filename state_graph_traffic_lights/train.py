import argparse
import math
import random
from dataclasses import dataclass
from collections import namedtuple, deque
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gym_envs.envs.state_traffic_env import StateTrafficEnv


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


@dataclass
class DQNConfig:
    hidden_dim: int = 64

    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 3e-4
    tau: float = 0.005
    memory_capacity: int = 50_000

    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: float = 2000

    grad_clip_value: float = 100.0

    # Scaling for your dict obs -> feature vector (tune if needed)
    occ_scale: float = 100.0
    veh_scale: float = 7200.0
    time_scale: float = 3600.0
    state_var_scale: float = 1.0


class StateDictFeaturizer:
    """
    Converts your env Dict observation to a flat vector.
    Expects keys (matching your first code):
      - occupancy: (4,)
      - vehicle_count: (4,)
      - current_state: int
      - time_in_state: (1,) or scalar
    Output dim = 10
    """
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.input_dim = 11

    def __call__(self, obs: Dict[str, Any]) -> np.ndarray:
        occ = np.asarray(obs["occupancy"], dtype=np.float32).reshape(-1)        # (4,)
        veh = np.asarray(obs["vehicle_count"], dtype=np.float32).reshape(-1)    # (4,)
        cs = float(obs["current_state"])
        tis = obs["time_in_state"]
        tis = float(tis[0]) if isinstance(tis, (np.ndarray, list, tuple)) else float(tis)
        stv = float(obs["all_states_used"])  # boolean to float

        # normalize
        occ_n = occ / self.cfg.occ_scale
        veh_n = veh / self.cfg.veh_scale
        tis_n = tis / self.cfg.time_scale

        # current_state normalization
        # if 1..3 => map to 0..1
        # if 0..2 => map to 0..1
        if cs >= 1.0:
            cs_n = (cs - 1.0) / 2.0
        else:
            cs_n = cs / 2.0

        feat = np.concatenate([occ_n, veh_n, np.array([cs_n, tis_n, stv], dtype=np.float32)], axis=0)
        return feat.astype(np.float32)


class DQNAgent:
    def __init__(
        self,
        action_dim: int,
        cfg: Optional[DQNConfig] = None,
        device: Optional[torch.device] = None,
        featurizer: Optional[StateDictFeaturizer] = None,
    ):
        self.cfg = cfg or DQNConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.featurizer = featurizer or StateDictFeaturizer(self.cfg)
        self.input_dim = self.featurizer.input_dim
        self.action_dim = action_dim

        self.policy_net = DQN(self.input_dim, self.cfg.hidden_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.input_dim, self.cfg.hidden_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.lr)
        self.memory = ReplayMemory(self.cfg.memory_capacity)
        self.steps_done = 0

    def _obs_to_torch(self, obs_dict: Dict[str, Any]) -> torch.Tensor:
        feat = self.featurizer(obs_dict)
        return torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1,D]

    def act(self, obs_dict: Dict[str, Any], eval_mode: bool = False) -> int:
        state = self._obs_to_torch(obs_dict)

        if eval_mode:
            print(f"        [ACT] action the model would select (eval mode): {int(self.policy_net(state).argmax(dim=1).item())}")
            with torch.no_grad():
                return int(self.policy_net(state).argmax(dim=1).item())

        sample = random.random()
        eps = self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * math.exp(
            -1.0 * self.steps_done / self.cfg.eps_decay
        )
        self.steps_done += 1

        print(f"        [ACT] action the model would select: {int(self.policy_net(state).argmax(dim=1).item())} eps={eps:.3f}")
        if sample > eps:
            with torch.no_grad():
                return int(self.policy_net(state).argmax(dim=1).item())
        print("        [ACT] random action selected")
        return random.randrange(self.action_dim)

    def remember(self, obs, action: int, next_obs, reward: float, done: bool):
        s = self._obs_to_torch(obs)  # [1,D]
        a = torch.tensor([[action]], device=self.device, dtype=torch.long)  # [1,1]
        r = torch.tensor([reward], device=self.device, dtype=torch.float32) # [1]
        d = torch.tensor([done], device=self.device, dtype=torch.bool)      # [1]
        ns = torch.zeros_like(s) if done else self._obs_to_torch(next_obs)

        self.memory.push(s, a, ns, r, d)

    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.cfg.batch_size:
            return None

        transitions = self.memory.sample(self.cfg.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch  = torch.cat(batch.state)   # [B,D]
        action_batch = torch.cat(batch.action)  # [B,1]
        reward_batch = torch.cat(batch.reward)  # [B]
        done_batch   = torch.cat(batch.done)    # [B] bool

        q_sa = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)  # [B]

        non_final_mask = ~done_batch
        next_state_values = torch.zeros(self.cfg.batch_size, device=self.device)

        if non_final_mask.any():
            non_final_next_states = torch.cat(
                [s for s, d in zip(batch.next_state, batch.done) if not d.item()]
            )
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        expected = reward_batch + self.cfg.gamma * next_state_values * non_final_mask.float()
        loss = F.smooth_l1_loss(q_sa, expected)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.cfg.grad_clip_value)
        self.optimizer.step()

        # soft update
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.data.mul_(1.0 - self.cfg.tau)
                tp.data.add_(self.cfg.tau * pp.data)

        return float(loss.item())

    def save(self, path: str):
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.policy_net.eval()
        self.target_net.eval()


def run_eval(env: StateTrafficEnv, agent: DQNAgent, episodes: int, max_steps: int):
    for ep in range(episodes):
        obs, info = env.reset()
        ep_return = 0.0

        for t in range(max_steps):
            action = agent.act(obs, eval_mode=True)

            # OUTPUT that other people want:
            print(f"    [EVAL] episode={ep} step={t} action={action}")

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            ep_return += float(reward)
            obs = next_obs
            if done:
                break

        print(f"[EVAL] episode={ep} steps={t+1} return={ep_return:.3f}")


def run_train(env: StateTrafficEnv, agent: DQNAgent, episodes: int, max_steps: int, save_path: str):
    for ep in range(episodes):
        obs, info = env.reset()
        ep_return = 0.0

        for t in range(max_steps):
            action = agent.act(obs, eval_mode=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            agent.remember(obs, action, next_obs, float(reward), done)
            loss = agent.train_step()

            ep_return += float(reward)
            obs = next_obs
            print(f"    [TRAIN] episode={ep} step={t} action={action} reward={reward:.3f} loss={loss}")
            if done:
                print("Simulation finished after {} timesteps".format(t+1))
                break

        print(f"[TRAIN] episode={ep} steps={t+1} return={ep_return:.3f}")

    agent.save(save_path)
    print(f"Saved model to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eval", "train"], default="train")
    parser.add_argument("--model", type=str, default="traffic_dqn.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--render_mode", type=str, default="console")
    args = parser.parse_args()

    # Create env by directly calling your class (no gym registration needed)
    env = gym.make(
        "StateTrafficEnv",
        render_mode=args.render_mode,
    )

    # action_dim must match your env: your env.action_space is Discrete(len(self.actions))
    action_dim = int(env.action_space.n)

    agent = DQNAgent(action_dim=action_dim)

    if args.mode == "train":
        run_train(env, agent, args.episodes, args.max_steps, args.model)
    else:
        # load weights if they exist
        try:
            agent.load(args.model)
            print(f"Loaded model: {args.model}")
        except Exception as e:
            print(f"Could not load model '{args.model}'. Running with random/untrained policy. Error: {e}")

        run_eval(env, agent, args.episodes, args.max_steps)

    env.close()


if __name__ == "__main__":
    main()
