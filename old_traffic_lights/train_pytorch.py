import math
import random
from collections import namedtuple, deque
from itertools import count
from multiprocessing import Process, cpu_count

from traci.connection import check

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from accelerate import Accelerator
from tensordict import TensorDict
import onnx
import traci

import gym_envs.envs.traffic_light_support_functions as tlsf
import wandb

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

eps_threshold = 0
steps_done = 0

hyperparameter_defaults = dict(
    num_episodes=100,
    TAU=0.014622523699104623,
    learning_rate=0.022850574714388262,
    batch_size=128,
    gamma=0.05420679682778693,
    eps_start=0.677398614743279,
    eps_decay=100,
    fc_layer_size_1=64,
    fc_layer_size_2=256,
    fc_phaseplan_layer_size=128,
    use_phaseplan=0
)
wandb.init(config=hyperparameter_defaults, project="traffic-light-dqn")
config = wandb.config

def train(config = None):
    starting_phases = [1, 1, 2, 2, 1, 2]
    phase_lengths = [
        [19, 2, 14, 3, 22],
        [41, 2, 13, 4],
        [2, 34, 3, 21],
        [2, 13, 3, 42],
        [40, 2, 14, 3, 1],
        [2, 13, 3, 42]
    ]

    initialPhaseplan = tlsf.change_phase_plan((5, 5), 60)
    episode_rewards = []
    episode_epsilons = []
    global steps_done
    global eps_threshold
    try:
        traci.close()
    except traci.exceptions.FatalTraCIError:
        pass

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    env = env = gym.make('TrafficEnv-V0', render_mode ='console', starting_phases = starting_phases, phase_lengths = phase_lengths, simulation_time= 3600,phase_change_step=5)
    global steps_done
    steps_done = 0
    with wandb.init(config=config):
        config = wandb.config
        num_episodes = config.num_episodes
        TAU = config.TAU
        learning_rate = config.learning_rate
        batch_size = config.batch_size
        gamma = config.gamma
        eps_start = config.eps_start
        eps_end = 0.05
        eps_decay = config.eps_decay
        use_phaseplan = config.use_phaseplan

        # Get number of actions from gym action space
        n_actions = env.action_space.n
        # Get the number of state observations
        state, info = env.reset()

        policy_net = DQN(env.observation_space, n_actions, config.fc_layer_size_1, config.fc_layer_size_2, config.fc_phaseplan_layer_size, use_phaseplan).to(device)
        target_net = DQN(env.observation_space, n_actions, config.fc_layer_size_1, config.fc_layer_size_2, config.fc_phaseplan_layer_size, use_phaseplan).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        #wandb.watch(policy_net)
        #wandb.watch(target_net)

        optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate, amsgrad=True)
        memory = ReplayMemory(10000)


        for i_episode in range(num_episodes):
            # Initialize the environment and get its state
            init_state, info = env.reset()
            #state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            state  = {
                'occupancy' : torch.tensor(init_state['occupancy'], dtype=torch.float32, device=device).unsqueeze(0),
                'vehicle_count' : torch.tensor(init_state['vehicle_count'], dtype=torch.float32, device=device).unsqueeze(0)
            }
            if use_phaseplan:
                state['last_five_phaseplan'] = torch.tensor(init_state['last_five_phaseplan'], dtype=torch.float32, device=device).unsqueeze(0)
            print("Episode number: ", i_episode)
            reward_per_episode = 0

            prev_action = select_action(state, policy_net, env, device, eps_start, eps_end, eps_decay)
            for t in count():
                action = prev_action
                wandb.log({"action": action.item()})
                observation, reward, terminated, truncated, _ = env.step(np.array([action.item()]))
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated
                reward_per_episode += reward.item()

                prev_action = select_action(state, policy_net, env, device, eps_start, eps_end, eps_decay)

                if terminated:
                    next_state = None
                else:
                    next_state = {
                        'occupancy' : torch.tensor(observation['occupancy'], dtype=torch.float32, device=device).unsqueeze(0),
                        'vehicle_count' : torch.tensor(observation['vehicle_count'], dtype=torch.float32, device=device).unsqueeze(0)
                    }
                    if use_phaseplan:
                        next_state['last_five_phaseplan'] = torch.tensor(observation['last_five_phaseplan'], dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma, device, use_phaseplan)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_epsilons.append(eps_threshold)
                    this_episode_threshold = eps_threshold

                    episode_rewards.append(reward_per_episode)
                    this_episode_reward = reward_per_episode

                    wandb.log({"reward": this_episode_reward, "episode": i_episode})
                    wandb.log({"epsilon": this_episode_threshold})
                    if len(episode_rewards) > 50:
                        mean_reward_value = rewards_mean(50, episode_rewards)
                        wandb.log({"mean_reward_value": mean_reward_value})
                    break
        last_mean_reward = mean_reward_value
        wandb.log({"last_mean_reward": last_mean_reward})
        sum_of_rewards = sum(episode_rewards)/len(episode_rewards)
        wandb.log({"sum_of_rewards": sum_of_rewards})
        onnx = torch.onnx.dynamo_export(policy_net, memory.sample(1)[0][0])
        onnx.save("model.onnx")
        wandb.save("model.onnx")

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, observation_space, n_actions, fc_layer_size_1 = 64, fc_layer_size_2 = 256, fc_phaseplan_layer_size = 128, use_phaseplan=0):
        super(DQN, self).__init__()
        # print(f"Layer sizes: {fc_layer_size_1}, {fc_layer_size_2}")

        # Occupancy és vehicle_count feldolgozása
        self.fc_occupancy = nn.Linear(observation_space['occupancy'].shape[0], fc_layer_size_1)
        self.fc_vehicle_count = nn.Linear(observation_space['vehicle_count'].shape[0], fc_layer_size_1)
        self.use_phaseplan = use_phaseplan

        if self.use_phaseplan:
            # Last five phaseplan feldolgozása
            phaseplan_shape = observation_space['last_five_phaseplan'].shape
            self.conv_phaseplan = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
            self.fc_phaseplan = nn.Linear(32 * phaseplan_shape[0] * phaseplan_shape[1] * phaseplan_shape[2], fc_phaseplan_layer_size)
            self.fc1 = nn.Linear(fc_layer_size_1 + fc_layer_size_1 + fc_phaseplan_layer_size, fc_layer_size_2)
        else:
            self.fc1 = nn.Linear(fc_layer_size_1 + fc_layer_size_1, fc_layer_size_2)  # Removed + 128
        
        self.fc2 = nn.Linear(fc_layer_size_2, n_actions)

    def forward(self,x):
        occupancy = F.relu(self.fc_occupancy(x['occupancy']))
        vehicle_count = F.relu(self.fc_vehicle_count(x['vehicle_count']))

        if self.use_phaseplan:
            # print("Last five phaseplan shape: ", x['last_five_phaseplan'].shape)
            phaseplan = x['last_five_phaseplan'].unsqueeze(1)  # Add channel dimension
            phaseplan = F.relu(self.conv_phaseplan(phaseplan))
            phaseplan = phaseplan.view(phaseplan.size(0), -1)
            phaseplan = F.relu(self.fc_phaseplan(phaseplan))
            combined = torch.cat((occupancy, vehicle_count, phaseplan), dim=1)
        else:
            combined = torch.cat((occupancy, vehicle_count), dim=1)

        # print(f"Occupancy shape: {occupancy.shape}")
        # print(f"Vehicle count shape: {vehicle_count.shape}")
        # print(f"Phaseplan shape: {phaseplan.shape}")

        # print(f"Combined shape: {combined.shape}")
        x = F.relu(self.fc1(combined))
        # print(f"After fc1 shape: {x.shape}")
        return self.fc2(x)


def select_action(state, policy_net: DQN, env, device, eps_start, eps_end, eps_decay):
    sample = random.random()
    global eps_threshold
    global steps_done
    eps_threshold = eps_end + (eps_start - eps_end) * \
                    math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def rewards_mean(mean_window, episode_rewards):
    if len(episode_rewards) < mean_window:
        return float(torch.tensor(episode_rewards, dtype=torch.float).mean().item()) if episode_rewards else 0.0
    else:
        return float(torch.tensor(episode_rewards[-100:], dtype=torch.float).mean().item())

def optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma, device, use_phaseplan):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    #non_final_next_states = torch.cat([s for s in batch.next_state
    #                                            if s is not None])
    non_final_next_states = {
        'occupancy': torch.cat([s['occupancy'] for s in batch.next_state if s is not None]),
        'vehicle_count': torch.cat([s['vehicle_count'] for s in batch.next_state if s is not None]),
        # 'last_five_phaseplan': torch.cat([s['last_five_phaseplan'] for s in batch.next_state if s is not None])
    }
    if use_phaseplan:
        non_final_next_states['last_five_phaseplan'] = torch.cat([s['last_five_phaseplan'] for s in batch.next_state if s is not None])

    #state_batch = torch.cat(batch.state)
    state_batch = {
        'occupancy': torch.cat([s['occupancy'] for s in batch.state]),
        'vehicle_count': torch.cat([s['vehicle_count'] for s in batch.state]),
        # 'last_five_phaseplan': torch.cat([s['last_five_phaseplan'] for s in batch.state])
    }
    if use_phaseplan:
        state_batch['last_five_phaseplan'] = torch.cat([s['last_five_phaseplan'] for s in batch.state])

    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if __name__ == "__main__":
    train(config)