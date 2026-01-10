import math
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Q Network model
class DQN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=3):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 2000
TAU = 0.005
LR = 3e-4
MEMORY_CAPACITY = 10000

# Initialize networks
policy_net = DQN(input_dim=3, hidden_dim=128, output_dim=3).to(device)
target_net = DQN(input_dim=3, hidden_dim=128, output_dim=3).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_CAPACITY)

env = gym.make('StateTrafficEnv', render_mode ='console')

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # state should be 2-D: [1, input_dim]
            return policy_net(state).argmax(dim=1).view(1,1)
    else:
        # randomly pick one of 3 actions
        return torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    # Convert to tensors
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    done_batch = torch.cat(batch.done)
    non_final_mask = ~done_batch
    non_final_next_states = torch.cat([s for s, d in zip(batch.next_state, batch.done) if not d])

    # Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) using target_net
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = reward_batch + (GAMMA * next_state_values * (~done_batch).float())

    criterion = nn.SmoothL1Loss()  # Huber loss
    loss = criterion(state_action_values.squeeze(), expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# ====== Main training loop (sketch) ======
num_episodes = 500  # adjust as needed

for i_episode in range(num_episodes):
    # Initialize the environment and get initial state
    # YOUR ENVIRONMENT → returns initial_state (np or torch array) of shape (3,)
    state_np = env.reset()
    state = torch.tensor([state_np], dtype=torch.float32, device=device)

    for t in range(1000):  # or until done
        action = select_action(state)
        # Execute action in ENVIRONMENT
        next_state_np, reward_val, done_flag, info = env.step(action.item())
        reward = torch.tensor([reward_val], device=device)
        done = torch.tensor([done_flag], device=device)
        if not done_flag:
            next_state = torch.tensor([next_state_np], dtype=torch.float32, device=device)
        else:
            next_state = torch.zeros_like(state)  # placeholder

        # Store transition
        memory.push(state, action, next_state, reward, done)

        # Move to next state
        state = next_state

        # Perform one optimization step
        optimize_model()

        # Soft update of target network
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU*policy_param.data + (1.0-TAU)*target_param.data)

        if done_flag:
            break

    # Optionally: print/log episode length, reward, etc.
    print(f"Episode {i_episode} finished after {t+1} timesteps")

print("Training complete")