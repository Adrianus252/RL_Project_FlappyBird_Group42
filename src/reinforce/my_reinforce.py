import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# === Policy Network for REINFORCE ===
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Softmax(dim=-1)  # Outputs action probabilities
        )

    def forward(self, x):
        return self.net(x)

# === REINFORCE Agent ===
class ReinforceAgent:
    def __init__(self, obs_dim, act_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma  # Discount factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(obs_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.memory = []  # Stores (state, action, reward) per episode

    # --- Select an action ---
    def select_action(self, state):
        state_t = torch.FloatTensor(state).to(self.device)
        probs = self.policy(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)  # Returns action & log_prob

    # --- Store a transition ---
    def store_transition(self, state, action, reward):
        self.memory.append((state, action, reward))

    # --- Compute discounted rewards ---
    def compute_returns(self):
        returns = []
        G = 0
        for _, _, reward in reversed(self.memory):
            G = reward + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    # --- Update policy ---
    def update(self):
        returns = self.compute_returns()
        policy_loss = []

        for (state, action, reward), G in zip(self.memory, returns):
            state_t = torch.FloatTensor(state).to(self.device)
            action_t = torch.tensor(action, dtype=torch.int64).to(self.device)

            probs = self.policy(state_t)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action_t)

            policy_loss.append(-log_prob * G)  # Gradient ascent

        loss = torch.stack(policy_loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory = []  # Clear memory after update

    # --- Save & Load ---
    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
