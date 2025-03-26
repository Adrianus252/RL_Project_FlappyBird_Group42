import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# === ActorCritic Network for discrete actions ===
class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(64, act_dim)  # discrete logits
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        z = self.shared(x)
        logits = self.policy_head(z)
        value = self.value_head(z)
        return logits, value

# === Simple Memory Buffer for One Episode ===
class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

# === PPO Agent ===
class PPOAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        lr=2.5e-4,
        gamma=0.99,
        lam=0.95,
        eps_clip=0.2,
        K_epochs=4,
        batch_size=64
    ):
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCriticNet(obs_dim, act_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)

        self.memory = PPOMemory()

    # --- Action selection ---
    def select_action(self, state):
        state_t = torch.FloatTensor(state).to(self.device)
        logits, value = self.net(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    # --- Store transition in memory ---
    def store_transition(self, state, action, log_prob, reward, done, value, info):
        # Strafe für zu große Sprünge
        if action > 0.8:
            reward -= 0.1  

        # Bestrafe das Hochfliegen zu stark
        reward -= abs(action) * 0.1  

        # Stärkere Belohnung fürs Überleben
        reward += 0.05  # Kleiner, aber langfristig hilfreich

        # Bonus für das erfolgreiche Durchfliegen einer Pipe
        if "score" in info:
            reward += info["score"] * 5  

        self.memory.store(state, action, log_prob, reward, done, value)

    # --- GAE Computation ---
    def compute_gae(self, next_value):
        rewards = self.memory.rewards
        dones = self.memory.dones
        values = self.memory.values + [next_value]

        advantages = []
        gae = 0.0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i+1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return advantages

    # --- PPO Update ---
    def update(self):
        # Prepare
        states = np.array(self.memory.states, dtype=np.float32)
        actions = np.array(self.memory.actions)
        old_log_probs = np.array(self.memory.log_probs, dtype=np.float32)
        dones = np.array(self.memory.dones, dtype=np.float32)
        values = np.array(self.memory.values, dtype=np.float32)

        # Next value (for final GAE)
        next_value = 0.0
        if dones[-1] == 0:
            s_t = torch.FloatTensor(states[-1]).to(self.device)
            _, nxt_val = self.net(s_t)
            next_value = nxt_val.item()

        advantages = self.compute_gae(next_value)
        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + values

        # Convert to torch
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        dataset = torch.utils.data.TensorDataset(states_t, actions_t, old_log_probs_t, advantages_t, returns_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.K_epochs):
            for batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in dataloader:
                logits, val = self.net(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)

                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * batch_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = torch.nn.MSELoss()(val.squeeze(), batch_returns)
                loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.memory.clear()

    # --- Save & Load ---
    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
