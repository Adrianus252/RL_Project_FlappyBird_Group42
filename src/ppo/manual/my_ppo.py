import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# === Actor Netzwerk ===
class ActorNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # Trainierbare log_std für Exploration

    def forward(self, x):
        logits = self.policy(x)
        probs = torch.softmax(logits, dim=-1)  # Sicherstellen, dass Werte Wahrscheinlichkeiten sind
        std = self.log_std.exp()
        return probs, std

# === Critic Netzwerk ===
class CriticNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.value(x)

# === Speicher für PPO ===
class PPOMemory:
    def __init__(self):
        self.states, self.actions, self.log_probs, self.rewards, self.dones, self.values = [], [], [], [], [], []

    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def get_all(self):
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.dones, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
        )

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

# === PPO-Agent ===
class PPOAgent:
    def __init__(
        self, obs_dim, act_dim, lr=1e-4, gamma=0.999, lam=0.95, eps_clip=0.2, 
        K_epochs=5, batch_size=64, entropy_coef=0.01
    ):
        self.gamma, self.lam, self.eps_clip = gamma, lam, eps_clip
        self.K_epochs, self.batch_size, self.entropy_coef = K_epochs, batch_size, entropy_coef

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = ActorNet(obs_dim, act_dim).to(self.device)
        self.critic = CriticNet(obs_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.memory = PPOMemory()

    def compute_gae(self, rewards, values, dones, next_value):
        advantages, gae = [], 0.0
        values = np.append(values, next_value)  # Next Value hinzufügen für korrekte Berechnung
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i+1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return np.array(advantages, dtype=np.float32)

    def update(self, target_kl=0.05): # relaxed KL target
        states, actions, old_log_probs, rewards, dones, values = self.memory.get_all()

        # Next Value aus dem letzten Zustand berechnen (falls die Episode nicht endet)
        last_state = torch.FloatTensor(states[-1]).to(self.device)
        next_value = self.critic(last_state).item() * (1 - dones[-1])

        advantages = self.compute_gae(rewards, values, dones, next_value)
        #advantages = np.array(advantages, dtype=np.float32)
        #  Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values

        # Konvertiere in Tensoren
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        dataset = torch.utils.data.TensorDataset(states_t, actions_t, old_log_probs_t, advantages_t, returns_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        #for _ in range(self.K_epochs):
        for epoch in range(self.K_epochs):
            total_kl = 0.0
            for batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in dataloader:
                probs, _ = self.actor(batch_states)
                value_pred = self.critic(batch_states).squeeze()

                dist = torch.distributions.Categorical(probs=probs)
                new_log_probs = dist.log_prob(batch_actions)

                kl = (batch_old_log_probs - new_log_probs).mean()
                total_kl += kl.item()

                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(value_pred, batch_returns)
                entropy = dist.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

            avg_kl = total_kl / len(dataloader)
            if avg_kl > target_kl:
                print(f"Early stopping PPO update due to high KL: {avg_kl:.5f} > {target_kl}")
                break

        self.memory.clear()

    def select_action(self, state):
        state_t = torch.FloatTensor(state).to(self.device)
        probs, std = self.actor(state_t)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), self.critic(state_t).item()

    def store_transition(self, state, action, log_prob, reward, done, value, info):
        reward += 0.1  # Überleben belohnen
        if action == 1:
            reward -= 0.02  # Kleine Bestrafung für Springen
        if "score" in info:
            reward += info["score"] * 5  # Fortschritt belohnen
        self.memory.store(state, action, log_prob, reward, done, value)

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
