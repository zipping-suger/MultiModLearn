import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gym

class EnergyModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(EnergyModel, self).__init__()
        self.energy_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.energy_net(sa).squeeze(-1)

def sample_action(energy_model, state, action_dim, device, num_samples=10):
    state = state.unsqueeze(0).repeat(num_samples, 1)
    actions = torch.rand((num_samples, action_dim), device=device) * 2 - 1  # Action range [-1, 1]
    energies = energy_model(state, actions)
    probs = torch.softmax(-energies, dim=0)
    dist = Categorical(probs)
    action_idx = dist.sample()
    return actions[action_idx], probs[action_idx]

def train_energy_reinforce(env, energy_model, optimizer, num_episodes, gamma=0.99, num_samples=10, device="cpu"):
    for episode in range(num_episodes):
        state = env.reset()
        rewards, states, actions, log_probs = [], [], [], []
        
        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            action, log_prob = sample_action(energy_model, state_tensor, env.action_space.shape[0], device, num_samples)
            next_state, reward, done, _ = env.step(action.cpu().numpy())
            
            rewards.append(reward)
            states.append(state_tensor)
            actions.append(action)
            log_probs.append(torch.log(log_prob))
            
            state = next_state
            if done:
                break
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Update energy model
        optimizer.zero_grad()
        loss = 0
        for state, action, log_prob, G in zip(states, actions, log_probs, returns):
            energy = energy_model(state.unsqueeze(0), action.unsqueeze(0))
            loss += -log_prob * G + energy.mean()  # REINFORCE + Energy penalty
        loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {sum(rewards):.2f}")

# Example usage
if __name__ == "__main__":
    
    env = gym.make("Pendulum-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    energy_model = EnergyModel(env.observation_space.shape[0], env.action_space.shape[0], hidden_dim=64).to(device)
    optimizer = optim.Adam(energy_model.parameters(), lr=0.001)

    train_energy_reinforce(env, energy_model, optimizer, num_episodes=500, device=device)