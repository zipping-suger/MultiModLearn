import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal

import os
import shutil
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot import TwoLinkRobotIK  # Ensure this is available


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_size, action_dim)  # Outputs mean of Gaussian
        self.log_std_head = nn.Linear(hidden_size, action_dim)  # Outputs log-std

        # Initialize weights properly
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.fc(x)
        mu = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-5, max=2)  # Avoid extreme values
        std = torch.exp(log_std)
        return mu, std


class REINFORCEAgent:
    def __init__(self, state_dim, action_dim, hidden_size=64, lr=1e-3, gamma=0.99):
        self.policy_net = PolicyNetwork(state_dim, hidden_size, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        mu, std = self.policy_net(state)

        # Debugging step to catch NaNs early
        if torch.isnan(mu).any() or torch.isnan(std).any():
            raise ValueError(f"NaN detected in policy output: mu={mu}, std={std}")

        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach().numpy(), log_prob

    def compute_returns(self, rewards):
        returns = rewards.copy()
        returns = torch.tensor(returns, dtype=torch.float32)

        # Safeguard against division by zero
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            returns = returns - returns.mean()
        return returns

    def update_policy(self, states, actions, log_probs, rewards):
        returns = self.compute_returns(rewards)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        log_probs = torch.stack(log_probs)

        # Compute loss
        loss = -(log_probs * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train_reinforce(agent, num_episodes, update_num, batch_size, log_dir):
    writer = SummaryWriter(log_dir)
    robot = TwoLinkRobotIK(3.0, 3.0)

    for episode in range(num_episodes):
        target_position = robot.sample_from_workspace(batch_size)
        state = target_position
        
        for _ in range(update_num):
            states, actions, log_probs, rewards = [], [], [], []

            # Collect one trajectory
            action, log_prob = agent.select_action(state)
            reward = -robot.evaluate_batch(action, target_position)
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            # Update policy
            agent.update_policy(states, actions, log_probs, rewards)

            # Log average reward over the batch
            avg_reward = sum(reward) / len(reward)
            writer.add_scalar('Average Reward', avg_reward, episode)

        if episode % 100 == 0:
            print(f"Episode {episode}, Average Reward: {avg_reward}")

    writer.close()


def main():
    # Hyperparameters
    state_dim = 2
    action_dim = 2
    hidden_size = 64
    learning_rate = 1e-3
    num_episodes = 10000
    update_num = 10
    batch_size = 32

    # Initialize REINFORCE agent
    agent = REINFORCEAgent(state_dim, action_dim, hidden_size, lr=learning_rate)

    # Setup logging
    log_dir = os.path.join("logs", "REINFORCE")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Train the agent
    train_reinforce(agent, num_episodes, update_num, batch_size, log_dir)

    # Save the model
    model_path = os.path.join(log_dir, "reinforce_model.pth")
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"Model saved at {model_path}")


if __name__ == "__main__":
    main()
