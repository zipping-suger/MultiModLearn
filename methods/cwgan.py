import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import shutil
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import RobotDataset

# Generator architecture remains the same
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, condition_size):
        super(Generator, self).__init__()
        self.latent_size = input_size
        self.model = nn.Sequential(
            nn.Linear(input_size + condition_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, condition):
        x = torch.cat([x, condition], dim=1)
        return self.model(x)

# Critic modified to output raw logits instead of sigmoid
class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, condition_size):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size + condition_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
            # Removed sigmoid to get raw logits
        )

    def forward(self, x, condition):
        x = torch.cat([x, condition], dim=1)
        return self.model(x)

def gradient_penalty(critic, real_samples, fake_samples, condition, device):
    
    alpha = torch.rand(real_samples.size(0), 1, device=device)
    alpha = alpha.expand_as(real_samples)
    
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    d_interpolates = critic(interpolates, condition)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return penalty


def train(dataloader, generator, critic, optimizer_g, optimizer_d, num_epochs, writer, lambda_gp=10, n_critic=5, device="cpu"):
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            position = batch['position'].float().to(device)
            angles = batch['angles'].float().to(device)

            # Train Critic
            for _ in range(n_critic):
                z = torch.randn(position.size(0), generator.latent_size, device=device)
                fake_angles = generator(z, position)

                real_validity = critic(angles, position)
                fake_validity = critic(fake_angles.detach(), position)
                gp = gradient_penalty(critic, angles, fake_angles, position, device)

                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp

                optimizer_d.zero_grad()
                d_loss.backward()
                optimizer_d.step()

            # Train Generator
            z = torch.randn(position.size(0), generator.latent_size, device=device)
            fake_angles = generator(z, position)
            g_loss = -torch.mean(critic(fake_angles, position))

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            # Logging
            writer.add_scalar('Loss/Critic', d_loss.item(), epoch * len(dataloader) + i)
            writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(dataloader) + i)

        print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Conditional Wasserstein GAN (CWGAN) model.')
    parser.add_argument('--data_file_path', type=str, required=False, default='data/gradient_data.npy', help='Path to the data file.')
    args = parser.parse_args()
    
    # Hyperparameters
    latent_size = 2
    hidden_size = 64
    output_size = 2
    condition_size = 2
    num_epochs = 300
    batch_size = 100
    learning_rate = 0.0002
    lambda_gp = 10
    n_critic = 5

    # Load dataset
    dataset = RobotDataset(args.data_file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(latent_size, hidden_size, output_size, condition_size).to(device)
    critic = Critic(output_size, hidden_size, 1, condition_size).to(device)

    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(critic.parameters(), lr=learning_rate)

    # Initialize SummaryWriter
    folder = os.path.join("logs", f"cwgan_model_{os.path.basename(args.data_file_path).split('.')[0]}")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    writer = SummaryWriter(folder)
    train(dataloader, generator, critic, optimizer_g, optimizer_d, num_epochs, writer, lambda_gp, n_critic, device)
    
    # Save models
    generator_model_name = f"generator_{os.path.basename(args.data_file_path).split('.')[0]}.pth"
    critic_model_name = f"critic{os.path.basename(args.data_file_path).split('.')[0]}.pth"

    generator_model_path = os.path.join(folder, generator_model_name)
    critic_model_path = os.path.join(folder, critic_model_name)

    torch.save(generator.state_dict(), generator_model_path)
    torch.save(critic.state_dict(), critic_model_path)

    print(f"Generator model saved at {generator_model_path}")
    print(f"Critic model saved at {critic_model_path}")
    
    writer.close()