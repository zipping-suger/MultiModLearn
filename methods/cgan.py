import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import shutil
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import RobotDataset

# Generator architecture remains the same
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, condition_size):
        super(Generator, self).__init__()
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

# Discriminator modified to output raw logits instead of sigmoid
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, condition_size):
        super(Discriminator, self).__init__()
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

def compute_discriminator_loss(discriminator, real_samples, fake_samples, condition):
    """Compute discriminator loss using log probabilities as per the formula"""
    # Get discriminator outputs
    d_real = discriminator(real_samples, condition)
    d_fake = discriminator(fake_samples, condition)
    
    # Compute log probabilities
    log_prob_real = torch.log(torch.sigmoid(d_real) + 1e-10)  # Add small epsilon to prevent log(0)
    log_prob_fake = torch.log(1 - torch.sigmoid(d_fake) + 1e-10)
    
    # Average over batch
    loss = -(torch.mean(log_prob_real) + torch.mean(log_prob_fake))
    
    return loss, torch.mean(torch.sigmoid(d_real)), torch.mean(torch.sigmoid(d_fake))

def compute_generator_loss(discriminator, fake_samples, condition):
    """Compute generator loss using log probabilities as per the formula"""
    d_fake = discriminator(fake_samples, condition)
    loss = -torch.mean(torch.log(torch.sigmoid(d_fake) + 1e-10))
    return loss

def train():
    # Hyperparameters
    latent_size = 2
    hidden_size = 64
    output_size = 2
    condition_size = 2
    num_epochs = 300
    batch_size = 100
    learning_rate = 0.0002

    # Load dataset
    data_file_path = 'data/gradient_data.npy'
    dataset = RobotDataset(data_file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    generator = Generator(latent_size, hidden_size, output_size, condition_size)
    discriminator = Discriminator(output_size, hidden_size, 1, condition_size)

    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Initialize SummaryWriter
    folder = os.path.join("../logs", f"cgan_model_{os.path.basename(data_file_path).split('.')[0]}")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    writer = SummaryWriter(folder)

    # Training loop following the algorithm in the image
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            position = batch['position'].float()
            angles = batch['angles'].float()
            
            # Step 1: Update discriminator
            for _ in range(20):  # Can adjust number of discriminator updates per generator update
                # Draw noise samples
                z = torch.randn(batch_size, latent_size)
                
                # Generate fake samples
                fake_angles = generator(z, position)
                
                # Update discriminator
                d_loss, real_score, fake_score = compute_discriminator_loss(
                    discriminator, angles, fake_angles.detach(), position
                )
                
                optimizer_d.zero_grad()
                d_loss.backward()
                optimizer_d.step()

            # Step 2 & 3: Update generator
            z = torch.randn(batch_size, latent_size)
            fake_angles = generator(z, position)
            g_loss = compute_generator_loss(discriminator, fake_angles, position)
            
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            # Log losses and scores
            writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch * len(dataloader) + i)
            writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(dataloader) + i)
            writer.add_scalar('Score/Real', real_score.item(), epoch * len(dataloader) + i)
            writer.add_scalar('Score/Fake', fake_score.item(), epoch * len(dataloader) + i)

        print(f"Epoch [{epoch+1}/{num_epochs}], Discriminator Loss: {d_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}")

    # Save models
    generator_model_name = f"generator_{os.path.basename(data_file_path).split('.')[0]}.pth"
    discriminator_model_name = f"discriminator_{os.path.basename(data_file_path).split('.')[0]}.pth"

    generator_model_path = os.path.join(folder, generator_model_name)
    discriminator_model_path = os.path.join(folder, discriminator_model_name)

    torch.save(generator.state_dict(), generator_model_path)
    torch.save(discriminator.state_dict(), discriminator_model_path)

    print(f"Generator model saved at {generator_model_path}")
    print(f"Discriminator model saved at {discriminator_model_path}")
    
    writer.close()

if __name__ == "__main__":
    train()