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
from robot import TwoLinkRobotIK

# Define Energy-based Model using MLP
class EnergyModel(nn.Module):
    def __init__(self, input_size, action_size, hidden_size):
        super(EnergyModel, self).__init__()
        self.energy_net = nn.Sequential(
            nn.Linear(input_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, y):
        combined = torch.cat([x, y], dim=-1)
        return self.energy_net(combined).squeeze(-1)

# InfoNCE-style loss function for energy model
def info_nce_loss(energy_model, x, y, counter_samples):
    positive_energy = -energy_model(x, y)
    neg_energies = torch.stack([-energy_model(x, neg) for neg in counter_samples], dim=1)
    denominator = torch.logsumexp(torch.cat([positive_energy.unsqueeze(-1), neg_energies], dim=-1), dim=-1)
    return torch.mean(denominator - positive_energy)

# Counter-sample generation
def generate_counter_samples_rand(y_min, y_max, batch_size, neg_count, device):
    return [torch.rand((batch_size, y_min.size(-1)), device=device) * (y_max - y_min) + y_min for _ in range(neg_count)]

# Generator model
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

# Training loop for energy model
def train_energy_model(dataloader, energy_model, optimizer_e, scheduler_e, num_epochs, writer, y_min, y_max, neg_count, device):
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            position = batch['position'].float().to(device)
            angles = batch['angles'].float().to(device)
            
            # Generate counter samples by random sampling
            counter_samples = generate_counter_samples_rand(y_min, y_max, position.size(0), neg_count, device)
            
            # Compute energy model loss
            e_loss = info_nce_loss(energy_model, position, angles, counter_samples)
            
            optimizer_e.zero_grad()
            e_loss.backward()
            optimizer_e.step()

            # Log losses
            writer.add_scalar('Loss/EnergyModel', e_loss.item(), epoch * len(dataloader) + i)

        scheduler_e.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Energy Model Loss: {e_loss.item():.4f}")

# Training loop for generator
def train_generator(dataloader, generator, energy_model, optimizer_g, num_epochs, writer, device, use_random_positions=False):
    
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            if use_random_positions:
                position = robot.sample_from_workspace(batch['position'].size(0)).float().to(device)
            else:
                position = batch['position'].float().to(device)
            
            # Update generator
            z = torch.randn(position.size(0), generator.latent_size, device=device)
            fake_angles = generator(z, position)
            g_loss = torch.mean(energy_model(position, fake_angles))
            
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            # Log losses
            writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(dataloader) + i)

        print(f"Epoch [{epoch+1}/{num_epochs}], Generator Loss: {g_loss.item():.4f}")

if __name__ == "__main__":
    global robot
    
    
    parser = argparse.ArgumentParser(description='Train Energy-Based GAN model.')
    parser.add_argument('--data_file_path', type=str, default='data/gradient_data_rs.npy', help='Path to the data file.')
    args = parser.parse_args()
    
    # Hyperparameters
    latent_size = 2
    hidden_size = 64
    output_size = 2
    condition_size = 2
    num_epochs_energy = 100
    num_epochs_generator = 300
    batch_size = 100
    learning_rate_g = 0.0002
    learning_rate_e = 0.001
    neg_count = 256

    # Data bounds for counter-sample generation
    y_min = torch.tensor([-3.14, -3.14])
    y_max = torch.tensor([3.14, 3.14])
    
    # Load the robot
    # Robot parameters
    L1 = 3.0  # Length of link 1
    L2 = 3.0  # Length of link 2

    # Create the robot
    robot = TwoLinkRobotIK(L1, L2)

    # Load dataset
    dataset = RobotDataset(args.data_file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize models
    generator = Generator(latent_size, hidden_size, output_size, condition_size).to(device)
    energy_model = EnergyModel(condition_size, output_size, hidden_size).to(device)

    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g)
    optimizer_e = optim.Adam(energy_model.parameters(), lr=learning_rate_e)
    scheduler_e = optim.lr_scheduler.StepLR(optimizer_e, step_size=20, gamma=0.5)

    # Initialize SummaryWriter
    folder = os.path.join("logs", f"ebgan2_model_{os.path.basename(args.data_file_path).split('.')[0]}")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    writer = SummaryWriter(folder)

    # Train the energy model
    train_energy_model(dataloader, energy_model, optimizer_e, scheduler_e, num_epochs_energy, writer, y_min.to(device), y_max.to(device), neg_count, device)
    
    # Train the generator using dataset positions
    train_generator(dataloader, generator, energy_model, optimizer_g, num_epochs_generator, writer, device, use_random_positions=False)
    
    # # Train the generator using random positions from workspace
    # train_generator(dataloader, generator, energy_model, optimizer_g, num_epochs_generator, writer, device, use_random_positions=True)
    
    # Save models
    generator_model_name = f"generator_{os.path.basename(args.data_file_path).split('.')[0]}.pth"
    energy_model_name = f"energy_model_{os.path.basename(args.data_file_path).split('.')[0]}.pth"

    generator_model_path = os.path.join(folder, generator_model_name)
    energy_model_path = os.path.join(folder, energy_model_name)

    torch.save(generator.state_dict(), generator_model_path)
    torch.save(energy_model.state_dict(), energy_model_path)

    print(f"Generator model saved at {generator_model_path}")
    print(f"Energy model saved at {energy_model_path}")
    
    writer.close()