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

# Energy-based Model
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

# Generator model
class Generator(nn.Module):
    def __init__(self, latent_size, hidden_size, action_size, condition_size):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.model = nn.Sequential(
            nn.Linear(latent_size + condition_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x, condition):
        x = torch.cat([x, condition], dim=1)
        return self.model(x)

# InfoNCE-style loss function
def info_nce_loss(energy_model, x, y, counter_samples):
    positive_energy = -energy_model(x, y)
    neg_energies = torch.stack([-energy_model(x, neg) for neg in counter_samples], dim=1)
    denominator = torch.logsumexp(torch.cat([positive_energy.unsqueeze(-1), neg_energies], dim=-1), dim=-1)
    return torch.mean(denominator - positive_energy)

# Counter-sample generation
def generate_counter_samples(y_min, y_max, batch_size, neg_count, device):
    # make y_min and y_max the same device as the other tensors
    y_min = y_min.to(device)
    y_max = y_max.to(device)
    return [torch.rand((batch_size, y_min.size(-1)), device=device) * (y_max - y_min) + y_min 
            for _ in range(neg_count)]

# Training function
def train(dataloader, generator, energy_model, optimizer_g, optimizer_e, scheduler_e, scheduler_g,
          num_epochs, writer, y_min, y_max, neg_count, repeat_energy_updates, device):
    for epoch in range(num_epochs):
        epoch_e_loss = 0.0
        epoch_g_loss = 0.0
        i = 0
        for batch_x, batch_y in dataloader:
            x_input = batch_x.float().to(device)
            y_target = batch_y.float().to(device)
            
            # Update energy model
            for _ in range(repeat_energy_updates):
                # Generate counter samples
                counter_samples = generate_counter_samples(y_min, y_max, x_input.size(0), neg_count, device)
                
                # Add generator samples to counter samples
                z = torch.randn(x_input.size(0), generator.latent_size, device=device)
                with torch.no_grad():
                    fake_y = generator(z, x_input)
                counter_samples.append(fake_y)
                
                # Compute energy model loss
                e_loss = info_nce_loss(energy_model, x_input, y_target, counter_samples)
                
                optimizer_e.zero_grad()
                e_loss.backward()
                optimizer_e.step()
                
                epoch_e_loss += e_loss.item()
            
            # Update generator
            z = torch.randn(x_input.size(0), generator.latent_size, device=device)
            fake_y = generator(z, x_input)
            g_loss = torch.mean(energy_model(x_input, fake_y))
            
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
            
            epoch_g_loss += g_loss.item()
            
            # Log losses
            if writer:
                writer.add_scalar('Loss/EnergyModel', e_loss.item(), epoch * len(dataloader) + i)
                writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(dataloader) + i)
            i += 1
            
        scheduler_e.step()
        scheduler_g.step()
        avg_e_loss = epoch_e_loss / len(dataloader) / repeat_energy_updates
        avg_g_loss = epoch_g_loss / len(dataloader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Energy Loss: {avg_e_loss:.4f}, "
              f"Generator Loss: {avg_g_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train Energy-Based GAN model.')
    parser.add_argument('--data_file_path', type=str, default='data/gradient_data_rs.npy', help='Path to the dataset')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for tensorboard logs')
    args = parser.parse_args()

    # Hyperparameters
    action_size = 2
    condition_size = 2
    hidden_size = 64
    latent_size = 2
    
    num_epochs = 100
    batch_size = 32
    neg_count = 256
    repeat_energy_updates = 5
    
    learning_rate_e = 0.001
    learning_rate_g = 0.0005
    
    # Data bounds for counter-sample generation
    y_min = torch.tensor([-3.14, -3.14])  # Joint angle limits (radians)
    y_max = torch.tensor([3.14, 3.14])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create models
    generator = Generator(latent_size, hidden_size, action_size, condition_size).to(device)
    energy_model = EnergyModel(condition_size, action_size, hidden_size).to(device)
    
    # Setup optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g)
    optimizer_e = optim.Adam(energy_model.parameters(), lr=learning_rate_e)
    scheduler_e = optim.lr_scheduler.StepLR(optimizer_e, step_size=20, gamma=0.5)
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=20, gamma=0.5)
    
    # Setup tensorboard
    log_dir = os.path.join(args.log_dir, 'ebgan_training')
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)
    
    # Load dataset
    dataset = RobotDataset(args.data_file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train
    train(dataloader, generator, energy_model, optimizer_g, optimizer_e, scheduler_e, scheduler_g,
          num_epochs, writer, y_min.to(device), y_max.to(device), neg_count, 
          repeat_energy_updates, device)
    
    # Save models
    torch.save(generator.state_dict(), os.path.join(log_dir, 'generator.pth'))
    torch.save(energy_model.state_dict(), os.path.join(log_dir, 'energy_model.pth'))
    
    writer.close()

if __name__ == "__main__":
    main()