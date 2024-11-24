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

# InfoNCE-style loss function
def info_nce_loss(energy_model, x, y, counter_samples):
    positive_energy = -energy_model(x, y)
    neg_energies = torch.stack([-energy_model(x, neg) for neg in counter_samples], dim=1)
    denominator = torch.logsumexp(torch.cat([positive_energy.unsqueeze(-1), neg_energies], dim=-1), dim=-1)
    return torch.mean(denominator - positive_energy)

# Counter-sample generation
def generate_counter_samples(y_min, y_max, batch_size, neg_count, device):
    return [torch.rand((batch_size, y_min.size(-1)), device=device) * (y_max - y_min) + y_min for _ in range(neg_count)]

# Training loop
def train(model, dataloader, criterion, optimizer, scheduler, num_epochs, writer, y_min, y_max, neg_count, device):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            positions = batch['position'].to(device)
            angles = batch['angles'].to(device)
            
            counter_samples = generate_counter_samples(y_min, y_max, positions.size(0), neg_count, device)
            
            # Compute loss
            loss = criterion(model, positions, angles, counter_samples)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
# Derivative-free optimizer for inference
def infer_angles(energy_model, target_position, y_min, y_max, samples=16384, iterations=3, sigma_init=0.33, scale=0.5):
    """
    Find angles that minimize energy for a given target position.

    Args:
        energy_model: Trained energy-based model.
        target_position: Target position (Cartesian coordinates), shape: (1, input_dim).
        y_min: Minimum values for joint angles (tensor).
        y_max: Maximum values for joint angles (tensor).
        samples: Number of random samples for initial exploration.
        iterations: Number of optimization iterations.
        sigma_init: Initial noise level for exploration.
        scale: Scaling factor for noise reduction.

    Returns:
        Optimal joint angles minimizing the energy.
    """
    device = target_position.device
    target_position = target_position.repeat(samples, 1)  # Repeat for batch inference
    sigma = sigma_init

    # Initialize random joint angle samples
    angles = torch.rand((samples, y_min.size(-1)), device=device) * (y_max - y_min) + y_min

    for _ in range(iterations):
        # Compute energies for current samples
        energies = energy_model(target_position, angles).detach()

        # Softmax over negative energies for sampling probabilities
        probabilities = torch.softmax(-energies, dim=0)

        # Resample based on probabilities
        indices = torch.multinomial(probabilities, num_samples=samples, replacement=True)
        angles = angles[indices]

        # Add noise for exploration
        angles += torch.randn_like(angles) * sigma
        angles = torch.clamp(angles, y_min, y_max)  # Clamp to valid joint angle bounds

        # Reduce noise scale
        sigma *= scale

    # Return the angles corresponding to the minimum energy
    best_idx = torch.argmin(energies)
    return angles[best_idx].unsqueeze(0)  # Shape: (1, action_dim)

# Main function
def main():
    # Hyperparameters
    input_size = 2  # x, y positions
    action_size = 2  # joint angles
    hidden_size = 64
    num_epochs = 50
    batch_size = 32
    neg_count = 256
    learning_rate = 0.001

    # Data bounds for counter-sample generation
    y_min = torch.tensor([-3.14, -3.14])  # Joint angle limits (radians)
    y_max = torch.tensor([3.14, 3.14])

    # Load dataset
    data_file_path = 'data/gradient_data_rs.npy'
    dataset = RobotDataset(data_file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnergyModel(input_size, action_size, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Setup TensorBoard
    folder = os.path.join("logs", f"ibc_model_{os.path.basename(data_file_path).split('.')[0]}")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    writer = SummaryWriter(folder)

    # Train the model
    train(model, dataloader, info_nce_loss, optimizer, scheduler, num_epochs, writer, y_min.to(device), y_max.to(device), neg_count, device)

    # Save the trained model
    model_name = f"ibc_model_{os.path.basename(data_file_path).split('.')[0]}.pth"
    model_path = os.path.join(folder, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    # Close the writer
    writer.close()

if __name__ == "__main__":
    main()
