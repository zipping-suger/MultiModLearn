import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil

from data_pipeline import RobotDataset
from enum import Enum, auto

class NoiseType(Enum):
    DIAGONAL = auto()
    ISOTROPIC = auto()
    ISOTROPIC_ACROSS_CLUSTERS = auto()
    FIXED = auto()


class MDNGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_gaussians, condition_size, noise_type=NoiseType.ISOTROPIC, fixed_noise_level=None):
        super(MDNGenerator, self).__init__()
        assert (fixed_noise_level is not None) == (noise_type is NoiseType.FIXED)
        self.latent_size = input_size
        self.output_size = output_size
        self.num_gaussians = num_gaussians
        self.noise_type = noise_type
        self.fixed_noise_level = fixed_noise_level
        num_sigma_channels = {
            NoiseType.DIAGONAL: output_size * num_gaussians,
            NoiseType.ISOTROPIC: num_gaussians,
            NoiseType.ISOTROPIC_ACROSS_CLUSTERS: 1,
            NoiseType.FIXED: 0,
        }[noise_type]
        self.hidden = nn.Sequential(
            nn.Linear(input_size + condition_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.pi = nn.Linear(hidden_size, num_gaussians)
        self.normal_network = nn.Linear(hidden_size, output_size * num_gaussians + num_sigma_channels)

    def forward(self, x, condition, eps=1e-6):
        x = torch.cat([x, condition], dim=1)
        hidden = self.hidden(x)
        log_pi = torch.log_softmax(self.pi(hidden), dim=1)
        normal_params = self.normal_network(hidden)
        mu = normal_params[..., :self.num_gaussians * self.output_size]
        sigma = normal_params[..., self.num_gaussians * self.output_size:]
        if self.noise_type is NoiseType.DIAGONAL:
            sigma = torch.exp(sigma + eps)
        if self.noise_type is NoiseType.ISOTROPIC:
            sigma = torch.exp(sigma + eps).repeat(1, self.output_size)
        if self.noise_type is NoiseType.ISOTROPIC_ACROSS_CLUSTERS:
            sigma = torch.exp(sigma + eps).repeat(1, self.num_gaussians * self.output_size)
        if self.noise_type is NoiseType.FIXED:
            sigma = torch.full_like(mu, fill_value=self.fixed_noise_level)
        mu = mu.view(-1, self.num_gaussians, self.output_size)
        sigma = sigma.view(-1, self.num_gaussians, self.output_size)
        return log_pi, mu, sigma
    
    def sample(self, x, condition):
        log_pi, mu, sigma = self.forward(x, condition)
        cum_pi = torch.cumsum(torch.exp(log_pi), dim=-1)
        rvs = torch.rand(len(x), 1).to(x.device)
        rand_pi = torch.searchsorted(cum_pi, rvs)
        rand_normal = torch.randn_like(mu) * sigma + mu
        samples = torch.gather(rand_normal, 1, rand_pi.unsqueeze(-1).expand(-1, -1, mu.size(-1))).squeeze(1)
        return samples

def mdn_loss(log_pi, mu, sigma, y):
    z_score = (y.unsqueeze(1) - mu) / sigma
    normal_loglik = (
        -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
        - torch.sum(torch.log(sigma), dim=-1)
    )
    loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
    return -loglik.mean()

def train(model, dataloader, mdn_loss, optimizer, scheduler, num_epochs, writer, device='cuda'):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            position = batch['position'].float().to(device)
            angle = batch['angles'].float().to(device)

            # Draw noise samples
            z = torch.randn(position.size(0), model.latent_size).to(device)
            
            # Generate MDN outputs
            log_pi, mu, sigma = model(z, position)
            
            # Compute MDN loss
            loss = mdn_loss(log_pi, mu, sigma, angle)
            
            # Update generator
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

def main():
    parser = argparse.ArgumentParser(description='Train MDN model.')
    parser.add_argument('--data_file_path', type=str, required=True, help='Path to the data file')
    args = parser.parse_args()

    # Hyperparameters
    input_size = 0
    hidden_size = 64
    output_size = 2
    num_gaussians = 2
    condition_size = 2
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.001

    # Load dataset
    dataset = RobotDataset(args.data_file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, criterion and optimizer
    model = MDNGenerator(input_size, hidden_size, output_size, num_gaussians, condition_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Setup TensorBoard
    folder = os.path.join("logs",  f"mdn_model_{os.path.basename(args.data_file_path).split('.')[0]}") # use model and data name
    # clear the folder
    if os.path.exists(folder):
        shutil.rmtree(folder)
    writer = SummaryWriter(folder)

    # Train the model
    train(model, dataloader, mdn_loss, optimizer, scheduler, num_epochs, writer, device)

    # Save the model with a name that includes the data it trained on
    model_name = f"mdn_model_{os.path.basename(args.data_file_path).split('.')[0]}.pth"
    
    # Save the model under the folder in the logs directory
    model_path = os.path.join(folder, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")
    
    # Close the writer
    writer.close()

if __name__ == "__main__":
    main()