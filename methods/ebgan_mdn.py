import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import shutil
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import RobotDataset

from methods.ebgan import EnergyModel, info_nce_loss, generate_counter_samples
from methods.mdn import MDNGenerator, mdn_loss


# InfoNCE-style loss function with dynamic scaling
def info_nce_loss(energy_model, x, y, counter_samples, generator_samples, alpha):
    
    positive_energy = -energy_model(x, y)
    neg_energies = torch.stack([-energy_model(x, neg) for neg in counter_samples], dim=1)
    generator_energy = -energy_model(x, generator_samples)

    # Apply dynamic scaling to the generator term
    denominator = torch.logsumexp(
        torch.cat([positive_energy.unsqueeze(-1), neg_energies, generator_energy.unsqueeze(-1)], dim=-1),
        dim=-1
    )
    return torch.mean(denominator - positive_energy)


def dynamic_scaling(epoch, total_epochs, min_scale=0.1):
    return max(1 - epoch / total_epochs, min_scale)


# Training loop combining energy model with MDN generator
def train_ebgan_mdn(dataloader, energy_model, generator, optimizer_e, optimizer_g, 
                    scheduler_e, scheduler_g, num_epochs, writer, 
                    y_min, y_max, neg_count, repeat_energy_updates, device, 
                    alpha = 1, dynamic_scaling_true = False,
                    min_scale = 0.1):
    energy_model.train()
    generator.train()
    # Initialize lists to track losses
    energy_losses = []
    generator_e_losses = []
    mdn_losses = []
    total_g_losses = []

    for epoch in range(num_epochs):
        epoch_e_loss = 0.0
        epoch_g_loss_e = 0.0
        epoch_g_loss_mdn = 0.0
        epoch_g_loss = 0.0
        batch_counter = 0
        
        if dynamic_scaling_true:
            alpha = dynamic_scaling(epoch, num_epochs, min_scale)
        
        for batch_x, batch_y in dataloader:
            x_input = batch_x.to(device)
            y_target = batch_y.to(device)

            for _ in range(repeat_energy_updates):
                # Draw noise samples
                z = torch.randn(x_input.size(0), generator.latent_size).to(device)
            
                # Generate fake samples
                fake_y_target = generator.sample(z, x_input)
                
                counter_samples = generate_counter_samples(y_min, y_max, x_input.size(0), neg_count, device)
                
                # Compute loss
                e_loss = info_nce_loss(energy_model, x_input, y_target, counter_samples, fake_y_target, alpha)
                
                # Backpropagation
                optimizer_e.zero_grad()
                e_loss.backward()
                optimizer_e.step()
                
                epoch_e_loss += e_loss.item()

            # Compute energy 
            z = torch.randn(x_input.size(0), generator.latent_size).to(device)
            fake_y_target = generator.sample(z, x_input)
            g_loss_e = energy_model(x_input, fake_y_target).mean()

            # Compute MDN loss
            log_pi, mu, sigma = generator(z, x_input)
            mdn_g_loss = mdn_loss(log_pi, mu, sigma, y_target)

            epoch_g_loss_e += g_loss_e.item()
            epoch_g_loss_mdn += mdn_g_loss.item()
            
            # Combined loss
            g_loss = g_loss_e + mdn_g_loss
            
            epoch_g_loss += g_loss.item()

            # Backpropagation
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            # Log losses
            if writer:
                writer.add_scalar('Loss/EnergyModel', e_loss.item(), epoch * len(dataloader) + batch_counter)
                writer.add_scalar('Loss/GeneratorE', g_loss_e.item(), epoch * len(dataloader) + batch_counter)
                writer.add_scalar('Loss/MDN', mdn_g_loss.item(), epoch * len(dataloader) + batch_counter)
                writer.add_scalar('Loss/Generator', g_loss.item(), epoch * len(dataloader) + batch_counter)
            
            batch_counter += 1
            
        scheduler_e.step()
        scheduler_g.step()
        avg_e_loss = epoch_e_loss / len(dataloader) / repeat_energy_updates
        avg_g_loss_e = epoch_g_loss_e / len(dataloader)
        avg_g_loss_mdn = epoch_g_loss_mdn / len(dataloader)
        avg_g_loss = epoch_g_loss / len(dataloader)
        
        # Store losses for plotting
        energy_losses.append(avg_e_loss)
        generator_e_losses.append(avg_g_loss_e)
        mdn_losses.append(avg_g_loss_mdn)
        total_g_losses.append(avg_g_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
                f"Energy Loss: {avg_e_loss:.4f}, "
                f"Generator Loss: {avg_g_loss:.4f}, "
                f"Generator Energy Loss: {avg_g_loss_e:.4f}, "
                f"Generator MDN Loss: {avg_g_loss_mdn:.4f}"
                )
        
        
    # Plot the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), energy_losses, label='Energy Model Loss')
    plt.plot(range(1, num_epochs + 1), generator_e_losses, label='Generator Energy Loss')
    plt.plot(range(1, num_epochs + 1), mdn_losses, label='MDN Loss')
    plt.plot(range(1, num_epochs + 1), total_g_losses, label='Total Generator Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves for EBGAN-MDN Training')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train Energy-Based GAN model.')
    parser.add_argument('--data_file_path', type=str, default='data/gradient_data_rs.npy', help='Path to the dataset')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for tensorboard logs')
    args = parser.parse_args()

    # Hyperparameters
    action_size = 2
    condition_size = 2
    hidden_size = 64
    latent_size = 0
    output_size = 2
    
    num_epochs = 100
    batch_size = 32
    neg_count = 256
    repeat_energy_updates = 5
    
    learning_rate_e = 0.001
    learning_rate_g = 0.001
    
    alpha = 1
    dynamic_scaling_true = False
    min_scale = 0.1
    
    num_gaussians = 10   # Number of gaussians for MDN
    
    # Data bounds for counter-sample generation
    y_min = torch.tensor([-3.14, -3.14])  # Joint angle limits (radians)
    y_max = torch.tensor([3.14, 3.14])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create models
    generator = MDNGenerator(latent_size, hidden_size, output_size, num_gaussians, condition_size).to(device)
    energy_model = EnergyModel(condition_size, action_size, hidden_size).to(device)
    
    # Setup optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate_g)
    optimizer_e = optim.Adam(energy_model.parameters(), lr=learning_rate_e)
    scheduler_e = optim.lr_scheduler.StepLR(optimizer_e, step_size=20, gamma=0.5)
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=20, gamma=0.5)
    
    # Setup tensorboard
    log_dir = os.path.join(args.log_dir, 'ebgan-mdn_training')
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)
    
    # Load dataset
    dataset = RobotDataset(args.data_file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train
    train_ebgan_mdn(dataloader, energy_model, generator, optimizer_e, optimizer_g, 
                    scheduler_e, scheduler_g, num_epochs, writer, 
                    y_min, y_max, neg_count, repeat_energy_updates, device,
                    alpha, dynamic_scaling_true, min_scale 
    )
    
    # Save models
    torch.save(generator.state_dict(), os.path.join(log_dir, 'mdn_generator.pth'))
    torch.save(energy_model.state_dict(), os.path.join(log_dir, 'energy_model.pth'))
    
    writer.close()

if __name__ == "__main__":
    main()