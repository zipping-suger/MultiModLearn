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

from methods.ebgan import EnergyModel, info_nce_loss, generate_counter_samples
from methods.mdn import MDNGenerator, mdn_loss


# Training loop combining energy model with MDN generator

def train_ebgan_mdn(dataloader, energy_model, generator, optimizer_e, optimizer_g, 
                    scheduler_e, scheduler_g, num_epochs, writer, 
                    y_min, y_max, neg_count, repeat_energy_updates, device):
    energy_model.train()
    generator.train()
    for epoch in range(num_epochs):
        epoch_e_loss = 0.0
        epoch_g_loss_e = 0.0
        epoch_g_loss_mdn = 0.0
        for i, batch in enumerate(dataloader):
            x_input = batch['position'].float().to(device)
            y_target = batch['angles'].float().to(device)
            
            for _ in range(repeat_energy_updates):
                # Draw noise samples
                z = torch.randn(x_input.size(0), generator.latent_size).to(device)
            
                # Generate fake samples
                fake_y_target = generator.sample(z, x_input)
                
                counter_samples = generate_counter_samples(y_min, y_max, x_input.size(0), neg_count, device)
                
                # Append the fake samples to the counter samples
                counter_samples.append(fake_y_target)
                
                # Compute loss
                e_loss = info_nce_loss(energy_model, x_input, y_target, counter_samples)
                
                # Backpropagation
                optimizer_e.zero_grad()
                e_loss.backward()
                optimizer_e.step()
                
                epoch_e_loss += e_loss.item()
                
            # Update generator
            
            # Compute energy 
            z = torch.randn(x_input.size(0), generator.latent_size).to(device)
            fake_y_target = generator.sample(z, x_input)
            g_loss_e = energy_model(x_input, fake_y_target).mean()
            
            # Compute MDN loss
            log_pi, mu, sigma = generator(z, x_input)
            mdn_g_loss = mdn_loss(log_pi, mu, sigma, y_target)
            
            epoch_g_loss_e += g_loss_e.item()
            
            # Combined loss
            g_loss = g_loss_e + mdn_g_loss
            
            epoch_g_loss_mdn += mdn_g_loss.item()
            
            # Backpropagation
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()
            
        
            # Log losses
            writer.add_scalar('Loss/EnergyModel', e_loss.item(), epoch * len(dataloader) + i)
            writer.add_scalar('Loss/Generator', g_loss_e.item(), epoch * len(dataloader) + i)
            writer.add_scalar('Loss/MDN', mdn_g_loss.item(), epoch * len(dataloader) + i)

        scheduler_e.step()
        scheduler_g.step()
        avg_e_loss = epoch_e_loss / len(dataloader) / repeat_energy_updates
        avg_g_loss_e = epoch_g_loss_e / len(dataloader)
        avg_g_loss_mdn = epoch_g_loss_mdn / len(dataloader)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
                f"Energy Loss: {avg_e_loss:.4f}, "
                f"Generator Energy Loss: {avg_g_loss_e:.4f}, "
                f"Generator MDN Loss: {avg_g_loss_mdn:.4f}")

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
    output_size = 2
    
    num_epochs = 100
    batch_size = 32
    neg_count = 256
    repeat_energy_updates = 10
    
    learning_rate_e = 0.001
    learning_rate_g = 0.001
    
    num_gaussians = 5   # Number of gaussians for MDN
    
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
                    y_min, y_max, neg_count, repeat_energy_updates, device)
    
    # Save models
    torch.save(generator.state_dict(), os.path.join(log_dir, 'mdn_generator.pth'))
    torch.save(energy_model.state_dict(), os.path.join(log_dir, 'energy_model.pth'))
    
    writer.close()

if __name__ == "__main__":
    main()