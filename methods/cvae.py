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


class Encoder(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, c):
        x = torch.cat([x, c], dim=1)
        h = self.fc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z, c):
        z = torch.cat([z, c], dim=1)
        x_recon = self.fc(z)
        return x_recon

class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, condition_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, condition_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        # add clipping to logvar
        logvar = torch.clamp(logvar, -10, 10)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, c)
        return x_recon, mu, logvar


def loss_function(x_recon, x, mu, logvar):
    BCE = nn.functional.mse_loss(x_recon, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())/x.size(0)
    return BCE + KLD

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a CVAE model.')
    parser.add_argument('--data_file_path', type=str, required=True, help='Path to the data file.')
    args = parser.parse_args()

    # Hyperparameters
    input_dim = 2
    condition_dim = 2
    hidden_dim = 64
    latent_dim = 1
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-3

    # Initialize model, optimizer
    model = CVAE(input_dim, condition_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Load dataset
    dataset = RobotDataset(args.data_file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize SummaryWriter
    folder = os.path.join("logs", f"cvae_model_{os.path.basename(args.data_file_path).split('.')[0]}")
    if os.path.exists(folder):
        shutil.rmtree(folder)
    writer = SummaryWriter(folder)


    # Training loop
    for epoch in range(num_epochs):
         for i, batch in enumerate(dataloader):
            position = batch['position'].float()
            angles = batch['angles'].float()
            
            x_recon, mu, logvar = model(angles, position)
            loss = loss_function(x_recon, angles, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log the loss
            writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)

         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
         
    # Save the model with a name that includes the data it trained on
    model_name = f"cvae_model_{os.path.basename(args.data_file_path).split('.')[0]}.pth"
         
    # Save the model under the folder in the logs directory
    model_path = os.path.join(folder, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")
    
    writer.close()