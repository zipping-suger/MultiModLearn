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

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)

def train(model, dataloader, criterion, optimizer, scheduler, num_epochs, writer):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            positions = batch['position']
            angles = batch['angles']
            
            outputs = model(positions)
            loss = criterion(outputs, angles)
            
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
    parser = argparse.ArgumentParser(description='Train MLP model.')
    parser.add_argument('--data_file_path', type=str, required=True, help='Path to the data file')
    args = parser.parse_args()

    # Hyperparameters
    input_size = 2
    hidden_size = 64
    output_size = 2
    num_epochs = 100
    batch_size = 32
    learning_rate = 0.001

    # Load dataset
    dataset = RobotDataset(args.data_file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, criterion and optimizer
    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Setup TensorBoard
    folder = os.path.join("logs",  f"mlp_model_{os.path.basename(args.data_file_path).split('.')[0]}") # use model and data name
    # clear the folder
    if os.path.exists(folder):
        shutil.rmtree(folder)
    writer = SummaryWriter(folder)

    # Train the model
    train(model, dataloader, criterion, optimizer, scheduler, num_epochs, writer)

    # Save the model with a name that includes the data it trained on
    model_name = f"mlp_model_{os.path.basename(args.data_file_path).split('.')[0]}.pth"
    
    # Save the model under the folder in the logs directory
    model_path = os.path.join(folder, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")
    
    # Close the writer
    writer.close()

if __name__ == "__main__":
    main()