import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import shutil
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot import TwoLinkRobotIK
from methods.mlp import MLP

def train_ik_with_differentiable_kinematics(model, robot: TwoLinkRobotIK, sample_size, batch_size, iteration, criterion, optimizer, writer, device='cuda'):
    model.train()
    for i in range(sample_size):
        # sample a batch of target positions from the workspace
        target_positions = robot.sample_from_workspace(batch_size)
        # move target positions to the same device as the model
        target_positions = torch.FloatTensor(target_positions).to(device)
        
        for j in range(iteration):
            # use the model to predict the joint angles
            predicted_angles = model(target_positions)
            
            # use the forward kinematics to compute the end effector positions
            predicted_positions = robot.forward_kinematics_batch(predicted_angles)
                
            # compute the loss between the predicted and target positions
            loss = criterion(predicted_positions, target_positions)
            
            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if j % 10 == 0:
                # log the loss
                writer.add_scalar('Loss/train', loss.item(), i * iteration + j)
                print(f'Iteration [{i+1}/{sample_size}], Loss: {loss.item():.4f}')
            
        
        # log the loss
        writer.add_scalar('Loss/train', loss.item(), i)
        print(f'Iteration [{i+1}/{sample_size}], Loss: {loss.item():.4f}')
        


def main():
    # Hyperparameters
    input_size = 2
    hidden_size = 64
    output_size = 2
    sample_size = 200
    iteration = 50
    batch_size = 32
    learning_rate = 0.001

    # Robot parameters
    L1 = 3.0  # Length of link 1
    L2 = 3.0  # Length of link 2

    # Create the robot
    robot = TwoLinkRobotIK(L1, L2)

    # Initialize model, criterion and optimizer
    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Check if GPU is available, and use it if it is
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)

    # Setup TensorBoard
    folder = os.path.join("logs",  f"mlp_model_direct_differentiable") # use model and data name
    # clear the folder
    if os.path.exists(folder):
        shutil.rmtree(folder)
    writer = SummaryWriter(folder)

    # Train the model
    train_ik_with_differentiable_kinematics(model, robot, sample_size, batch_size, iteration, criterion, optimizer, writer, device=device)
    
    # Save the model with a name that includes the data it trained on
    model_name = f"mlp_model_direct_differentiable.pth"
    
    # Save the model under the folder in the logs directory
    model_path = os.path.join(folder, model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")
    
    # Close the writer
    writer.close()

if __name__ == "__main__":
    main()