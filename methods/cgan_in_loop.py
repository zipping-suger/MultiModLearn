import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import time
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Assuming the classes Generator, Discriminator, and RobotDataset are defined in cgan.py
from cgan import Generator, Discriminator, train
from robot import TwoLinkRobotIK
from data_pipeline import RobotDataset, generate_data_gradient_descent

def generate_data_cgan(robot: TwoLinkRobotIK, generator, num_samples):
    """Generate seed using the generator model."""
    start_time = time.time()
    data = []    
    generator.eval()
    
    samples = robot.sample_from_workspace(num_samples)
    
    for target_x, target_y in tqdm(samples, desc="Generating data with cgan seed"):
            latent_vector = torch.randn(1, generator.latent_size)
            condition = torch.tensor([target_x, target_y], dtype=torch.float32).unsqueeze(0)
            seed =  generator(latent_vector, condition).detach().numpy()[0]
            theta1, theta2 = robot.solve_ik_gradient_descent((target_x, target_y), 
                                                             seed)
            data.append([target_x, target_y, theta1, theta2])
    
    end_time = time.time()
    print(f"Time taken to generate {num_samples} samples using gradient descent: {end_time - start_time:.2f} seconds")
    return np.array(data)


def main():
    # Hyperparameters
    latent_size = 2
    hidden_size = 64
    output_size = 2
    condition_size = 2
    num_epochs = 300
    batch_size = 100
    learning_rate = 0.0002
    initial_data_size = 2000
    generated_data_size = 2000
    iterations = 5
    
    replace_data = False # Set to True if you want to replace the initial data with the generated data
    # otherwise, the generated data will be appended to the initial data


    # Initialize robot
    robot = TwoLinkRobotIK(3.0, 3.0)

    # Collect initial data
    initial_data = generate_data_gradient_descent(robot, initial_data_size, fixed_seed=False)
    dataset = RobotDataset(initial_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    generator = Generator(latent_size, hidden_size, output_size, condition_size)
    discriminator = Discriminator(output_size, hidden_size, 1, condition_size)

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Initialize SummaryWriter
    folder = "logs/cgan_training"
    if replace_data:
        folder += "_replace"
    else:
        folder += "_append"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    writer = SummaryWriter(folder)

    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}/{iterations}")

        # Train CGAN
        train(dataloader, generator, discriminator, optimizer_g, optimizer_d, num_epochs, writer)
        
        # Save models under the logs folder
        torch.save(generator.state_dict(), os.path.join(folder, f"generator_{iteration}.pt"))
        torch.save(discriminator.state_dict(), os.path.join(folder, f"discriminator_{iteration}.pt"))
        
        # Generate new data
        generated_data = generate_data_cgan(robot, generator, generated_data_size)
        
        if replace_data:
            initial_data = generated_data
            dataset = RobotDataset(initial_data)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            continue
        else:
            # Combine old and new data
            combined_data = np.vstack((initial_data, generated_data))
            dataset = RobotDataset(combined_data)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Update initial data
            initial_data = combined_data

    writer.close()

if __name__ == "__main__":
    main()