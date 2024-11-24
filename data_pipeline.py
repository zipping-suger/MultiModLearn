import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
from robot import TwoLinkRobotIK
import time
import scipy.spatial

def generate_data_analytical(robot: TwoLinkRobotIK, num_samples: int):
    start_time = time.time()
    data = []
    
    # sample the rectangular workspace
    x_range = (-robot.link1_length - robot.link2_length, robot.link1_length + robot.link2_length)
    y_range = (-robot.link1_length - robot.link2_length, robot.link1_length + robot.link2_length)
    
    for _ in range(num_samples):
        target_x = np.random.uniform(*x_range)
        target_y = np.random.uniform(*y_range)
        solutions = robot.solve_ik_analytical(target_x, target_y)
        for theta1, theta2 in solutions:
            data.append([target_x, target_y, theta1, theta2])
    
    end_time = time.time()
    print(f"Time taken to generate {num_samples} samples using analytical solutions: {end_time - start_time:.2f} seconds")
    return np.array(data)

def generate_data_gradient_descent(robot: TwoLinkRobotIK, num_samples: int, fixed_seed: bool = True):
    start_time = time.time()
    data = []
    
    # sample the rectangular workspace
    x_range = (-robot.link1_length - robot.link2_length, robot.link1_length + robot.link2_length)
    y_range = (-robot.link1_length - robot.link2_length, robot.link1_length + robot.link2_length)
    
    for _ in range(num_samples):
        target_x = np.random.uniform(*x_range)
        target_y = np.random.uniform(*y_range)
        if fixed_seed:
            theta1, theta2 = robot.solve_ik_gradient_descent((target_x, target_y))
        else:
            theta1, theta2 = robot.solve_ik_gradient_descent((target_x, target_y), 
                                                             seed=(np.random.uniform(-np.pi, np.pi), 
                                                                   np.random.uniform(-np.pi, np.pi)))
        
        data.append([target_x, target_y, theta1, theta2])
    
    end_time = time.time()
    print(f"Time taken to generate {num_samples} samples using gradient descent: {end_time - start_time:.2f} seconds")
    return np.array(data)

def filter_conflicts(data: np.ndarray, radius: float = 1, epsilon: float = 0.1) -> np.ndarray:
    """
    Detect and reject conflicting samples in the dataset based on local neighborhood interpolation.
    
    Args:
        data: numpy array of shape (N, 4) containing [x, y, theta1, theta2] for each sample
        radius: radius for finding neighbors
        epsilon: threshold for conflict detection
    
    Returns:
        filtered_data: numpy array containing non-conflicting samples
    """
    # Separate positions and angles
    positions = data[:, :2]  # x, y coordinates
    angles = data[:, 2:]    # theta1, theta2 angles
    
    # Build KD-tree for efficient neighbor search
    tree = scipy.spatial.cKDTree(positions)
    
    # Initialize array to track rejected samples
    rejected = np.zeros(len(data), dtype=bool)
    
    # Calculate metrics for each sample
    metrics = np.zeros(len(data))
    
    for m in range(len(data)):
        # Find neighbors within radius
        neighbors = tree.query_ball_point(positions[m], radius)
        
        if len(neighbors) < 2:  # Skip if not enough neighbors
            continue
            
        # Calculate interpolated position and target
        weights = 1.0 / np.maximum(
            np.linalg.norm(positions[neighbors] - positions[m], axis=1),
            1e-6
        )
        weights = weights / np.sum(weights)
        
        p_avg = np.average(positions[neighbors], weights=weights, axis=0)
        t_avg = np.average(angles[neighbors], weights=weights, axis=0)
        
        # Calculate metric (using L2 norm of the difference between actual and interpolated angles)
        metrics[m] = np.linalg.norm(angles[m] - t_avg)
    
    # Compute average metric
    valid_metrics = metrics[metrics > 0]
    if len(valid_metrics) == 0:
        return data
    
    metric_avg = np.mean(valid_metrics)
    
    # Reject samples based on metric threshold
    for m in range(len(data)):
        if metrics[m] > metric_avg + epsilon:
            # Reject sample and its neighbors
            neighbors = tree.query_ball_point(positions[m], radius)
            rejected[neighbors] = True
    
    # Return non-rejected samples
    return data[~rejected]



class RobotDataset(Dataset):
    def __init__(self, file_path):
        self.data = np.load(file_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        target_x, target_y, theta1, theta2 = sample
        return {
            'position': torch.tensor([target_x, target_y], dtype=torch.float32),
            'angles': torch.tensor([theta1, theta2], dtype=torch.float32)
        }

if __name__ == "__main__":
    # Robot parameters
    L1 = 3.0  # Length of link 1
    L2 = 3.0  # Length of link 2

    # Create the robot
    robot = TwoLinkRobotIK(L1, L2)

    # Data generation parameters
    num_samples = 2000
    
    save_path = "data/"

    # # Generate data using analytical solutions
    # analytical_data = generate_data_analytical(robot, num_samples)
    # print(f"Generated {len(analytical_data)} samples using analytical solutions.")

    # Generate data using gradient descent
    gradient_data = generate_data_gradient_descent(robot, num_samples, fixed_seed=False)
    print(f"Generated {len(gradient_data)} samples using gradient descent.")

    # Save data to files
    # np.save(f"{save_path}analytical_data.npy", analytical_data)
    np.save(f"{save_path}gradient_data_rs.npy", gradient_data)


    # # Test filtering function
    # filtered_data = filter_conflicts(analytical_data)
    # print(f"Filtered data shape: {filtered_data.shape}")
    # print(f"Original data shape: {analytical_data.shape}")
    # print(f"Number of rejected samples: {len(analytical_data) - len(filtered_data)}")
    
    # Test gradient descent with filtered data
    filtered_gradient_data = filter_conflicts(gradient_data)
    print(f"Filtered gradient data shape: {filtered_gradient_data.shape}")
    print(f"Original gradient data shape: {gradient_data.shape}")
    print(f"Number of rejected samples: {len(gradient_data) - len(filtered_gradient_data)}")
    
    # Save filtered data to files
    # np.save(f"{save_path}filtered_analytical_data.npy", filtered_data)
    np.save(f"{save_path}filtered_gradient_data_rs.npy", filtered_gradient_data)
