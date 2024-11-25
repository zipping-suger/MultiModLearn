import torch
from torch.utils.data import Dataset
import numpy as np
from robot import TwoLinkRobotIK
import time
import os
from tqdm import tqdm
import scipy.spatial
import argparse

def generate_data_analytical(robot: TwoLinkRobotIK, num_samples: int):
    start_time = time.time()
    data = []
    
    samples = robot.sample_from_workspace(num_samples)
    
    for target_x, target_y in tqdm(samples, desc="Generating analytical data"):
        solutions = robot.solve_ik_analytical(target_x, target_y)
        for theta1, theta2 in solutions:
            data.append([target_x, target_y, theta1, theta2])
    
    end_time = time.time()
    print(f"Time taken to generate {num_samples} samples using analytical solutions: {end_time - start_time:.2f} seconds")
    return np.array(data)

def generate_data_gradient_descent(robot: TwoLinkRobotIK, num_samples: int, fixed_seed: bool = True):
    start_time = time.time()
    data = []
    
    samples = robot.sample_from_workspace(num_samples)
    
    for target_x, target_y in tqdm(samples, desc="Generating gradient descent data"):
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

def rrt_sample_workspace(robot: TwoLinkRobotIK, num_samples, max_iterations, step_size):
    """Generate an RRT graph structure in the workspace."""
    from collections import defaultdict

    graph = defaultdict(list)  # Adjacency list representation of the graph
    nodes = [(0, 0)]  # Start at the center of the workspace
    for _ in range(max_iterations):
        random_point = robot.sample_from_workspace(1)[0]
        nearest_idx = np.argmin([np.linalg.norm(np.array(random_point) - np.array(node)) for node in nodes])
        nearest_node = nodes[nearest_idx]
        direction = np.array(random_point) - np.array(nearest_node)
        direction = direction / np.linalg.norm(direction) * step_size
        new_point = tuple(np.array(nearest_node) + direction)

        # Check if the new point is valid
        if robot.is_valid_workspace_point(new_point):
            nodes.append(new_point)
            graph[nearest_node].append(new_point)
            graph[new_point].append(nearest_node)

        if len(nodes) >= num_samples:
            break

    return nodes, graph

def generate_data_incremental_sampling(robot: TwoLinkRobotIK, num_samples, max_iterations=1000, step_size=0.3):
    """Generate IK data using RRT sampling and incremental seeding."""
    start_time = time.time()
    data = []

    # Step 1: Generate RRT graph structure
    nodes, graph = rrt_sample_workspace(robot, num_samples, max_iterations, step_size)

    # Step 2: Solve IK incrementally, propagating solutions
    solved_seeds = {nodes[0]: (0, 0)}  # Start with the center seed
    visited = set()
    queue = [nodes[0]]

    with tqdm(total=len(nodes), desc="Generating incremental sampling data") as pbar:
        while queue:
            current_node = queue.pop(0)
            visited.add(current_node)

            current_seed = solved_seeds[current_node]
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    theta1, theta2 = robot.solve_ik_gradient_descent(neighbor, seed=current_seed)
                    data.append([*neighbor, theta1, theta2])
                    solved_seeds[neighbor] = (theta1, theta2)
                    queue.append(neighbor)
            pbar.update(1)

    print(f"Data generation completed in {time.time() - start_time:.2f} seconds.")
    return data
    

def filter_conflicts(data: np.ndarray, radius: float = 1, epsilon: float = 0.1) -> np.ndarray:
    positions = data[:, :2]
    angles = data[:, 2:]
    tree = scipy.spatial.cKDTree(positions)
    rejected = np.zeros(len(data), dtype=bool)
    metrics = np.zeros(len(data))
    
    for m in range(len(data)):
        neighbors = tree.query_ball_point(positions[m], radius)
        if len(neighbors) < 2:
            continue
        weights = 1.0 / np.maximum(
            np.linalg.norm(positions[neighbors] - positions[m], axis=1),
            1e-6
        )
        weights = weights / np.sum(weights)
        p_avg = np.average(positions[neighbors], weights=weights, axis=0)
        t_avg = np.average(angles[neighbors], weights=weights, axis=0)
        metrics[m] = np.linalg.norm(angles[m] - t_avg)
    
    valid_metrics = metrics[metrics > 0]
    if len(valid_metrics) == 0:
        return data
    
    metric_avg = np.mean(valid_metrics)
    
    for m in range(len(data)):
        if metrics[m] > metric_avg + epsilon:
            neighbors = tree.query_ball_point(positions[m], radius)
            rejected[neighbors] = True
    
    return data[~rejected]

def visualize_data(data):
    import matplotlib.pyplot as plt
    data = np.array(data)
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c='blue', label='Target Position')
    ax.set_aspect('equal')
    plt.show()

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
        
class RobotDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        target_x, target_y, theta1, theta2 = sample
        return {
            'position': torch.tensor([target_x, target_y], dtype=torch.float32),
            'angles': torch.tensor([theta1, theta2], dtype=torch.float32)
        }

def main():
    parser = argparse.ArgumentParser(description="Generate robot data using different methods")
    parser.add_argument('--method', type=str, choices=['analytical', 'gradient', 'incremental'], required=True, help="Method to generate data")
    parser.add_argument('--num_samples', type=int, default=5000, help="Number of samples to generate")
    parser.add_argument('--save_path', type=str, default="data/", help="Path to save the generated data")
    parser.add_argument('--fixed_seed', action='store_true', help="Use fixed seed for gradient descent method")
    args = parser.parse_args()

    L1 = 3.0
    L2 = 3.0
    robot = TwoLinkRobotIK(L1, L2)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.method == 'analytical':
        data = generate_data_analytical(robot, args.num_samples)
        np.save(f"{args.save_path}analytical_data.npy", data)
    elif args.method == 'gradient':
        data = generate_data_gradient_descent(robot, args.num_samples, fixed_seed=args.fixed_seed)
        np.save(f"{args.save_path}gradient_data.npy", data)
    elif args.method == 'incremental':
        data = generate_data_incremental_sampling(robot, args.num_samples)
        np.save(f"{args.save_path}incremental_data.npy", data)
        visualize_data(data)

if __name__ == "__main__":
    main()
