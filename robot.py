import torch
import numpy as np
import matplotlib.pyplot as plt

class TwoLinkRobotIK:
    def __init__(self, link1_length, link2_length):
        """
        Initialize the 2-link robotic arm.

        Parameters:
            link1_length (float): Length of the first link.
            link2_length (float): Length of the second link.
        """
        self.link1_length = link1_length
        self.link2_length = link2_length
        
    def forward_kinematics(self, theta1, theta2):
        """
        Compute the end-effector position using forward kinematics.

        Parameters:
            theta1 (torch.Tensor): Joint angle 1 in radians.
            theta2 (torch.Tensor): Joint angle 2 in radians.

        Returns:
            torch.Tensor: End-effector position (x, y).
        """
        x = self.link1_length * torch.cos(theta1) + self.link2_length * torch.cos(theta1 + theta2)
        y = self.link1_length * torch.sin(theta1) + self.link2_length * torch.sin(theta1 + theta2)
        return torch.stack([x, y])
        
    def forward_kinematics_np(self, theta1, theta2):
        """
        Compute the end-effector position using forward kinematics.

        Parameters:
            theta1 (float): Joint angle 1 in radians.
            theta2 (float): Joint angle 2 in radians.

        Returns:
            np.ndarray: End-effector position (x, y).
        """
        x = self.link1_length * np.cos(theta1) + self.link2_length * np.cos(theta1 + theta2)
        y = self.link1_length * np.sin(theta1) + self.link2_length * np.sin(theta1 + theta2)
        return np.array([x, y])

    
    def forward_kinematics_batch(self, angles_batch):
        """
        Compute the end-effector positions for a batch of joint angles using forward kinematics.

        Parameters:
            angles_batch (torch.Tensor): Batch of joint angles with shape (N, 2), where N is the number of samples.

        Returns:
            torch.Tensor: Batch of end-effector positions with shape (N, 2).
        """
        theta1 = angles_batch[:, 0]
        theta2 = angles_batch[:, 1]
        x = self.link1_length * torch.cos(theta1) + self.link2_length * torch.cos(theta1 + theta2)
        y = self.link1_length * torch.sin(theta1) + self.link2_length * torch.sin(theta1 + theta2)
        return torch.stack([x, y], dim=1)

    def solve_ik_gradient_descent(self, target_position, seed = (0,0), learning_rate=0.1, iterations=1000, tolerance=1e-6):
        """
        Solve the inverse kinematics using gradient descent.

        Parameters:
            target_position (tuple): Target position (x, y).
            learning_rate (float): Learning rate for optimization.
            iterations (int): Number of iterations for gradient descent.
            tolerance (float): Tolerance for early stopping.

        Returns:
            tuple: Optimized joint angles (theta1, theta2) in radians.
        """
        # Initialize joint angles (theta1, theta2) as learnable parameters
        theta1 = torch.tensor(seed[0], dtype=torch.float32, requires_grad=True)
        theta2 = torch.tensor(seed[1], dtype=torch.float32, requires_grad=True)

        # Target position as a tensor
        target = torch.tensor(target_position)

        # Optimizer
        optimizer = torch.optim.Adam([theta1, theta2], lr=learning_rate)

        previous_loss = float('inf')

        for i in range(iterations):
            optimizer.zero_grad()

            # Compute forward kinematics
            end_effector = self.forward_kinematics(theta1, theta2)

            # Compute loss (Euclidean distance)
            loss = torch.nn.functional.mse_loss(end_effector, target)

            # Backpropagate
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_([theta1, theta2], max_norm=1.0)

            # Update angles
            optimizer.step()

            # Early stopping
            if abs(previous_loss - loss.item()) < tolerance:
                # print(f"Early stopping at iteration {i}, Loss: {loss.item():.6f}")
                break

            previous_loss = loss.item()

            # Print progress every 100 iterations
            if i % 100 == 0:
                # print(f"Iteration {i}, Loss: {loss.item():.6f}")
                pass

        return theta1.item(), theta2.item()

    def solve_ik_analytical(self, target_x, target_y):
        """
        Solve the inverse kinematics using an analytical method.

        Parameters:
            target_x (float): Target x-coordinate of the end effector.
            target_y (float): Target y-coordinate of the end effector.

        Returns:
            list of tuples: All possible solutions for (theta1, theta2) in radians.
        """
        solutions = []

        # Compute the distance to the target
        distance = np.sqrt(target_x**2 + target_y**2)

        # Check if the target is reachable
        if distance > (self.link1_length + self.link2_length) or distance < abs(self.link1_length - self.link2_length):
            # print("Target is unreachable.")
            return solutions

        # Compute theta2 (elbow angle)
        cos_theta2 = (distance**2 - self.link1_length**2 - self.link2_length**2) / (2 * self.link1_length * self.link2_length)
        theta2_options = [
            np.arccos(np.clip(cos_theta2, -1.0, 1.0)),  # Elbow down
            -np.arccos(np.clip(cos_theta2, -1.0, 1.0))  # Elbow up
        ]

        for theta2 in theta2_options:
            # Compute theta1 (shoulder angle)
            k1 = self.link1_length + self.link2_length * np.cos(theta2)
            k2 = self.link2_length * np.sin(theta2)
            theta1 = np.arctan2(target_y, target_x) - np.arctan2(k2, k1)

            # Store the solution
            solutions.append((theta1, theta2))

        return solutions
    
    def sample_from_workspace(self, num_samples, square = True):
        """
        Sample random positions from the workspace.

        Parameters:
            num_samples (int): Number of samples to generate.

        Returns:
            np.ndarray: Array of sampled positions (x, y).
        """
        samples = []
        x_range = (-self.link1_length - self.link2_length, self.link1_length + self.link2_length)
        y_range = (-self.link1_length - self.link2_length, self.link1_length + self.link2_length)
        
        for _ in range(num_samples):
            while True:
                target_x = np.random.uniform(*x_range)
                target_y = np.random.uniform(*y_range)
                distance = np.sqrt(target_x**2 + target_y**2)
                if square or (distance <= (self.link1_length + self.link2_length) and distance >= abs(self.link1_length - self.link2_length)):
                    samples.append((target_x, target_y))
                    break
        
        return np.array(samples)
    
    def evaluate(self, theta1, theta2, target_x, target_y):
        """
        Evaluate the error between the end effector position and the target position.

        Parameters:
            theta1 (float): Angle of the first link in radians.
            theta2 (float): Angle of the second link in radians.
            target_x (float): Target x-coordinate of the end effector.
            target_y (float): Target y-coordinate of the end effector.

        Returns:
            float: Euclidean distance between the end effector and the target position.
        """
        end_effector = self.forward_kinematics_np(theta1, theta2)
        target = np.array([target_x, target_y])
        return np.linalg.norm(end_effector - target)
    
    def evaluate_batch(self, angles_batch, target_positions):
        """
        Evaluate the error between the end effector positions and the target positions for a batch of samples.

        Parameters:
            angles_batch (torch.Tensor): Batch of joint angles with shape (N, 2), where N is the number of samples.
            target_positions (torch.Tensor): Batch of target positions with shape (N, 2).

        Returns:
            np.ndarray: Batch of Euclidean distances between the end effectors and the target positions.
        """
        
        # convert input to tensor if not already
        if not torch.is_tensor(angles_batch):
            angles_batch = torch.tensor(angles_batch, dtype=torch.float32)
        if not torch.is_tensor(target_positions):
            target_positions = torch.tensor(target_positions, dtype=torch.float32) 
        
        end_effectors = self.forward_kinematics_batch(angles_batch).detach().numpy()
        target_positions = target_positions.numpy()
        return np.linalg.norm(end_effectors - target_positions, axis=1)
    

    def plot(self, theta1, theta2):
        """
        Plot the 2-link robotic arm.

        Parameters:
            theta1 (float): Angle of the first link in radians.
            theta2 (float): Angle of the second link in radians.
        """
        # Compute joint positions
        joint1 = (self.link1_length * torch.cos(torch.tensor(theta1)),
                  self.link1_length * torch.sin(torch.tensor(theta1)))
        end_effector = (
            joint1[0] + self.link2_length * torch.cos(torch.tensor(theta1 + theta2)),
            joint1[1] + self.link2_length * torch.sin(torch.tensor(theta1 + theta2)),
        )

        # Plot the robot arm
        plt.figure(figsize=(6, 6))
        plt.plot([0, joint1[0], end_effector[0]], [0, joint1[1], end_effector[1]], '-o')
        plt.xlim(-self.link1_length - self.link2_length - 1, self.link1_length + self.link2_length + 1)
        plt.ylim(-self.link1_length - self.link2_length - 1, self.link1_length + self.link2_length + 1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid()
        plt.title("2-Link Robotic Arm")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()


# Example usage
if __name__ == "__main__":
    # Robot parameters
    L1 = 3.0  # Length of link 1
    L2 = 3.0  # Length of link 2

    # Target position
    target_x = 8.0
    target_y = 8.0

    # Create the robot
    robot = TwoLinkRobotIK(L1, L2)

    # Solve IK analytically
    solutions = robot.solve_ik_analytical(target_x, target_y)
    print(f"Analytical solutions:")
    for i, (theta1, theta2) in enumerate(solutions):
        print(f"Solution {i + 1}: Theta1 = {np.degrees(theta1):.2f}°, Theta2 = {np.degrees(theta2):.2f}°")
        robot.plot(theta1, theta2)

    # Solve IK using gradient descent
    theta1, theta2 = robot.solve_ik_gradient_descent((target_x, target_y))
    print(f"Gradient Descent Solution: Theta1 = {np.degrees(theta1):.2f}°, Theta2 = {np.degrees(theta2):.2f}°")
    robot.plot(theta1, theta2)
    
    
    # # Solve IK using gradient descent N times with different seed
    
    # N = 10
    # for i in range(N):
    #     theta1, theta2 = robot.solve_ik_gradient_descent((target_x, target_y), seed=(np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)))
    #     print(f"Gradient Descent Solution: Theta1 = {np.degrees(theta1):.2f}°, Theta2 = {np.degrees(theta2):.2f}°")
    #     robot.plot(theta1, theta2)
    
    
    
    
