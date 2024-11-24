import torch
import numpy as np
import matplotlib.pyplot as plt
from robot import TwoLinkRobotIK
from methods.mlp import MLP
from methods.cgan import Generator
from methods.cvae import CVAE 
from methods.reinforce import PolicyNetwork
from methods.ibc import EnergyModel, infer_angles
import argparse

# Robot parameters
L1 = 3.0  # Length of link 1
L2 = 3.0  # Length of link 2

# Create the robot
robot = TwoLinkRobotIK(L1, L2)

# Hyperparameters
input_size = 2
hidden_size = 64
output_size = 2
latent_size = 2

# cgan and cvae 
condition_size = 2

# cvae
input_dim = 2
latent_dim = 1

def load_model(model_type):
    if model_type == 'mlp':
        model = MLP(input_size, hidden_size, output_size)
        # model.load_state_dict(torch.load('logs/mlp_model_gradient_data/mlp_model_model_gradient_data.pth', weights_only=True))
        # model.load_state_dict(torch.load('logs/mlp_model_gradient_data_rs/mlp_model_gradient_data_rs.pth', weights_only=True))
        model.load_state_dict(torch.load('logs/mlp_model_direct_differentiable/mlp_model_direct_differentiable.pth', weights_only=True))
        model.eval()
        return model
    elif model_type == 'cgan':
        generator = Generator(latent_size, hidden_size, output_size, condition_size)
        # generator.load_state_dict(torch.load('logs/cgan_model_gradient_data/cgan_model_gradient_data.pth', weights_only=True))
        generator.load_state_dict(torch.load('logs/cgan_model_gradient_data_rs/generator_gradient_data_rs.pth', weights_only=True))
        generator.eval()
        return generator
    elif model_type == 'cvae':
        cvae = CVAE(input_dim, condition_size, hidden_size, latent_dim)
        # cvae.load_state_dict(torch.load('logs/cvae_model_gradient_data/cvae_model_gradient_data.pth', weights_only=True))
        cvae.load_state_dict(torch.load('logs/cvae_model_gradient_data_rs/cvae_model_gradient_data_rs.pth', weights_only=True))
        cvae.eval()
        return cvae
    elif model_type == 'reinforce':
        reinforce_policy_net = PolicyNetwork(input_size, hidden_size, output_size)
        reinforce_policy_net.load_state_dict(torch.load('logs/REINFORCE/reinforce_model.pth', weights_only=True))
        reinforce_policy_net.eval()
        return reinforce_policy_net
    elif model_type == 'ibc':
        energy_model = EnergyModel(input_size=input_size, action_size=output_size, hidden_size=hidden_size)
        # energy_model.load_state_dict(torch.load('logs/ibc_model_gradient_data/ibc_model_gradient_data.pth', weights_only=True))
        energy_model.load_state_dict(torch.load('logs/ibc_model_gradient_data_rs/ibc_model_gradient_data_rs.pth', weights_only=True))
        energy_model.eval()
        energy_model.to('cuda')
        return energy_model

def ik_methods(method, target_position):
    if method == 'gradient_descent':
        target_position = [float(target_position[0]), float(target_position[1])]
        return robot.solve_ik_gradient_descent(target_position)
    elif method == 'mlp':
        input_tensor = torch.tensor(target_position, dtype=torch.float32).unsqueeze(0)
        output_tensor = mlp_model(input_tensor)
        return output_tensor[0].detach().numpy()
    elif method == 'cgan':
        latent_vector = torch.randn(1, latent_size)
        condition = torch.tensor(target_position, dtype=torch.float32).unsqueeze(0)
        return cgan_model(latent_vector, condition).detach().numpy()[0]
    elif method == 'cvae':
        condition = torch.tensor(target_position, dtype=torch.float32).unsqueeze(0)
        latent_sample = torch.randn(1, latent_dim)
        return cvae_model.decoder(latent_sample, condition).detach().numpy()[0]
    elif method == 'reinforce':
        input_tensor = torch.tensor(target_position, dtype=torch.float32).unsqueeze(0)
        output_tensor, _ = reinforce_model(input_tensor)
        return output_tensor[0].detach().numpy()
    elif method == 'ibc':
        target_tensor = torch.tensor(target_position, dtype=torch.float32).unsqueeze(0).to('cuda')
        y_min = torch.tensor([-3.14, -3.14], device='cuda')
        y_max = torch.tensor([3.14, 3.14], device='cuda')
        inferred_angles = infer_angles(energy_model, target_tensor, y_min, y_max)
        return inferred_angles.cpu().numpy()[0]

def main(method_type):
    global mlp_model, cgan_model, cvae_model, reinforce_model, energy_model

    mlp_model = load_model('mlp')
    cgan_model = load_model('cgan')
    cvae_model = load_model('cvae')
    reinforce_model = load_model('reinforce')
    energy_model = load_model('ibc')

    # Initialize plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-L1 - L2 - 1, L1 + L2 + 1)
    ax.set_ylim(-L1 - L2 - 1, L1 + L2 + 1)
    ax.set_aspect('equal', adjustable='box')
    ax.grid()
    plt.title("2-Link Robotic Arm")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Create a grid of target positions
    x_range = np.linspace(-L1 - L2, L1 + L2, 100)
    y_range = np.linspace(-L1 - L2, L1 + L2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)

    if method_type != 'gradient_descent':  # For all methods except gradient descent
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                target_position = (X[i, j], Y[i, j])
                theta1, theta2 = ik_methods(method_type, target_position)
                Z[i, j] = robot.evaluate(theta1, theta2, target_position[0], target_position[1])

        heatmap = ax.imshow(Z, extent=(-L1 - L2, L1 + L2, -L1 - L2, L1 + L2), origin='lower', cmap='coolwarm', alpha=0.6)

    line, = ax.plot([], [], '-o')

    def update_plot(theta1, theta2):
        joint1 = (L1 * np.cos(theta1), L1 * np.sin(theta1))
        end_effector = (joint1[0] + L2 * np.cos(theta1 + theta2), joint1[1] + L2 * np.sin(theta1 + theta2))
        line.set_data([0, joint1[0], end_effector[0]], [0, joint1[1], end_effector[1]])
        line.set_color('black')
        line.set_linewidth(2.5)
        fig.canvas.draw()

    def on_mouse_move(event):
        if event.inaxes != ax:
            return
        target_x, target_y = event.xdata, event.ydata
        target_position = (target_x, target_y)
        target_position = [float(target_position[0]), float(target_position[1])]
        theta1, theta2 = ik_methods(method_type, target_position)
        update_plot(theta1, theta2)

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='2-Link Robotic Arm Inverse Kinematics')
    parser.add_argument('--method', type=str, default='gradient_descent', choices=['gradient_descent', 'mlp', 'cgan', 'cvae', 'reinforce', 'ibc'], help='IK method to use')
    args = parser.parse_args()
    print(f"Using {args.method} method for inverse kinematics")
    main(args.method)
