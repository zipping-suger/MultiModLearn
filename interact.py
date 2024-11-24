import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from robot import TwoLinkRobotIK
from mlp import MLP
from cgan import Generator
from cvae import CVAE  # Import the CVAE class

# Robot parameters
L1 = 3.0  # Length of link 1
L2 = 3.0  # Length of link 2

# Create the robot
robot = TwoLinkRobotIK(L1, L2)

# Load the trained MLP model
input_size = 2
hidden_size = 64
output_size = 2
model = MLP(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('logs/mlp_model_gradient_data_rs/mlp_model_gradient_data_rs.pth'))
model.eval()

# Load the trained CGAN model
latent_size = 2
hidden_size = 64
output_size = 2
condition_size = 2

generator = Generator(latent_size, hidden_size, output_size, condition_size)
generator.load_state_dict(torch.load('logs/cgan_model_gradient_data_rs/generator_gradient_data_rs.pth'))
generator.eval()

# Load the trained CVAE model
input_dim = 2
condition_dim = 2
hidden_dim = 64
latent_dim = 1

cvae = CVAE(input_dim, condition_dim, hidden_dim, latent_dim)
cvae.load_state_dict(torch.load('logs/cvae_model_gradient_data/cvae_model_gradient_data.pth'))
cvae.eval()

# Initialize plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-L1 - L2 - 1, L1 + L2 + 1)
ax.set_ylim(-L1 - L2 - 1, L1 + L2 + 1)
ax.set_aspect('equal', adjustable='box')
ax.grid()
plt.title("2-Link Robotic Arm")
plt.xlabel("X")
plt.ylabel("Y")

# Initialize robot plot
line, = ax.plot([], [], '-o')

def update_plot(theta1, theta2):
    joint1 = (L1 * np.cos(theta1), L1 * np.sin(theta1))
    end_effector = (joint1[0] + L2 * np.cos(theta1 + theta2), joint1[1] + L2 * np.sin(theta1 + theta2))
    line.set_data([0, joint1[0], end_effector[0]], [0, joint1[1], end_effector[1]])
    fig.canvas.draw()

def on_mouse_move(event):
    if event.inaxes != ax:
        return
    target_x, target_y = event.xdata, event.ydata
    target_position = (target_x, target_y)
    
    # convert to Float from Double
    target_position = [float(target_position[0]), float(target_position[1])]
    
    # Choose method
    method = 'cvae'  # Change this to 'gradient_descent', 'mlp', 'cgan', or 'cvae' to use different methods
    
    if method == 'gradient_descent':
        # Solve IK using gradient descent
        theta1, theta2 = robot.solve_ik_gradient_descent(target_position)
    elif method == 'mlp':
        # Solve IK using the neural network
        input_tensor = torch.tensor(target_position, dtype=torch.float32).unsqueeze(0)
        output_tensor = model(input_tensor)
        theta1, theta2 = output_tensor[0].detach().numpy()
    elif method == 'cgan':
        # Solve IK using the CGAN
        latent_vector = torch.randn(1, latent_size)
        condition = torch.tensor(target_position, dtype=torch.float32).unsqueeze(0)
        theta1, theta2 = generator(latent_vector, condition).detach().numpy()[0]
    elif method == 'cvae':
        # Solve IK using the CVAE
        condition = torch.tensor(target_position, dtype=torch.float32).unsqueeze(0)
        # Sample from the latent space
        latent_sample = torch.randn(1, latent_dim)
        theta1, theta2 = cvae.decoder(condition, latent_sample).detach().numpy()[0]
        
    
    update_plot(theta1, theta2)

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
plt.show()