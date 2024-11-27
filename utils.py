import torch
from methods.ibc import EnergyModel
from methods.cgan import Discriminator

# Derivative-free optimizer for inference
def ebm_infer(energy_model, target_position, y_min, y_max, samples=16384, iterations=3, sigma_init=0.33, scale=0.5):
    """
    Find angles that minimize energy for a given target position.

    Args:
        energy_model: Trained energy-based model.
        target_position: Target position (Cartesian coordinates), shape: (1, input_dim).
        y_min: Minimum values for joint angles (tensor).
        y_max: Maximum values for joint angles (tensor).
        samples: Number of random samples for initial exploration.
        iterations: Number of optimization iterations.
        sigma_init: Initial noise level for exploration.
        scale: Scaling factor for noise reduction.

    Returns:
        Optimal joint angles minimizing the energy.
    """
    device = target_position.device
    target_position = target_position.repeat(samples, 1)  # Repeat for batch inference
    sigma = sigma_init

    # Initialize random joint angle samples
    angles = torch.rand((samples, y_min.size(-1)), device=device) * (y_max - y_min) + y_min

    for _ in range(iterations):
        # Compute energies for current samples
        # If energy_model is type EnergyModel, you can use energy_model(target_position, angles)
        # elif energy_model is typr Discriminator, you can use -energy_model(angles, target_position)
        if isinstance(energy_model, EnergyModel):
            energies = energy_model(target_position, angles)
        elif isinstance(energy_model, Discriminator):
            energies = -energy_model(angles, target_position)
        else:
            raise ValueError("Invalid energy model type.")

        # Softmax over negative energies for sampling probabilities
        probabilities = torch.softmax(-energies, dim=0)

        # Resample based on probabilities
        indices = torch.multinomial(probabilities, num_samples=samples, replacement=True)
        angles = angles[indices]

        # Add noise for exploration
        angles += torch.randn_like(angles) * sigma
        angles = torch.clamp(angles, y_min, y_max)  # Clamp to valid joint angle bounds

        # Reduce noise scale
        sigma *= scale

    # Return the angles corresponding to the minimum energy
    best_idx = torch.argmin(energies)
    return angles[best_idx].unsqueeze(0)  # Shape: (1, action_dim)
