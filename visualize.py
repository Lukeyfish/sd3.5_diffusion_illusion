import matplotlib.pyplot as plt
import torch
import math

class Sampling:
    def __init__(self, sigma_min=0.01, sigma_max=1.0):
        self.sigma_min = sigma_min  # Minimum noise level
        self.sigma_max = sigma_max  # Maximum noise level

    def sigma(self, t):
        """Maps timestep t to sigma."""
        # Example: Exponential decay (adjust as needed for your model)
        return self.sigma_max * (self.sigma_min / self.sigma_max) ** t

    def timestep(self, sigma):
        """Maps sigma to timestep t."""
        # Ensure sigma and self.sigma_max are tensors
        sigma = torch.tensor(sigma, dtype=torch.float32)
        sigma_max = torch.tensor(self.sigma_max, dtype=torch.float32)
        sigma_min = torch.tensor(self.sigma_min, dtype=torch.float32)

        return torch.log(sigma / sigma_max) / torch.log(sigma_min / sigma_max)
    

def get_sigmas(sampling, steps):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(sampling.sigma(ts))
        sigs += [0.0]
        return torch.FloatTensor(sigs)
    
def get_sigmas_quadratic(sampling, steps):
    start = sampling.timestep(sampling.sigma_max)
    end = sampling.timestep(sampling.sigma_min)
    
    # Quadratic spacing for slower start
    timesteps = torch.linspace(0, 1, steps) ** 2  # Squaring for quadratic scale
    timesteps = start + timesteps * (end - start)  # Scale to [start, end]
    
    sigs = [sampling.sigma(ts) for ts in timesteps]
    sigs.append(0.0)  # Append final sigma
    return torch.FloatTensor(sigs)

def get_sigmas_logarithmic(sampling, steps):
    start = sampling.timestep(sampling.sigma_max)
    end = sampling.timestep(sampling.sigma_min)
    
    # Logarithmic spacing for slower start
    timesteps = torch.logspace(0, 1, steps, base=10.0) - 1
    timesteps = timesteps / timesteps.max()  # Normalize to [0, 1]
    timesteps = start + timesteps * (end - start)  # Scale to [start, end]
    
    sigs = [sampling.sigma(ts) for ts in timesteps]
    sigs.append(0.0)  # Append final sigma
    return torch.FloatTensor(sigs)

def get_sigmas_cosine(sampling, steps):
    beta_start = sampling.sigma_max
    beta_end = sampling.sigma_min
    T = steps - 1  # Total number of steps minus one for correct scaling

    # Cosine schedule formula
    timesteps = torch.arange(steps, dtype=torch.float32)  # Steps: 0, 1, ..., T
    beta_t = beta_end + 0.5 * (beta_start - beta_end) * (1 + torch.cos(math.pi * timesteps / T))

    # Convert beta_t to sigmas (assuming beta maps directly to noise levels)
    sigs = [sampling.sigma(t) for t in beta_t]
    sigs.append(0.0)  # Append final sigma
    return torch.FloatTensor(sigs)

def get_sigmas_custom(sampling, steps, alpha=0.5):
    start = sampling.timestep(sampling.sigma_max)
    end = sampling.timestep(sampling.sigma_min)
    
    # Mix linear and quadratic schedules with weight `alpha`
    linear_timesteps = torch.linspace(start, end, steps)
    quadratic_timesteps = torch.linspace(0, 1, steps) ** 2
    quadratic_timesteps = start + quadratic_timesteps * (end - start)
    timesteps = alpha * linear_timesteps + (1 - alpha) * quadratic_timesteps
    
    sigs = [sampling.sigma(ts) for ts in timesteps]
    sigs.append(0.0)  # Append final sigma
    return torch.FloatTensor(sigs)


sampling = Sampling(sigma_min=0.01, sigma_max=1.0)

steps = 50


linear_sigmas = get_sigmas(sampling, steps)
quadratic_sigmas = get_sigmas_quadratic(sampling, steps)
logarithmis_sigmas = get_sigmas_logarithmic(sampling, steps)
cosine_sigmas = get_sigmas_cosine(sampling, steps)
custom_sigmas = get_sigmas_custom(sampling, steps, alpha=0.1)

plt.plot(linear_sigmas, label="Linear")
plt.plot(quadratic_sigmas, label="Quadratic")
plt.plot(cosine_sigmas, label="Cosine")
plt.plot(logarithmis_sigmas, label="Logarithmic")
plt.plot(custom_sigmas, label="Custom")
plt.legend()
plt.xlabel("Step")
plt.ylabel("Sigma")
plt.title("Noise Schedules")
plt.savefig("SAMPLERS")