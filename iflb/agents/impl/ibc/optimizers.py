import dataclasses

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

# =================================================================== #
# Model optimization.
# =================================================================== #


@dataclasses.dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    lr_scheduler_step: int = 100
    lr_scheduler_gamma: float = 0.99


# =================================================================== #
# Stochastic optimization for EBM training and inference.
# =================================================================== #


@dataclasses.dataclass
class StochasticOptimizerConfig:
    bounds: np.ndarray
    """Bounds on the samples, min/max for each dimension."""

    iters: int
    """The total number of inference iters."""

    train_samples: int
    """The number of counter-examples to sample per iter during training."""

    inference_samples: int
    """The number of candidates to sample per iter during inference."""

@dataclasses.dataclass
class RandomConfig(StochasticOptimizerConfig):
    iters: int = 1
    train_samples: int = 256
    inference_samples: int = 2 ** 14


@dataclasses.dataclass
class DerivativeFreeConfig(StochasticOptimizerConfig):
    noise_scale: float = 0.33
    noise_shrink: float = 0.5
    iters: int = 3
    train_samples: int = 256
    inference_samples: int = 2 ** 14

@dataclasses.dataclass
class LangevinConfig(StochasticOptimizerConfig):
    noise_scale: float = 0.33
    iters: int = 50
    train_samples: int = 128
    inference_samples: int = 512
    step_size_init: float = 1e-1
    step_size_final: float = 1e-5
    step_size_power: float = 2.0


@dataclasses.dataclass
class RandomOptimizer:
    device: torch.device
    train_samples: int
    inference_samples: int
    bounds: np.ndarray

    @staticmethod
    def initialize(
        config: RandomConfig, device_type: str
    ):
        return RandomOptimizer(
            device=torch.device(device_type if torch.cuda.is_available() else "cpu"),
            train_samples=config.train_samples,
            inference_samples=config.inference_samples,
            bounds=config.bounds,
        )

    def _sample(self, num_samples: int) -> torch.Tensor:
        """Helper method for drawing samples from the uniform random distribution."""
        size = (num_samples, self.bounds.shape[1])
        samples = np.random.uniform(self.bounds[0, :], self.bounds[1, :], size=size)
        return torch.as_tensor(samples, dtype=torch.float32, device=self.device)

    def sample(self, batch_size: int, ebm: nn.Module) -> torch.Tensor:
        del ebm  # The derivative-free optimizer does not use the ebm for sampling.
        samples = self._sample(batch_size * self.train_samples)
        return samples.reshape(batch_size, self.train_samples, -1)

    @torch.no_grad()
    def infer(self, x: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        samples = self._sample(x.size(0) * self.inference_samples)
        samples = samples.reshape(x.size(0), self.inference_samples, -1)

        energies = ebm(x, samples)
        best_idxs = energies.argmin(dim=-1)
        return samples[torch.arange(samples.size(0)), best_idxs, :]


@dataclasses.dataclass
class DerivativeFreeOptimizer:
    """A simple derivative-free optimizer. Great for up to 5 dimensions."""

    device: torch.device
    noise_scale: float
    noise_shrink: float
    iters: int
    train_samples: int
    inference_samples: int
    bounds: np.ndarray

    @staticmethod
    def initialize(
        config: DerivativeFreeConfig, device_type: str
    ):
        return DerivativeFreeOptimizer(
            device=torch.device(device_type if torch.cuda.is_available() else "cpu"),
            noise_scale=config.noise_scale,
            noise_shrink=config.noise_shrink,
            iters=config.iters,
            train_samples=config.train_samples,
            inference_samples=config.inference_samples,
            bounds=config.bounds,
        )

    def _sample(self, num_samples: int) -> torch.Tensor:
        """Helper method for drawing samples from the uniform random distribution."""
        size = (num_samples, self.bounds.shape[1])
        samples = np.random.uniform(self.bounds[0, :], self.bounds[1, :], size=size)
        return torch.as_tensor(samples, dtype=torch.float32, device=self.device)

    def sample(self, x: torch.Tensor, ebm: nn.Module, **kwargs) -> torch.Tensor:
        del ebm  # The derivative-free optimizer does not use the ebm for sampling.
        batch_size = x.size(0)
        del x
        samples = self._sample(batch_size * self.train_samples)
        return samples.reshape(batch_size, self.train_samples, -1)

    @torch.no_grad()
    def infer(self, x: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action given a trained EBM."""
        noise_scale = self.noise_scale
        bounds = torch.as_tensor(self.bounds).to(self.device)

        samples = self._sample(x.size(0) * self.inference_samples)
        samples = samples.reshape(x.size(0), self.inference_samples, -1)

        for i in range(self.iters):
            # Compute energies.
            energies = ebm(x, samples)
            probs = F.softmax(-1.0 * energies, dim=-1)

            # Resample with replacement.
            idxs = torch.multinomial(probs, self.inference_samples, replacement=True)
            samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]

            # Add noise and clip to target bounds.
            samples = samples + torch.randn_like(samples) * noise_scale
            samples = samples.clamp(min=bounds[0, :], max=bounds[1, :])

            noise_scale *= self.noise_shrink

        # Return target with highest probability.
        energies = ebm(x, samples)
        probs = F.softmax(-1.0 * energies, dim=-1)
        best_idxs = probs.argmax(dim=-1)
        return samples[torch.arange(samples.size(0)), best_idxs, :]


@dataclasses.dataclass
class LangevinOptimizer:

    device: torch.device
    noise_scale: float
    iters: int
    train_samples: int
    inference_samples: int
    bounds: np.ndarray
    step_size_init: float
    step_size_final: float
    step_size_power: float

    @staticmethod
    def initialize(
        config: LangevinConfig, device_type: str
    ):
        return LangevinOptimizer(
            device=torch.device(device_type if torch.cuda.is_available() else "cpu"),
            noise_scale=config.noise_scale,
            step_size_init=config.step_size_init,
            step_size_final=config.step_size_final,
            step_size_power=config.step_size_power,
            iters=config.iters,
            train_samples=config.train_samples,
            inference_samples=config.inference_samples,
            bounds=config.bounds,
        )

    def _sample(self, num_samples: int) -> torch.Tensor:
        """Helper method for drawing samples from the uniform random distribution."""
        size = (num_samples, self.bounds.shape[1])
        samples = np.random.uniform(self.bounds[0, :], self.bounds[1, :], size=size)
        return torch.as_tensor(samples, dtype=torch.float32, device=self.device)

    def _get_step_size(self, iteration):
        blend = iteration / (self.iters - 1)
        blend = blend ** self.step_size_power
        step_size = self.step_size_init + blend * (self.step_size_final - self.step_size_init)
        return step_size

    def langevin_step(self, x: torch.Tensor, y_init: torch.Tensor, ebm, iters=None):
        if iters is None:
            iters = self.iters
        bounds = torch.as_tensor(self.bounds).to(self.device)
        delta_y_clip = 0.1
        delta_y_clip = delta_y_clip * 0.5 * (bounds[1] - bounds[0])

        y = y_init
        for i in range(iters):
            y.requires_grad = True
            energies = ebm(x, y)    
            dE_dy = grad(energies.sum(), y)[0]

            stepsize = self._get_step_size(i)
            noise = torch.normal(0, self.noise_scale, size=dE_dy.shape, device=dE_dy.device)
            delta_y = - stepsize * dE_dy + np.sqrt(2 * stepsize) * noise
            delta_y = torch.clamp(delta_y, -delta_y_clip, delta_y_clip)

            y = y + delta_y
            y = torch.clamp(y, bounds[0, :], bounds[1, :])
            y = y.detach()
        return y

    def sample(self, x: torch.Tensor, ebm: nn.Module, uniform=False, inference=False, return_energies=False) -> torch.Tensor:
        num_samples = self.inference_samples if inference else self.train_samples
        batch_size = x.size(0)
        samples = self._sample(batch_size * num_samples).reshape(batch_size, num_samples, -1)
        if uniform:
            return samples
        
        samples = self.langevin_step(x, samples, ebm)
        # bounds = torch.as_tensor(self.bounds).to(self.device)
        # delta_action_clip = 0.1
        # delta_action_clip = delta_action_clip * 0.5 * (bounds[1] - bounds[0]) # torch.from_numpy((self.bounds[1] - self.bounds[0])).to(self.device)
        # for i in range(self.iters):
        #     # print(samples)
        #     samples.requires_grad = True
        #     energies = ebm(x, samples)
        #     dE_dy = grad(energies.sum(), samples)[0]

        #     stepsize = self._get_step_size(i)
        #     noise = torch.normal(0, self.noise_scale, size=dE_dy.shape, device=dE_dy.device)
        #     delta_y = - stepsize * dE_dy + np.sqrt(2 * stepsize) * noise
        #     delta_y = torch.clamp(delta_y, -delta_action_clip, delta_action_clip)
        #     # print(samples, dE_dy, delta_y)
        #     samples = samples + delta_y
        #     samples = torch.clamp(samples, bounds[0, :], bounds[1, :])
        #     samples = samples.detach()
        # # if return_grad:
        # #     samples.requires_grad = True
        # #     energies = ebm(x, samples)
        # #     dE_dy = grad(energies.sum(), samples, create_graph=True)[0]
        # #     return samples.detach(), dE_dy
        if return_energies:
            return samples, ebm(x, samples)
        return samples

    def infer(self, x: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action given a trained EBM."""
        noise_scale = self.noise_scale

        samples, energies = self.sample(x, ebm, inference=True, return_energies=True)
        with torch.no_grad():
            # Return target with highest probability.
            probs = F.softmax(-1.0 * energies, dim=-1)
            best_idxs = probs.argmax(dim=-1)
        return samples[torch.arange(samples.size(0)), best_idxs, :]
