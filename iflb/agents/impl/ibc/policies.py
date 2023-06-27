import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import numpy as np
from torch.autograd import grad

from . import models, optimizers

allowed_optimizers = {
    "dfo": optimizers.DerivativeFreeOptimizer,
    "random": optimizers.RandomOptimizer,
    "langevin": optimizers.LangevinOptimizer,
}


@dataclasses.dataclass
class ImplicitPolicy:
    """An implicit conditional EBM trained with an InfoNCE objective."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    stochastic_optimizer: optimizers.DerivativeFreeOptimizer
    device: torch.device
    steps: int
    gradient_penalty: bool

    @staticmethod
    def initialize(
        model_config: DictConfig,
        optim_config: optimizers.OptimizerConfig,
        stochastic_optim_config: optimizers.StochasticOptimizerConfig,
        device_type: str,
        cnn: bool,
        stochastic_optimizer_type: str,
        gradient_penalty: bool,
    ):
        device = torch.device(device_type if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if cnn:
            model = models.EBMConvMLP(config=model_config)
        else:
            model = models.EBM(config=model_config)
        model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optim_config.learning_rate,
            weight_decay=optim_config.weight_decay,
            betas=(optim_config.beta1, optim_config.beta2),
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=optim_config.lr_scheduler_step,
            gamma=optim_config.lr_scheduler_gamma,
        )

        assert stochastic_optimizer_type in allowed_optimizers
        stochastic_optimizer = allowed_optimizers[stochastic_optimizer_type].initialize(
            stochastic_optim_config,
            device_type,
        )

        return ImplicitPolicy(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            stochastic_optimizer=stochastic_optimizer,
            device=device,
            steps=0,
            gradient_penalty=gradient_penalty
        )

    def training_step(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> dict:
        self.model.train()
        # input = input.to(self.device) # 2d float
        # target = target.to(self.device) 

        # Generate N negatives, one for each element in the batch: (B, N, D).
        negatives = self.stochastic_optimizer.sample(input, self.model)

        # Merge target and negatives: (B, N+1, D).
        targets = torch.cat([target.unsqueeze(dim=1), negatives], dim=1)
        targets.requires_grad = self.gradient_penalty

        # Generate a random permutation of the positives and negatives.
        permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
        targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]

        # Get the original index of the positive. This will serve as the class label
        # for the loss.
        ground_truth = (permutation == 0).nonzero()[:, 1].to(self.device)

        # For every element in the mini-batch, there is 1 positive for which the EBM
        # should output a low energy value, and N negatives for which the EBM should
        # output high energy values.
        energy = self.model(input, targets)

        # Interpreting the energy as a negative logit, we can apply a cross entropy loss
        # to train the EBM.
        logits = -1.0 * energy
        loss = F.cross_entropy(logits, ground_truth)

        M = 1  # from IBC paper
        if self.gradient_penalty:
            negatives_idx = (permutation != 0).nonzero().to(self.device)
            dE_dy = grad(energy.sum(), targets, create_graph=True)[0][negatives_idx[:, 0], negatives_idx[:, 1]]  # select only the gradients of the sampled points
            grad_linf = torch.norm(dE_dy, p=torch.inf, dim=1)
            penalty = (torch.maximum(grad_linf - M, torch.zeros_like(grad_linf).to(self.device)) ** 2).sum()
            loss += penalty
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.steps += 1

        return {
            "loss": loss.item(),
            "lr": self.scheduler.get_last_lr()[0],
        }

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.stochastic_optimizer.infer(input.to(self.device), self.model) # returns the single best action

    def get_action(self, input: np.array) -> np.array:
        if len(input.shape) == 1:
            input = input[None]
        return self.predict(input).squeeze().cpu().numpy()

    def update_stats(self, *args):
        self.model.update_stats(*args)
