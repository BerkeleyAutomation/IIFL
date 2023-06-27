import numpy as np
import torch
import os.path as osp

from . import policies
from . import optimizers
from . import models

torchify = lambda x, device: torch.FloatTensor(x).to(device)

class IBC(object):
    def __init__(self,
                 observation_space,
                 action_space,
                 args,
                 logdir):
        self.env_name = args.env_name
        self.logdir = logdir
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.cnn = args.cnn
        self.num_policies = args.agent_cfg.num_policies
        assert self.num_policies >= 1 and self.num_policies <= 2 # only 1 or 2 implicit policies
        self.log_freq = args.log_freq

        self.observation_space = observation_space
        self.action_space = action_space
        self.args = args
        self.init_policies()

    def init_policies(self):
        args = self.args
        observation_space = self.observation_space
        action_space = self.action_space
        assert self.num_policies >= 1 and self.num_policies <= 2
        target_bounds = np.array([action_space.low, action_space.high])
        agent_args = args.agent_cfg
        if agent_args.stochastic_optimizer_type == "dfo":
                stochastic_optim_config = optimizers.DerivativeFreeConfig(
                    bounds=target_bounds,
                    train_samples=agent_args.stochastic_optimizer_train_samples,
                )
        elif agent_args.stochastic_optimizer_type == "langevin":
            stochastic_optim_config = optimizers.LangevinConfig(
                bounds=target_bounds,
                train_samples=agent_args.stochastic_optimizer_train_samples,
                inference_samples = agent_args.stochastic_optimizer_inference_samples,
            )
        
        optim_config = optimizers.OptimizerConfig(learning_rate=agent_args.lr)
        if args.cnn:
            mlp_config = models.MLPConfig(
                input_dim=32 + action_space.shape[0], # 32 = (16 filters in last layer of ConvNet) * (2 coordinates for spatial softmax)
                hidden_dim=agent_args.hidden_size,
                hidden_layers=agent_args.hidden_layers,
                spectral_norm=agent_args.spectral_norm
            )
            model_config = models.ConvMLPConfig(
                cnn_config=models.CNNConfig(),
                mlp_config=mlp_config
            ) # TODO: coord_conv and spatial_reduction
        else: 
            model_config = models.MLPConfig(
                input_dim=observation_space.shape[0] + action_space.shape[0],
                hidden_dim=agent_args.hidden_size,
                hidden_layers=agent_args.hidden_layers,
                spectral_norm=agent_args.spectral_norm,
                normalize_inputs=agent_args.normalize_inputs,
            )
        self.policy = [policies.ImplicitPolicy.initialize(
            model_config=model_config,
            optim_config=optim_config,
            stochastic_optim_config=stochastic_optim_config,
            device_type=self.device,
            cnn=args.cnn,
            stochastic_optimizer_type=agent_args.stochastic_optimizer_type,
            gradient_penalty=agent_args.gradient_penalty,
        ) for _ in range(self.num_policies)]

    def update_stats(self, memory):
        states = np.array([t[0] for t in memory.buffer])
        actions = np.array([t[1] for t in memory.buffer])
        states_mean = torchify(np.mean(states, axis=0), self.device)
        states_std = torchify(np.std(states, axis=0), self.device)
        actions_mean = torchify(np.mean(actions, axis=0), self.device)
        actions_std = torchify(np.std(actions, axis=0), self.device)

        for policy in self.policy:
            policy.update_stats(states_mean, states_std, actions_mean, actions_std)

    def train(self, memory, batch_size):
        for i, policy in enumerate(self.policy):
            state_batch, action_batch, _, _, _ = memory[i].sample(
                batch_size=batch_size)
            state_batch = torchify(state_batch, self.device)
            action_batch = torchify(action_batch, self.device)

            log = policy.training_step(state_batch, action_batch)
        return log

    def save(self, logdir=None):
        if logdir == None:
            logdir = self.logdir
        torch.save(self.policy, osp.join(logdir, "policy.ckpt"))
    
    def load(self, logdir):
        self.policy = torch.load(osp.join(logdir, "policy.ckpt"))

    def get_actions(self, states):
        '''
        Get action from current task policy
        '''
        policy = self.policy[0]
        return policy.get_action(states)

    def get_policy_uncertainty(self, states):
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        if self.num_policies == 1:
            return np.zeros(states.shape[0])
        else:
            # Estimate Jeffreys Divergence of actors at state
            samples_0 = self.policy[0].stochastic_optimizer.sample(states, self.policy[0].model, inference=True)
            samples_1 = self.policy[1].stochastic_optimizer.sample(states, self.policy[1].model, inference=True)

            J = torch.mean(self.policy[1].model(states, samples_0) - self.policy[0].model(states, samples_0), dim=1) + \
            torch.mean(self.policy[0].model(states, samples_1) - self.policy[1].model(states, samples_1), dim=1)

            return J.detach().cpu().numpy()
