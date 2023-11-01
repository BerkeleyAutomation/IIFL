"""
Supervisor for Isaac Gym environments that does a forward pass on a trained RL agent. 
Assumes the agent is trained with IsaacGymEnvs (https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
"""
import numpy as np
import torch
from .base_supervisor import ParallelSupervisor
from rl_games.torch_runner import Runner
from gym import spaces

class IsaacGymRLSupervisor(ParallelSupervisor):
    def __init__(self, envs, cfg):
        assert cfg.vec_env, "Must be Isaac Gym env to use this supervisor"
        icfg = cfg.isaacgym_cfg
        num_envs = icfg['task']['env']['numEnvs']
        obs_dim = icfg['task']['env']['numObservations']
        act_dim = icfg['task']['env']['numActions']
        icfg = icfg.copy()
        icfg['train']['params']['config']['env_info'] = {
                'action_space': spaces.Box(np.ones(act_dim) * -1., np.ones(act_dim) * 1.),
                'observation_space': spaces.Box(np.ones(obs_dim) * -np.Inf, np.ones(obs_dim) * np.Inf),
                'agents': num_envs,
                'batch': True
            }
        self.num_players = len(icfg['checkpoints'])
        # setup RL games runner
        players = []
        for i in range(self.num_players):
            icfg_temp = icfg.copy()
            icfg_temp['train']['params']['load_path'] = icfg['checkpoints'][i]
            icfg_temp['train']['params']['load_checkpoint'] = True if icfg['checkpoints'][i] else False
            rlg_config_dict = icfg_temp['train']
            runner = Runner()
            runner.load(rlg_config_dict)
            runner.reset()
            player = runner.create_player()
            player.restore(icfg['checkpoints'][i])
            player.has_batch_dimension = True
            players.append(player)
        self.players = players
        self.actions = [None] * self.num_players
        self.prefetch = cfg.supervisor_cfg.prefetch
        self.prefetched = [False] * self.num_players
        self.device = envs.device
        # if joints are faulty, the online expert will reverse the force offset.
        self.disable_joints = cfg.disable_joints
        # make sure these match the values in ant.py
        self.force_offset = 0.5
        self.joint_mask = np.array([0.,1.,0.,1.,0.,1.,0.,1.])

    def prefetch_actions(self, states):
        """
        To optimize efficiency, this executes the NN forward pass for all states at once.
        Should be called once per timestep. 
        """
        for idx in range(self.num_players):
            # set ith tensor
            self.actions[idx] = self.players[idx].get_action(states, True)
            if self.disable_joints:
                self.actions[idx] -= torch.tensor(self.force_offset * self.joint_mask, dtype=torch.float32).to(self.device)
                self.actions[idx] = torch.clamp(self.actions[idx], -1., 1.)
            self.prefetched[idx] = True

    def get_action(self, state, player_idx, env_idx=None):
        if self.num_players == 1:
            player_idx = 0
        if self.prefetch:
            assert self.prefetched[player_idx]
            return self.actions[player_idx][env_idx].cpu()
        else:
            self.players[player_idx].has_batch_dimension = False
            act = self.players[player_idx].get_action(state, True).cpu()
            if self.disable_joints:
                act -= torch.tensor(self.force_offset * self.joint_mask, dtype=torch.float32)
                act = torch.clamp(act, -1., 1.)
            return act

