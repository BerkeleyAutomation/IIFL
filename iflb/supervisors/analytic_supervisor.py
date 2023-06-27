"""
A wrapper for envs that define the supervisor functions themselves
"""
from .base_supervisor import ParallelSupervisor 

class AnalyticSupervisor(ParallelSupervisor):
    def __init__(self, envs, cfg):
        self.vec_env = cfg.vec_env
        if self.vec_env:
            self.supervisor_fn = envs.human_action
        else:
            self.supervisor_fns = [env.human_action for env in envs]

    def get_action(self, state, player_idx, env_idx=0):
        if self.vec_env:
            return self.supervisor_fn(state=state, player_idx=player_idx, env_idx=env_idx)
        else:
            return self.supervisor_fns[env_idx](state)


