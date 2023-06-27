from .base_allocation import Allocation
import numpy as np

ONLINE_THRESH_WINDOW=1000

class CURAllocation(Allocation):
    def __init__(self, exp_cfg):
        super().__init__(exp_cfg)
        self.uncertainties = []
    """
    An allocation strategy that prioritizes robots by (C)onstraint Violations, (U)ncertainty, and (R)isk
    """
    def allocate(self, allocation_metrics):
        if 'C' in self.cfg.order:
            assert "constraint_violation" in allocation_metrics, "Agent {} does not provide the required metrics for {} allocation.".format(self.exp_cfg.agent, self.cfg.order)
        if 'R' in self.cfg.order:
            assert "risk" in allocation_metrics, "Agent {} does not provide the required metrics for {} allocation.".format(self.exp_cfg.agent, self.cfg.order)
        if 'U' in self.cfg.order:
            assert "uncertainty" in allocation_metrics, "Agent {} does not provide the required metrics for {} allocation.".format(self.exp_cfg.agent, self.cfg.order)
            if self.cfg.online_thresh:
                self.uncertainties.extend(allocation_metrics["uncertainty"])
                new_thresh = self.cfg.uncertainty_thresh
                if len(self.uncertainties) // self.exp_cfg.num_envs > ONLINE_THRESH_WINDOW // 10:
                    new_thresh = np.percentile(self.uncertainties[-ONLINE_THRESH_WINDOW*self.exp_cfg.num_envs:], 99.9)
        num_to_free = 0 # number of robots that have zero priority
        free_idx = set()
        priorities = list()
        num_violating = 0

        for i in range(self.exp_cfg.num_envs):
            constraint_violation = allocation_metrics["constraint_violation"][i]
            if 'R' in self.cfg.order:
                try:
                    risk = allocation_metrics["risk"][i][0]
                except:
                    risk = allocation_metrics["risk"][i]
            if 'U' in self.cfg.order:
                uncertainty = allocation_metrics["uncertainty"][i]

            if constraint_violation:
                num_violating += 1
                if self.cfg.warmup_penalty > allocation_metrics["time"]:
                    cv = -1 # penalize instead of prioritize constraint violation during warmup
                elif num_violating > int(self.cfg.cv_thresh * self.exp_cfg.num_humans):
                    # treat CVs beyond threshold as non-factors in prioritization
                    cv = 0
                else:
                    cv = 1
            else:
                cv = 0

            if 'R' in self.cfg.order and risk < self.cfg.risk_thresh:
                # treat risk below this threshold as 0 risk
                risk = 0

            if 'U' in self.cfg.order and uncertainty < self.cfg.uncertainty_thresh:
                # treat uncertainty below this threshold as 0 uncertainty
                uncertainty = 0

            if ('C' not in self.cfg.order or cv <= 0) and ('R' not in self.cfg.order or risk == 0) and ('U' not in self.cfg.order or uncertainty == 0):
                # free human
                if i not in free_idx:
                    num_to_free += 1
                    free_idx.add(i)

            priority = list()
            for elem in self.cfg.order:
                if elem == 'C':
                    priority.append(cv)
                if elem == 'R':
                    priority.append(risk)
                if elem == 'U':
                    priority.append(uncertainty)
            priority.append(np.random.random())
            priority = tuple(priority)
            priorities.append(priority)
        if self.cfg.online_thresh:
            self.cfg.uncertainty_thresh = new_thresh
        env_priorities = sorted(range(len(priorities)), key = lambda x: priorities[x], reverse=True)
        if self.cfg.free_humans:
            return env_priorities[:self.exp_cfg.num_envs - num_to_free]
        else:
            return env_priorities
